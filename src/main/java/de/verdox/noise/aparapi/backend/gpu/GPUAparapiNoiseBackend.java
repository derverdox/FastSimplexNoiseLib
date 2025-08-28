package de.verdox.noise.aparapi.backend.gpu;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import de.verdox.noise.AparapiBackendUtil;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.OpenCLTuner;
import de.verdox.noise.aparapi.backend.AparapiNoiseBackend;
import de.verdox.noise.aparapi.kernel.cpu.CPUScalarSimplexNoiseKernel;
import de.verdox.util.FormatUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class GPUAparapiNoiseBackend<KERNEL extends CPUScalarSimplexNoiseKernel> extends AparapiNoiseBackend<KERNEL> {
    protected final NoiseBackendBuilder.GPUNoiseBackendBuilder params;

    public GPUAparapiNoiseBackend(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
        super(preferredDevice, params.getNoiseCalculationMode(), result, width, height, depth);
        this.params = params;
    }

    @Override
    public void logSetup() {
        OpenCLDevice openCLDevice = (OpenCLDevice) preferredDevice;
        final int maxWG = openCLDevice.getMaxWorkGroupSize();
        final int[] maxIt = openCLDevice.getMaxWorkItemSize();
        final int CUs = openCLDevice.getMaxComputeUnits();
        final long lmem = openCLDevice.getLocalMemSize();
        System.out.println();

        OpenCLTuner.Plan plan = OpenCLTuner.plan(openCLDevice, width, height, depth, false, true);

        System.out.println("=== GPUOpenCLBackend ===");
        System.out.println("Device: " + openCLDevice.getName() + " (" + openCLDevice.getType() + ") | Vendor: " + openCLDevice
                .getOpenCLPlatform().getVersion());
        System.out.println("Compute Units: " + CUs);
        System.out.println("> L1 Cache: " + FormatUtil.formatBytes2(lmem));
        System.out.println("> Warp Size: " + AparapiBackendUtil.detectPreferredWarp(openCLDevice) + " threads");
        System.out.println(plan);


        System.out.println("Allocated: " + FormatUtil.formatBytes2((long) width * height * depth * Float.BYTES) + " / " + FormatUtil.formatBytes2(openCLDevice.getMaxMemAllocSize()));
        System.out.println("MaxWorkGroupSize: " + maxWG + " | MaxWorkItemSizes: " + Arrays.toString(maxIt));

        if (use1DIndexing) {
            System.out.printf("Mode: 3D | local=(%d,%d,%d) | slabDepth=%d | dims=(%d,%d,%d)%n",
                    localX, localY, localZ, slabDepth, width, height, depth);
        } else {
            System.out.printf("Mode: 1D | local=%d | slabDepth=%d | dims=(%d,%d,%d)%n",
                    local1D, slabDepth, width, height, depth);
        }
        System.out.println("================================");
    }

    /**
     * The simple way. Passes one big buffer to the gpu and lets it calculate the rest.
     */
    public static class Simple extends GPUAparapiNoiseBackend<CPUScalarSimplexNoiseKernel.Simple> {
        protected OpenCLTuner.Plan plan;
        protected Range range;

        public Simple(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
            super(preferredDevice, params, result, width, height, depth);
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Simple setup() {
            //TODO: Also allow for 2D setup
            plan = OpenCLTuner.plan((OpenCLDevice) preferredDevice, width, height, depth, !use1DIndexing, true);
            range = plan.toRange();
            this.kernel = createKernel();
            return this.kernel;
        }

        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            kernel.setExplicit(true);
            kernel.bindOutput(result);
            kernel.setExplicit(true);
            kernel.setParameters(0, 0, 0, width, height, depth, frequency, 0);
            kernel.execute(range);
            kernel.get(result);
        }

        @Override
        public void generate3DNoise3DIndexed(float x0, float y0, float z0, float frequency) {
            //TODO:
            throw new UnsupportedOperationException();
        }

        @Override
        public void generate2DNoise1DIndexed(float x0, float y0, float frequency) {
            //TODO:
            throw new UnsupportedOperationException();
        }

        @Override
        public void generate2DNoise2DIndexed(float x0, float y0, float frequency) {
            //TODO:
            throw new UnsupportedOperationException();
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Simple createKernel() {
            return use1DIndexing ? new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing1D(calculationMode) : new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing3D(calculationMode);
        }
    }

    public static class Batched extends GPUAparapiNoiseBackend<CPUScalarSimplexNoiseKernel.Batched> {
        public static final int TILE = 256;
        private final List<Tile> tiles = new ArrayList<>();

        public Batched(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
            super(preferredDevice, params, result, width, height, depth);
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Batched setup() {
            // Kernel erstellen & auf explizite Transfers stellen
            this.kernel = createKernel();
            kernel.setExplicit(true);
            kernel.bindOutput(result); // write-only → kein put()

            // Tiles vorab planen (1D-Range; local=256, global gepaddet)
            for (int zBase = 0; zBase < depth; zBase += TILE) {
                int td = Math.min(TILE, depth - zBase);
                for (int yBase = 0; yBase < height; yBase += TILE) {
                    int th = Math.min(TILE, height - yBase);
                    for (int xBase = 0; xBase < width; xBase += TILE) {
                        int tw = Math.min(TILE, width - xBase);

                        OpenCLTuner.Plan plan = OpenCLTuner.plan((OpenCLDevice) preferredDevice, tw, th, td,
                                /*prefer3D=*/true,
                                /*alignBaseIndexTo32=*/true);
                        Range tileRange = plan.toRange();

                        int baseIndex = xBase + yBase * width + zBase * width * height;

                        tiles.add(new Tile(xBase, yBase, zBase, tw, th, td, baseIndex, tileRange));
                    }
                }
            }
            return this.kernel;
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Batched createKernel() {
            return use1DIndexing ? new CPUScalarSimplexNoiseKernel.Batched.Noise3DIndexing1D(calculationMode) : new CPUScalarSimplexNoiseKernel.Batched.Noise3DIndexing3D(calculationMode);
        }

        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            // Tiles rechnen; kein get() im Loop
            for (Tile t : tiles) {
                float bx = x0 + t.bx * frequency;
                float by = y0 + t.by * frequency;
                float bz = z0 + t.bz * frequency;

                kernel.setParameters(bx, by, bz, t.tw, t.th, t.td, frequency, t.baseIndex);
                kernel.globalWidth = width;
                kernel.globalHeight = height;
                kernel.execute(t.range);
            }
            // EIN get() am Ende (oder blockweise, falls gewünscht)
            kernel.get(result);
        }

        @Override
        public void generate3DNoise3DIndexed(float x0, float y0, float z0, float frequency) {
            //TODO:
            throw new UnsupportedOperationException();
        }

        @Override
        public void generate2DNoise1DIndexed(float x0, float y0, float frequency) {
            //TODO:
            throw new UnsupportedOperationException();
        }

        @Override
        public void generate2DNoise2DIndexed(float x0, float y0, float frequency) {
            //TODO:
            throw new UnsupportedOperationException();
        }

        private record Tile(int bx, int by, int bz, int tw, int th, int td,
                            int baseIndex, Range range) {
        }
    }
}
