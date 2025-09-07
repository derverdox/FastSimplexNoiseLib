package de.verdox.noise.aparapi.backend.gpu;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import de.verdox.noise.AparapiBackendUtil;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.OpenCLTuner;
import de.verdox.noise.aparapi.backend.AparapiNoiseBackend;
import de.verdox.noise.aparapi.kernel.cpu.CPUScalarSimplexNoiseKernel;
import de.verdox.util.FormatUtil;
import de.verdox.util.LODUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class GPUAparapiNoiseBackend<KERNEL extends CPUScalarSimplexNoiseKernel> extends AparapiNoiseBackend<KERNEL> {
    protected final NoiseBackendBuilder.GPUNoiseBackendBuilder params;

    public GPUAparapiNoiseBackend(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
        super(preferredDevice, params.getNoiseCalculationMode(), result, width, height, depth);
        this.params = params;
    }

    public GPUAparapiNoiseBackend(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int depth) {
        super(preferredDevice, params.getNoiseCalculationMode(), result, width, depth);
        this.params = params;
    }

    @Override
    public void logSetup() {
        OpenCLDevice dev = (OpenCLDevice) preferredDevice;
        final int maxWG = dev.getMaxWorkGroupSize();
        final int[] maxIt = dev.getMaxWorkItemSize();
        final int CUs = dev.getMaxComputeUnits();
        final long lmem = dev.getLocalMemSize();
        System.out.println();

        // LOD-Dimensionen für Logging/Plan
        final int lod = params.getLodLevel();
        final var lodMode = params.getLodMode();

        int Wlod, Hlod, Dlod;
        OpenCLTuner.Plan planLog;

        if (params.is3DMode()) {
            LODUtil.LOD3DParams lp = LODUtil.computeLOD3D(width, height, depth, 0f, 0f, 0f, 1f, lod, lodMode);
            Wlod = lp.widthLOD(); Hlod = lp.heightLOD(); Dlod = lp.depthLOD();
            planLog = OpenCLTuner.plan(dev, Wlod, Hlod, Dlod, /*prefer3D*/ true, /*align*/ true);
        } else {
            LODUtil.LOD2DParams lp = LODUtil.computeLOD2D(width, depth, 0f, 0f, 1f, lod, lodMode);
            Wlod = lp.widthLOD(); Hlod = 1; Dlod = lp.depthLOD();
            planLog = OpenCLTuner.plan2D(dev, Wlod, Dlod, /*prefer2D*/ true, /*align*/ true);
        }

        System.out.println("=== GPUOpenCLBackend ===");
        System.out.println("Device: " + dev.getName() + " (" + dev.getType() + ") | Vendor: " + dev.getOpenCLPlatform().getVersion());
        System.out.println("Compute Units: " + CUs);
        System.out.println("> L1 Cache: " + FormatUtil.formatBytes2(lmem));
        System.out.println("> Warp Size: " + AparapiBackendUtil.detectPreferredWarp(dev) + " threads");
        System.out.println(planLog);

        long allocElems = (long) Wlod * Hlod * Dlod;
        System.out.println("Allocated (LOD dims): " + FormatUtil.formatBytes2(allocElems * Float.BYTES) +
                " / " + FormatUtil.formatBytes2(dev.getMaxMemAllocSize()));
        System.out.println("MaxWorkGroupSize: " + maxWG + " | MaxWorkItemSizes: " + Arrays.toString(maxIt));

        if (use1DIndexing) {
            System.out.printf("Mode: 1D | dims(LOD)=(%d,%d,%d)%n", Wlod, Hlod, Dlod);
        } else {
            System.out.printf("Mode: 3D/2D | dims(LOD)=(%d,%d,%d)%n", Wlod, Hlod, Dlod);
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

        public Simple(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int depth) {
            super(preferredDevice, params, result, width, depth);
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Simple setup() {
            // LOD ist fix → Range einmalig mit LOD-Dimensionen planen
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();

            if (params.is3DMode()) {
                LODUtil.LOD3DParams lp = LODUtil.computeLOD3D(width, height, depth, 0f, 0f, 0f, 1f, lod, lodMode);
                plan  = OpenCLTuner.plan((OpenCLDevice) preferredDevice,
                        lp.widthLOD(), lp.heightLOD(), lp.depthLOD(),
                        /*prefer3D*/ !use1DIndexing,
                        /*align*/ true);
            } else {
                LODUtil.LOD2DParams lp = LODUtil.computeLOD2D(width, depth, 0f, 0f, 1f, lod, lodMode);
                plan  = OpenCLTuner.plan2D((OpenCLDevice) preferredDevice,
                        lp.widthLOD(), lp.depthLOD(),
                        /*prefer2D*/ !use1DIndexing,
                        /*align*/ true);
            }
            range = plan.toRange();
            this.kernel = createKernel();
            return this.kernel;
        }

        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            final var lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, params.getLodLevel(), params.getLodMode());
            kernel.setExplicit(true);
            kernel.bindOutput(result);
            kernel.setParameters(lp.baseX(), lp.baseY(), lp.baseZ(),
                    lp.widthLOD(), lp.heightLOD(), lp.depthLOD(),
                    lp.frequencyLOD(),
                    0, params.getSeed());
            kernel.execute(range); // Range wurde im setup() mit LOD-Dims geplant
            kernel.get(result);
        }

        @Override
        public void generate3DNoise3DIndexed(float x0, float y0, float z0, float frequency) {
            final var lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, params.getLodLevel(), params.getLodMode());
            kernel.setExplicit(true);
            kernel.bindOutput(result);
            kernel.setParameters(lp.baseX(), lp.baseY(), lp.baseZ(),
                    lp.widthLOD(), lp.heightLOD(), lp.depthLOD(),
                    lp.frequencyLOD(),
                    0, params.getSeed());
            kernel.execute(range);
            kernel.get(result);
        }

        @Override
        public void generate2DNoise1DIndexed(float x0, float y0, float frequency) {
            final var lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, params.getLodLevel(), params.getLodMode());
            kernel.setExplicit(true);
            kernel.bindOutput(result);
            kernel.setParameters(
                    lp.baseX(),
                    /*baseY*/ 0f,
                    /*baseZ*/ lp.baseZ(),
                    lp.widthLOD(), /*height*/ 1, lp.depthLOD(),
                    lp.frequencyLOD(),
                    /*baseIndex*/ 0,
                    params.getSeed()
            );
            kernel.execute(range);
            kernel.get(result);
        }

        @Override
        public void generate2DNoise2DIndexed(float x0, float y0, float frequency) {
            final var lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, params.getLodLevel(), params.getLodMode());
            kernel.setExplicit(true);
            kernel.bindOutput(result);
            kernel.setParameters(
                    lp.baseX(),
                    0f,
                    lp.baseZ(),
                    lp.widthLOD(), 1, lp.depthLOD(),
                    lp.frequencyLOD(),
                    0,
                    params.getSeed()
            );
            kernel.execute(range);
            kernel.get(result);
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Simple createKernel() {
            if (use1DIndexing) {
                return params.is3DMode()
                        ? new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing1D(calculationMode)
                        : new CPUScalarSimplexNoiseKernel.Simple.Noise2DIndexing1D(calculationMode);
            } else {
                return params.is3DMode()
                        ? new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing3D(calculationMode)
                        : new CPUScalarSimplexNoiseKernel.Simple.Noise2DIndexing2D(calculationMode);
            }
        }
    }

    public static class Batched extends GPUAparapiNoiseBackend<CPUScalarSimplexNoiseKernel.Batched> {
        public static final int TILE = 256;
        private final List<Tile> tiles = new ArrayList<>();

        public Batched(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
            super(preferredDevice, params, result, width, height, depth);
        }

        public Batched(OpenCLDevice preferredDevice, NoiseBackendBuilder.GPUNoiseBackendBuilder params, float[] result, int width, int depth) {
            super(preferredDevice, params, result, width, depth);
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Batched setup() {
            // Kernel erstellen & auf explizite Transfers stellen
            this.kernel = createKernel();
            kernel.setExplicit(true);
            kernel.bindOutput(result); // write-only → kein put()

            // Tiles anhand der LOD-Dimensionen erzeugen
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();

            tiles.clear();

            if (params.is3DMode()) {
                LODUtil.LOD3DParams lp = LODUtil.computeLOD3D(width, height, depth, 0f, 0f, 0f, 1f, lod, lodMode);
                final int W = lp.widthLOD(), H = lp.heightLOD(), D = lp.depthLOD();

                for (int zBase = 0; zBase < D; zBase += TILE) {
                    int td = Math.min(TILE, D - zBase);
                    for (int yBase = 0; yBase < H; yBase += TILE) {
                        int th = Math.min(TILE, H - yBase);
                        for (int xBase = 0; xBase < W; xBase += TILE) {
                            int tw = Math.min(TILE, W - xBase);

                            OpenCLTuner.Plan p = OpenCLTuner.plan((OpenCLDevice) preferredDevice, tw, th, td,
                                    /*prefer3D*/ true, /*align*/ true);
                            Range r = p.toRange();

                            int baseIndex = xBase + yBase * W + zBase * W * H;
                            tiles.add(new Tile(xBase, yBase, zBase, tw, th, td, baseIndex, r));
                        }
                    }
                }
            } else {
                LODUtil.LOD2DParams lp = LODUtil.computeLOD2D(width, depth, 0f, 0f, 1f, lod, lodMode);
                final int W = lp.widthLOD(), D = lp.depthLOD();

                for (int zBase = 0; zBase < D; zBase += TILE) {
                    int td = Math.min(TILE, D - zBase);
                    for (int xBase = 0; xBase < W; xBase += TILE) {
                        int tw = Math.min(TILE, W - xBase);

                        OpenCLTuner.Plan p = OpenCLTuner.plan2D((OpenCLDevice) preferredDevice, tw, td,
                                /*prefer2D*/ true, /*align*/ true);
                        Range r = p.toRange();

                        int baseIndex = xBase + zBase * W; // H=1
                        tiles.add(new Tile(xBase, /*yBase*/ 0, zBase, tw, /*th*/ 1, td, baseIndex, r));
                    }
                }
            }
            return this.kernel;
        }

        @Override
        protected CPUScalarSimplexNoiseKernel.Batched createKernel() {
            return use1DIndexing
                    ? (params.is3DMode()
                    ? new CPUScalarSimplexNoiseKernel.Batched.Noise3DIndexing1D(calculationMode)
                    : new CPUScalarSimplexNoiseKernel.Batched.Noise2DIndexing1D(calculationMode))
                    : (params.is3DMode()
                    ? new CPUScalarSimplexNoiseKernel.Batched.Noise3DIndexing3D(calculationMode)
                    : new CPUScalarSimplexNoiseKernel.Batched.Noise2DIndexing2D(calculationMode));
        }

        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            final var lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, params.getLodLevel(), params.getLodMode());
            // globalWidth/-Height für den Batched-Kernel gemäß LOD-Dims
            if (kernel instanceof CPUScalarSimplexNoiseKernel.Batched k) {
                k.globalWidth = lp.widthLOD();
                k.globalHeight = lp.heightLOD();
            }

            for (Tile t : tiles) {
                float bx = lp.baseX() + t.bx * lp.frequencyLOD();
                float by = lp.baseY() + t.by * lp.frequencyLOD();
                float bz = lp.baseZ() + t.bz * lp.frequencyLOD();

                kernel.setParameters(bx, by, bz, t.tw, t.th, t.td, lp.frequencyLOD(), t.baseIndex, params.getSeed());
                kernel.execute(t.range);
            }
            kernel.get(result);
        }

        @Override
        public void generate3DNoise3DIndexed(float x0, float y0, float z0, float frequency) {
            final var lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, params.getLodLevel(), params.getLodMode());
            if (kernel instanceof CPUScalarSimplexNoiseKernel.Batched k) {
                k.globalWidth = lp.widthLOD();
                k.globalHeight = lp.heightLOD();
            }

            for (Tile t : tiles) {
                float bx = lp.baseX() + t.bx * lp.frequencyLOD();
                float by = lp.baseY() + t.by * lp.frequencyLOD();
                float bz = lp.baseZ() + t.bz * lp.frequencyLOD();

                kernel.setParameters(bx, by, bz, t.tw, t.th, t.td, lp.frequencyLOD(), t.baseIndex, params.getSeed());
                kernel.execute(t.range);
            }
            kernel.get(result);
        }

        @Override
        public void generate2DNoise1DIndexed(float x0, float y0, float frequency) {
            final var lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, params.getLodLevel(), params.getLodMode());

            for (Tile t : tiles) {
                float bx = lp.baseX() + t.bx * lp.frequencyLOD();
                float bz = lp.baseZ() + t.bz * lp.frequencyLOD();

                kernel.setParameters(
                        bx,
                        0f,
                        bz,
                        t.tw, /*height*/ t.th, /*depth*/ t.td,  // th ist 1
                        lp.frequencyLOD(),
                        t.baseIndex,
                        params.getSeed()
                );
                kernel.execute(t.range);
            }
            kernel.get(result);
        }

        @Override
        public void generate2DNoise2DIndexed(float x0, float y0, float frequency) {
            final var lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, params.getLodLevel(), params.getLodMode());

            for (Tile t : tiles) {
                float bx = lp.baseX() + t.bx * lp.frequencyLOD();
                float bz = lp.baseZ() + t.bz * lp.frequencyLOD();

                kernel.setParameters(
                        bx,
                        0f,
                        bz,
                        t.tw, /*height*/ t.th, /*depth*/ t.td,
                        lp.frequencyLOD(),
                        t.baseIndex,
                        params.getSeed()
                );
                kernel.execute(t.range);
            }
            kernel.get(result);
        }
    }

    private record Tile(int bx, int by, int bz, int tw, int th, int td,
                        int baseIndex, Range range) {
    }
}
