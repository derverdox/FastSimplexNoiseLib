package de.verdox.noise.aparapi.backend;


import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import de.verdox.noise.AparapiBackendUtil;
import de.verdox.noise.aparapi.kernel.scalar.AbstractScalarSimplexNoise3DAparapiKernel;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel1D;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel3D;

import java.util.Arrays;

public class GPUOpenCLBackend extends AparapiNoiseBackend<AbstractScalarSimplexNoise3DAparapiKernel> {
    protected final OpenCLDevice openCLDevice;
    private boolean canDirectWrite;

    private float[] scratch;
    private int dzCap;

    public GPUOpenCLBackend(OpenCLDevice openCLDevice, float[] result, int width, int height, int depth) {
        super(openCLDevice, result, width, height, depth);
        this.openCLDevice = openCLDevice;
        this.executionMode = Kernel.EXECUTION_MODE.GPU;
    }

    // ----------- Public API -----------

    /**
     * Rechnet das 3D-Noise-Volumen in 'result' (Z-major).
     */
    public void generate(float x0, float y0, float z0, float frequency) {

        if (use3DRange) generate3D(x0, y0, z0, frequency);
        else generate1D(x0, y0, z0, frequency);
    }

    public void generate3D(float x0, float y0, float z0, float frequency) {
        final int plane = width * height;

        if (canDirectWrite) {
            kernel.bindOutput(result);
            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                int dz = Math.min(slabDepth, depth - zStart);
                Range range = Range.create3D(width, height, dz);
                kernel.setParameters(
                        x0, y0, z0 + zStart * frequency,
                        width, height, dz, frequency,
                        zStart * plane
                );
                kernel.execute(range);
            }
        } else {
            // Staging: in scratch rechnen und zurückkopieren
            kernel.bindOutput(scratch);
            for (int zStart = 0; zStart < depth; zStart += dzCap) {
                int dz = Math.min(dzCap, depth - zStart);
                Range range = Range.create3D(width, height, dz);

                kernel.setParameters(x0, y0, z0 + zStart * frequency, width, height, dz, frequency, /*argBase=*/0);
                kernel.setExecutionMode(executionMode);
                kernel.execute(range);

                int sliceElems = plane * dz;
                System.arraycopy(scratch, 0, result, zStart * plane, sliceElems);
            }
        }
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float frequency) {
        final int plane = width * height;

        if (canDirectWrite) {
            kernel.bindOutput(result);
            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                int dz = Math.min(slabDepth, depth - zStart);
                int sliceElems = plane * dz;

                int g = (local1D > 1) ? roundUp(sliceElems, local1D) : sliceElems;
                Range range = (local1D > 1) ? Range.create(g, local1D) : Range.create(g);

                kernel.setParameters(x0, y0, z0 + zStart * frequency, width, height, dz, frequency, zStart * plane);
                kernel.setExecutionMode(executionMode);
                kernel.execute(range);
            }
        } else {
            kernel.bindOutput(scratch);
            for (int zStart = 0; zStart < depth; zStart += dzCap) {
                int dz = Math.min(dzCap, depth - zStart);
                int sliceElems = plane * dz;

                int g = (local1D > 1) ? roundUp(sliceElems, local1D) : sliceElems;
                Range range = (local1D > 1) ? Range.create(g, local1D) : Range.create(g);

                kernel.setParameters(x0, y0, z0 + zStart * frequency, width, height, dz, frequency, /*argBase=*/0);
                kernel.setExecutionMode(executionMode);
                kernel.execute(range);

                System.arraycopy(scratch, 0, result, zStart * plane, sliceElems);
            }
        }
    }

    // ----------- Setup / Heuristiken -----------

    /**
     * Vendor-aware Setup (NVIDIA=32, AMD=64), 3D bevorzugt, ohne Guards.
     */
    @Override
    protected AbstractScalarSimplexNoise3DAparapiKernel setup() {
        this.canDirectWrite = checkIfCanWriteDirectly();
        final int maxWG = openCLDevice.getMaxWorkGroupSize();
        final int[] maxIt = openCLDevice.getMaxWorkItemSize();
        final int CUs = openCLDevice.getMaxComputeUnits();

        final int warp = AparapiBackendUtil.detectPreferredWarp(openCLDevice);

        int[] local3D = AparapiBackendUtil.pickLocal3D(width, height, depth, maxWG, maxIt, warp);
        if (local3D != null) {
            use3DRange = true;
            localX = local3D[0];
            localY = local3D[1];
            localZ = local3D[2];
            slabDepth = AparapiBackendUtil.pickDzFor3D(width, height, depth, localX, localY, localZ, CUs);
            kernel = new ScalarSimplexNoise3DKernel3D();
        } else {
            use3DRange = false;
            local1D = AparapiBackendUtil.pickLocal1D(maxWG, warp);
            slabDepth = AparapiBackendUtil.pickDzFor1D(width, height, depth, local1D, CUs);
            kernel = new ScalarSimplexNoise3DKernel1D();
        }

        // ---- Staging-Setup, falls direkte Bindung nicht geht
        if (!canDirectWrite) {
            long maxAlloc = openCLDevice.getMaxMemAllocSize();   // oft ~2 GB
            long globalMem = openCLDevice.getGlobalMemSize();     // z. B. 11–24 GB

            long target = 128L * 1024 * 1024;                     // 128 MB (gute Default-Chunkgröße)
            long capA = (long) (maxAlloc * 0.80);               // 80% vom maxAlloc
            long capB = (long) (globalMem * 0.25);               // 25% vom VRAM (zusätzlicher Schutz)
            long bytes = Math.min(target, Math.min(capA, capB));

            int plane = width * height;
            long maxFloats = Math.max(plane, bytes / Float.BYTES);
            int scratchElems = (int) ((maxFloats / plane) * plane); // Vielfaches von plane
            if (scratchElems < plane) {
                throw new IllegalStateException("One plane doesn't fit into device alloc limit");
            }
            scratch = new float[scratchElems];
            dzCap = Math.max(1, scratchElems / plane);
        }
        return kernel;
    }

    private boolean checkIfCanWriteDirectly() {
        long maxAlloc = openCLDevice.getMaxMemAllocSize();
        long resultBytes = (long) width * height * depth * Float.BYTES;
        double safety = 0.80;
        return resultBytes <= (long) (maxAlloc * safety);
    }

    @Override
    public void logSetup() {
        final int maxWG = openCLDevice.getMaxWorkGroupSize();
        final int[] maxIt = openCLDevice.getMaxWorkItemSize();
        final int CUs = openCLDevice.getMaxComputeUnits();
        final long lmem = openCLDevice.getLocalMemSize();
        System.out.println("=== GPUOpenCLBackend ===");
        System.out.println("Device: " + openCLDevice.getName() + " (" + openCLDevice.getType() + ") | Vendor: " + openCLDevice
                .getOpenCLPlatform().getVersion());
        System.out.println("Write Mode: " + (canDirectWrite ? "Direct" : "Non-Direct"));
        if(canDirectWrite) {
            System.out.println("Allocated: " + AparapiBackendUtil.formatBytes((long) width * height * depth * Float.BYTES) + " / " + AparapiBackendUtil.formatBytes(openCLDevice.getMaxMemAllocSize()));
        }
        else {
            System.out.println("Allocated: " + AparapiBackendUtil.formatBytes((long) scratch.length * Float.BYTES) + " / " + AparapiBackendUtil.formatBytes(openCLDevice.getMaxMemAllocSize()));
        }

        System.out.println("OpenCL: " + openCLDevice.getOpenCLPlatform().getVersion());
        System.out.println("MaxWorkGroupSize: " + maxWG + " | MaxWorkItemSizes: " + Arrays.toString(maxIt));
        System.out.println("MaxComputeUnits: " + CUs + " | LocalMemSize: " + AparapiBackendUtil.formatBytes(lmem));

        if (use3DRange) {
            System.out.printf("Mode: 3D | local=(%d,%d,%d) | slabDepth=%d | dims=(%d,%d,%d)%n",
                    localX, localY, localZ, slabDepth, width, height, depth);
        } else {
            System.out.printf("Mode: 1D | local=%d | slabDepth=%d | dims=(%d,%d,%d)%n",
                    local1D, slabDepth, width, height, depth);
        }
        System.out.println("================================");
    }
}