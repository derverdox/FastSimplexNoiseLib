package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import de.verdox.util.FormatUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class CPUJavaJtpBackend extends CPUJavaNoiseBackend {
    public CPUJavaJtpBackend(Device preferred, boolean vectorized, boolean optimizeCache, float[] result, int w, int h, int d) {
        super(preferred, vectorized, optimizeCache, result, w, h, d);
        this.executionMode = Kernel.EXECUTION_MODE.JTP;
    }

    protected AbstractSimplexNoise3DAparapiKernel setup() {
        this.use3DRange = false;
        this.local1D = 0;
        this.slabDepth = Math.min(32, depth);
        return super.setup();
    }

    @Override
    protected int threadsUsed() {
        return optimizeCache ? physicalProcessors : Runtime.getRuntime().availableProcessors();
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float f) {
        final int plane = width * height;
        if (optimizeCache && this.cacheOptKernels != null) {

            List<Future<?>> futures = new ArrayList<>(1024);

            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                final int dz = Math.min(slabDepth, depth - zStart);

                for (int yStart = 0; yStart < height; yStart += rowsPerTask) {
                    final int rows = Math.min(rowsPerTask, height - yStart);

                    int finalYStart = yStart;
                    int finalZStart = zStart;

                    Future<?> future = cacheOptPool.submit(() -> {
                        AbstractSimplexNoise3DAparapiKernel kernel = cacheOptKernels.get();
                        float[] slab = slabsPerThread.get();

                        int needed = width * rows * dz;
                        if (needed > slab.length) {
                            throw new IllegalArgumentException("slab has wrong size");
                        }
                        kernel.bindOutput(slab);

                        kernel.setParameters(
                                x0,
                                y0 + finalYStart * f,           // world-space Y offset
                                z0 + finalZStart * f,           // world-space Z offset
                                width, rows, dz,     // local slab dims
                                f,
                                0                          // base=0 inside slab
                        );

                        kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
                        kernel.execute(Range.create(needed, 1));

                        // Copy slab back into the big result (z-slice by z-slice)
                        final int rowStride = width * rows;
                        for (int z = 0; z < dz; z++) {
                            final int src = z * rowStride;
                            final int dst = (finalZStart + z) * plane + finalYStart * width;
                            System.arraycopy(slab, src, result, dst, rowStride);
                        }
                    });
                    futures.add(future);
                }
            }

            for (int i = 0; i < futures.size(); i++) {
                try {
                    futures.get(i).get();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                } catch (ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }
        } else {
            kernel.bindOutput(result);
            kernel.setExecutionMode(executionMode);
            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                int dz = Math.min(slabDepth, depth - zStart);
                int slice = plane * dz;
                Range range = Range.create(slice); // keine local size im JTP
                kernel.setParameters(x0, y0, z0 + zStart * f, width, height, dz, f, zStart * plane);
                kernel.setExecutionMode(executionMode);
                kernel.execute(range);
            }
        }
    }

    @Override
    public void logSetup() {
        System.out.println();
        System.out.println("=== CPU-Parallel Backend ===");
        super.logSetup();
    }
}