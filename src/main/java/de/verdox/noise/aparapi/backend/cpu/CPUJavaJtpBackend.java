package de.verdox.noise.aparapi.backend.cpu;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import de.verdox.util.HardwareUtil;

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
        final int L     = vectorized ? HardwareUtil.getVectorLaneLength() : 1;
        final int Wv    = vectorized ? (width + L - 1) / L : width;

        if (optimizeCache && this.cacheOptKernels != null) {

            List<Future<?>> futures = new ArrayList<>(1024);

            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                final int dz = Math.min(slabDepth, depth - zStart);

                for (int yStart = 0; yStart < height; yStart += rowsPerTask) {
                    final int rows = Math.min(rowsPerTask, height - yStart);

                    final int finalYStart = yStart;
                    final int finalZStart = zStart;

                    Future<?> future = cacheOptPool.submit(() -> {
                        AbstractSimplexNoise3DAparapiKernel kernel = cacheOptKernels.get();
                        float[] slab = slabsPerThread.get();

                        // Puffergröße bleibt "echte" Elemente
                        final int neededElemsForSlab = width * rows * dz;
                        if (neededElemsForSlab > slab.length) {
                            throw new IllegalArgumentException("slab has wrong size");
                        }
                        kernel.bindOutput(slab);

                        // Lokale Parameter (wie im Seq-Backend)
                        kernel.setParameters(
                                x0,
                                y0 + finalYStart * f,
                                z0 + finalZStart * f,
                                width, rows, dz,
                                f,
                                0 // base innerhalb des Slabs
                        );

                        // WICHTIG: globale Range = Vektor-Elemente, nicht echte Elemente
                        final int elemsToCompute = Wv * rows * dz;

                        kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ); // jeder Task läuft sequenziell
                        kernel.execute(Range.create(elemsToCompute, 1));

                        // Slab -> result zurückkopieren (nur width*rows echte Werte je z-Schicht)
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

            for (Future<?> ftr : futures) {
                try {
                    ftr.get();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                } catch (ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }
        } else {
            // Direkt in das Zielarray schreiben
            kernel.bindOutput(result);
            kernel.setExecutionMode(executionMode);

            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                final int dz = Math.min(slabDepth, depth - zStart);

                // Vector-global size für diesen z-Block
                final int global = Wv * height * dz;

                // base zeigt in das große Zielarray auf die erste Schicht dieses Blocks
                kernel.setParameters(
                        x0, y0, z0 + zStart * f,
                        width, height, dz,
                        f,
                        zStart * plane
                );

                // im JTP-Mode keine lokale Größe
                kernel.execute(Range.create(global));
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
