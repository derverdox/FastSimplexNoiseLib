package de.verdox.noise.aparapi.backend.cpu;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.aparapi.backend.AparapiNoiseBackend;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoiseKernel;
import de.verdox.noise.aparapi.kernel.cpu.CPUScalarSimplexNoiseKernel;
import de.verdox.noise.aparapi.kernel.cpu.CPUVectorSimplexNoiseKernel;
import de.verdox.util.HardwareUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public abstract class CPUJavaAparapiNoiseBackend extends AparapiNoiseBackend<AbstractSimplexNoiseKernel> {
    protected final NoiseBackendBuilder.CPUNoiseBackendBuilder params;


    public CPUJavaAparapiNoiseBackend(Device preferredDevice,
                                      NoiseBackendBuilder.CPUNoiseBackendBuilder params,
                                      float[] result, int width, int height, int depth) {
        super(preferredDevice, params.getNoiseCalculationMode(), result, width, height, depth);
        this.params = params;
    }

    public int threads() {
        return switch (params.getParallelismMode()) {
            case SEQUENTIAL -> 1;
            case PARALLELISM_CORES -> HardwareUtil.getPhysicalProcessorCount();
            case PARALLELISM_THREADS -> Runtime.getRuntime().availableProcessors();
        };
    }

    @Override
    protected AbstractSimplexNoiseKernel createKernel() {
        if(params.isVectorize()) {
            if(use1DIndexing) {
                return new CPUVectorSimplexNoiseKernel.Simple.Noise3DIndexing1D(params.getNoiseCalculationMode());
            }
            else {
                return new CPUVectorSimplexNoiseKernel.Simple.Noise3DIndexing3D(params.getNoiseCalculationMode());
            }
        }
        else {
            if(use1DIndexing) {
                return new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing1D(params.getNoiseCalculationMode());
            }
            else {
                return new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing3D(params.getNoiseCalculationMode());
            }
        }
    }

    /**
     * Tries to only use L1 and L2 cache of the processor
     */
    public static class CacheOnly extends CPUJavaAparapiNoiseBackend {
        protected int rowsPerTask;
        protected float[] cacheSlab;

        protected ThreadLocal<AbstractSimplexNoiseKernel> cacheOptKernels;
        protected ThreadLocal<float[]> slabsPerThread;
        protected ExecutorService cacheOptPool;
        private int maxSlabElems;

        public CacheOnly(Device preferredDevice, NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
            super(preferredDevice, params, result, width, height, depth);
        }

        @Override
        protected AbstractSimplexNoiseKernel setup() {
            int threads = threads();
            long l3 = HardwareUtil.readCaches().l3.sizeBytes();

            if(use1DIndexing) {
                this.slabDepth = pickDzFor1D(width, height, depth, Float.BYTES, threads, l3);
                this.rowsPerTask = pickRowsPerTaskFor1DWithDz(width, slabDepth, Float.BYTES, threads, l3);
                maxSlabElems = width * rowsPerTask * slabDepth;

                cacheSlab = new float[maxSlabElems];
                if(cacheOptPool == null) {
                    cacheOptPool = Executors.newFixedThreadPool(threads, r -> {
                        Thread t = new Thread(r);
                        t.setDaemon(true);
                        return t;
                    });
                }

                this.cacheOptKernels = ThreadLocal.withInitial(this::createKernel);
                this.slabsPerThread = ThreadLocal.withInitial(() -> new float[maxSlabElems]);
            }
            else {
                //TODO:
                throw new UnsupportedOperationException("No setup implemented yet for 3D Indexing with ram usage prevention");
            }
            this.kernel = createKernel();
            return this.kernel;
        }

        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            final int plane = width * height;
            final int L = params.isVectorize() ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = params.isVectorize() ? (width + L - 1) / L : width;

            if(params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

                for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                    final int dz = Math.min(slabDepth, depth - zStart);

                    for (int yStart = 0; yStart < height; yStart += rowsPerTask) {
                        final int rows = Math.min(rowsPerTask, height - yStart);

                        // Slab vorbereiten
                        Arrays.fill(cacheSlab, 0);
                        kernel.setExplicit(true);
                        kernel.bindOutput(cacheSlab);

                        // Lokale Dimensionen an den Kernel geben
                        kernel.setParameters(
                                x0,
                                y0 + yStart * frequency,      // Y-Offset in Weltkoords
                                z0 + zStart * frequency,      // Z-Offset in Weltkoords
                                width, rows, dz,      // lokale Slab-Dims (Breite bleibt UNGEPUFFERT)
                                frequency,
                                0                     // base=0 (Slab-only)
                        );

                        // WICHTIG: bei vectorized = Wv * rows * dz  (sonst: width * rows * dz)
                        final int elems = Wv * rows * dz;
                        kernel.execute(Range.create(elems, 1)); // 1D-NDRange
                        kernel.get(cacheSlab);

                        // Slab zurückkopieren (nur die realen 'width' Elemente je Zeile)
                        final int rowStride = width * rows; // Elemente pro z-Schicht im Slab
                        for (int z = 0; z < dz; z++) {
                            final int src = z * rowStride;
                            final int dst = (zStart + z) * plane + yStart * width;
                            System.arraycopy(cacheSlab, src, result, dst, rowStride);
                        }
                    }
                }
            }
            else {

                List<Future<?>> futures = new ArrayList<>(1024);

                for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                    final int dz = Math.min(slabDepth, depth - zStart);

                    for (int yStart = 0; yStart < height; yStart += rowsPerTask) {
                        final int rows = Math.min(rowsPerTask, height - yStart);

                        final int finalYStart = yStart;
                        final int finalZStart = zStart;

                        Future<?> future = cacheOptPool.submit(() -> {
                            AbstractSimplexNoiseKernel kernel = cacheOptKernels.get();
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
                                    y0 + finalYStart * frequency,
                                    z0 + finalZStart * frequency,
                                    width, rows, dz,
                                    frequency,
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
            }
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
        public void logSetup() {

        }
    }


    public static class Simple extends CPUJavaAparapiNoiseBackend {
        public Simple(Device preferredDevice, NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
            super(preferredDevice, params, result, width, height, depth);
        }

        @Override
        protected AbstractSimplexNoiseKernel setup() {
            if(use1DIndexing) {
                this.local1D = 0;
                this.slabDepth = params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL) ? depth : Math.min(32, depth);
            }
            else {
                //TODO:
                throw new UnsupportedOperationException("No setup implemented yet for 3D Indexing with ram usage prevention");
            }
            this.kernel = createKernel();
            return this.kernel;
        }

        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            final int plane = width * height;
            final int L     = params.isVectorize() ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv    = params.isVectorize() ? (width + L - 1) / L : width;
            if(params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.bindOutput(result);
                kernel.setParameters(x0, y0, z0, width, height, depth, frequency, 0);
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

                // global size: vectorized => Wv * height * depth ; sonst plane * depth
                final int global = Wv * height * depth;
                kernel.execute(Range.create(global, 1));
            }
            else {
                kernel.bindOutput(result);
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.JTP);

                for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                    final int dz = Math.min(slabDepth, depth - zStart);

                    // Vector-global size für diesen z-Block
                    final int global = Wv * height * dz;

                    // base zeigt in das große Zielarray auf die erste Schicht dieses Blocks
                    kernel.setParameters(
                            x0, y0, z0 + zStart * frequency,
                            width, height, dz,
                            frequency,
                            zStart * plane
                    );

                    // im JTP-Mode keine lokale Größe
                    kernel.execute(Range.create(global));
                }
            }
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
        public void logSetup() {

        }
    }

    /**
     * Wählt Y-Zeilen pro Task (1D-CPU-Backend) anhand von L3-Cache und Threads.
     * Idee: Pro Task soll W * rows * bytesPerVoxel (plus Overhead) in den L3-Anteil pro Thread passen.
     *
     * @param W             Breite (X)
     * @param bytesPerVoxel z.B. Float.BYTES
     * @param threads       parallel aktive Worker (z.B. Runtime.getRuntime().availableProcessors())
     * @param l3Bytes       gesamte L3-Größe (Bytes). Wenn 0/unknown -> Fallback.
     */
    public static int pickRowsPerTaskL3Aware(int W, int bytesPerVoxel, int threads, long l3Bytes) {
        // Fallback, falls L3 unbekannt:
        if (l3Bytes <= 0) {
            long target = 768L << 10; // ~768KB wie bisher
            long rowBytes = (long) W * bytesPerVoxel;
            int rows = (int) Math.max(1L, target / Math.max(1, rowBytes));
            rows = Math.max(8, (rows / 8) * 8);
            return rows;
        }

        // Wir nutzen nur einen Teil des L3 (z.B. 50%), geteilt durch Threads.
        // Safety > 1 lässt Platz für Code/Stacks/Temporaries & Write-Allocate-Effekte.
        final double L3_FRACTION = 0.50;     // 50% des L3 budgetieren
        final double SAFETY = 1.5;      // 1.5x Sicherheitsfaktor

        long perThreadBudget = (long) ((l3Bytes * L3_FRACTION) / Math.max(1, threads));
        long rowBytes = (long) W * bytesPerVoxel;

        // Ziel: rows * rowBytes * SAFETY <= perThreadBudget
        long maxRowsByCache = (long) Math.floor(perThreadBudget / Math.max(1.0, rowBytes * SAFETY));
        int rows = (int) Math.max(1, Math.min(Integer.MAX_VALUE, maxRowsByCache));

        // Runde auf sinnvolle Kachelgröße (Multiplikator), z.B. 8:
        int QUANT = 8;
        rows = Math.max(QUANT, (rows / QUANT) * QUANT);

        // Unter- und Obergrenzen, um Mini-/Riesen-Tasks zu vermeiden:
        rows = Math.max(QUANT, rows);
        rows = Math.min(rows, 1 << 15); // z.B. Deckel 32768 Zeilen

        return rows;
    }

    /**
     * Wähle eine dz-Slab-Tiefe für CPU-Backend (3D), sodass W*H*dz*bytesPerVoxel in ein Budget passt,
     * z.B. 50% des L3. Optional: runde auf Vielfaches einer "Thread-Kachel".
     */
    public static int pickSlabDepthForL3(int W, int H, int D, int bytesPerVoxel, int threads) {
        HardwareUtil.CacheSizes cs = HardwareUtil.readCaches();

        long l3 = (cs.l3.sizeBytes() > 0) ? cs.l3.sizeBytes() : 0;
        double frac = (cs.l3.sizeBytes() > 0) ? 0.50 : 0.25;
        long budget = (long) Math.max(1, l3 * frac);

        long plane = (long) W * H * bytesPerVoxel;
        int dz = (int) Math.max(1L, Math.min((long) D, budget / Math.max(1, plane)));

        long minElems = (long) threads * 8192L; // heuristisch
        long elems = (long) W * H * dz;
        if (elems < minElems) {
            long need = (minElems + (W * H) - 1) / (W * H);
            dz = (int) Math.min((long) D, Math.max((long) dz, need));
        }

        int base = 16;
        dz = Math.max(base, (dz / base) * base);
        dz = Math.min(dz, D);

        dz = Math.min(dz, 1024);
        return dz;
    }

    public static int pickDzFor1D(int W, int H, int D, int b, int threads, long l3) {
        final double FRACTION = 0.50;
        final int DZ_Q = 16, DZ_MAX = 1024;
        long budget = (long) Math.max(1, Math.max(0, l3) * FRACTION);

        long planeBytes = (long) W * H * b;
        int dz = (int) Math.max(1L, Math.min((long) D, budget / Math.max(1, planeBytes)));

        long minElems = (long) threads * 8192L;
        long elems = (long) W * H * dz;
        if (elems < minElems) {
            long need = (minElems + (W * H) - 1) / (W * H);
            dz = (int) Math.min((long) D, Math.max((long) dz, need));
        }
        dz = Math.max(DZ_Q, (dz / DZ_Q) * DZ_Q);
        dz = Math.min(dz, Math.min(D, DZ_MAX));
        return dz;
    }

    public static int pickRowsPerTaskFor1DWithDz(int W, int dz, int b, int threads, long l3) {
        final double FRACTION = 0.50, SAFETY = 1.5;
        final int ROW_Q = 8;

        long perThread = (long) ((Math.max(0, l3) * FRACTION) / Math.max(1, threads));
        long bytesPerRow = (long) W * dz * b;

        if (perThread <= 0 || bytesPerRow <= 0) {
            long target = 768L << 10; // Fallback ~768 KiB
            int rows = (int) Math.max(1L, target / Math.max(1, bytesPerRow));
            return Math.max(ROW_Q, (rows / ROW_Q) * ROW_Q);
        }

        long maxRows = (long) Math.floor(perThread / Math.max(1.0, bytesPerRow * SAFETY));
        int rows = (int) Math.max(1L, Math.min(Integer.MAX_VALUE, maxRows));
        return Math.max(ROW_Q, (rows / ROW_Q) * ROW_Q);
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
