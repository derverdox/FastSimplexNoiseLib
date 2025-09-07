package de.verdox.noise.aparapi.backend.cpu;

import com.aparapi.Kernel;
import com.aparapi.Range;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.aparapi.backend.AparapiNoiseBackend;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoiseKernel;
import de.verdox.noise.aparapi.kernel.cpu.CPUScalarSimplexNoiseKernel;
import de.verdox.noise.aparapi.kernel.cpu.CPUVectorSimplexNoiseKernel;
import de.verdox.util.HardwareUtil;
import de.verdox.util.LODUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public abstract class CPUJavaAparapiNoiseBackend extends AparapiNoiseBackend<AbstractSimplexNoiseKernel> {
    protected final NoiseBackendBuilder.CPUNoiseBackendBuilder params;


    public CPUJavaAparapiNoiseBackend(NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
        super(null, params.getNoiseCalculationMode(), result, width, height, depth);
        this.params = params;
    }

    public CPUJavaAparapiNoiseBackend(NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int depth) {
        super(null, params.getNoiseCalculationMode(), result, width, depth);
        this.params = params;
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
        int dz = (int) Math.max(1L, Math.min(D, budget / Math.max(1, plane)));

        long minElems = (long) threads * 8192L; // heuristisch
        long elems = (long) W * H * dz;
        if (elems < minElems) {
            long need = (minElems + ((long) W * H) - 1) / ((long) W * H);
            dz = (int) Math.min(D, Math.max(dz, need));
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
        int dz = (int) Math.max(1L, Math.min(D, budget / Math.max(1, planeBytes)));

        long minElems = (long) threads * 8192L;
        long elems = (long) W * H * dz;
        if (elems < minElems) {
            long need = (minElems + ((long) W * H) - 1) / ((long) W * H);
            dz = (int) Math.min(D, Math.max(dz, need));
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

    public int threads() {
        return switch (params.getParallelismMode()) {
            case SEQUENTIAL -> 1;
            case PARALLELISM_CORES -> HardwareUtil.getPhysicalProcessorCount();
            case PARALLELISM_THREADS -> Runtime.getRuntime().availableProcessors();
        };
    }

    @Override
    protected AbstractSimplexNoiseKernel createKernel() {
        if (params.isVectorize()) {
            if (use1DIndexing) {
                return params.is3DMode() ? new CPUVectorSimplexNoiseKernel.Simple.Noise3DIndexing1D(params.getNoiseCalculationMode()) : new CPUVectorSimplexNoiseKernel.Simple.Noise2DIndexing1D(params.getNoiseCalculationMode());
            } else {
                return params.is3DMode() ? new CPUVectorSimplexNoiseKernel.Simple.Noise3DIndexing3D(params.getNoiseCalculationMode()) : new CPUVectorSimplexNoiseKernel.Simple.Noise2DIndexing2D(params.getNoiseCalculationMode());
            }
        } else {
            if (use1DIndexing) {
                return params.is3DMode() ? new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing1D(params.getNoiseCalculationMode()) : new CPUScalarSimplexNoiseKernel.Simple.Noise2DIndexing1D(params.getNoiseCalculationMode());
            } else {
                return params.is3DMode() ? new CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing3D(params.getNoiseCalculationMode()) : new CPUScalarSimplexNoiseKernel.Simple.Noise2DIndexing2D(params.getNoiseCalculationMode());
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

        public CacheOnly(NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
            super(params, result, width, height, depth);
        }

        public CacheOnly(NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int depth) {
            super(params, result, width, depth);
        }


        @Override
        protected AbstractSimplexNoiseKernel setup() {
            int threads = threads();
            long l3 = HardwareUtil.readCaches().l3.sizeBytes();

            if (use1DIndexing) {
                // (dein bestehender 1D-Setup-Code bleibt unverändert)
                this.slabDepth = pickDzFor1D(width, height, depth, Float.BYTES, threads, l3);
                this.rowsPerTask = pickRowsPerTaskFor1DWithDz(width, slabDepth, Float.BYTES, threads, l3);
                maxSlabElems = width * rowsPerTask * slabDepth;

                cacheSlab = new float[maxSlabElems];
                if (cacheOptPool == null) {
                    cacheOptPool = Executors.newFixedThreadPool(threads, r -> {
                        Thread t = new Thread(r);
                        t.setDaemon(true);
                        return t;
                    });
                }
                this.cacheOptKernels = ThreadLocal.withInitial(this::createKernel);
                this.slabsPerThread = ThreadLocal.withInitial(() -> new float[maxSlabElems]);
            } else {
                // === 3D-Indexing ===
                // Zuerst sinnvolle Z-Slab-Tiefe für den L3-Anteil pro Thread wählen …
                this.slabDepth = pickSlabDepthForL3(width, height, depth, Float.BYTES, threads);
                // … dann Y-Blockgröße (Zeilen pro Task) passend zur Z-Tiefe und L3 wählen
                this.rowsPerTask = pickRowsPerTaskFor1DWithDz(width, slabDepth, Float.BYTES, threads, l3);

                // Puffergrößen auf Basis der lokalen Tile-Maße:
                this.maxSlabElems = width * rowsPerTask * slabDepth;
                this.cacheSlab = new float[maxSlabElems];

                if (cacheOptPool == null) {
                    cacheOptPool = Executors.newFixedThreadPool(threads, r -> {
                        Thread t = new Thread(r);
                        t.setDaemon(true);
                        return t;
                    });
                }
                this.cacheOptKernels = ThreadLocal.withInitial(this::createKernel);
                this.slabsPerThread = ThreadLocal.withInitial(() -> new float[maxSlabElems]);
            }

            this.kernel = createKernel();
            return this.kernel;
        }


        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD3DParams lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), H = lp.heightLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BY = lp.baseY(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final int L = params.isVectorize() ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = params.isVectorize() ? (W + L - 1) / L : W;

            final int slabDepthNow = Math.max(1, Math.min(slabDepth, D));
            final int rowsPerTaskNow = Math.max(1, Math.min(rowsPerTask, H));
            final int plane = W * H;

            if (params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

                for (int zStart = 0; zStart < D; zStart += slabDepthNow) {
                    final int dz = Math.min(slabDepthNow, D - zStart);

                    for (int yStart = 0; yStart < H; yStart += rowsPerTaskNow) {
                        final int rows = Math.min(rowsPerTaskNow, H - yStart);

                        // Slab vorbereiten
                        final int need = W * rows * dz;
                        if (cacheSlab == null || cacheSlab.length < need) cacheSlab = new float[need];
                        Arrays.fill(cacheSlab, 0f);

                        kernel.setExplicit(true);
                        kernel.bindOutput(cacheSlab);
                        kernel.setParameters(
                                BX,
                                BY + yStart * FQ,
                                BZ + zStart * FQ,
                                W, rows, dz,
                                FQ,
                                0, params.getSeed()
                        );

                        final int elems = Wv * rows * dz;
                        kernel.execute(Range.create(elems, 1));
                        kernel.get(cacheSlab);

                        final int rowStride = W * rows;
                        for (int z = 0; z < dz; z++) {
                            final int src = z * rowStride;
                            final int dst = (zStart + z) * plane + yStart * W;
                            System.arraycopy(cacheSlab, src, result, dst, rowStride);
                        }
                    }
                }
            } else {
                List<Future<?>> futures = new ArrayList<>(1024);

                for (int zStart = 0; zStart < D; zStart += slabDepthNow) {
                    final int dz = Math.min(slabDepthNow, D - zStart);

                    for (int yStart = 0; yStart < H; yStart += rowsPerTaskNow) {
                        final int rows = Math.min(rowsPerTaskNow, H - yStart);

                        final int fz = zStart, fy = yStart;

                        futures.add(cacheOptPool.submit(() -> {
                            AbstractSimplexNoiseKernel k = cacheOptKernels.get();
                            float[] slab = slabsPerThread.get();

                            final int need = W * rows * dz;
                            if (slab.length < need) {
                                slab = new float[need];
                                slabsPerThread.set(slab);
                            }
                            k.bindOutput(slab);

                            k.setParameters(
                                    BX,
                                    BY + fy * FQ,
                                    BZ + fz * FQ,
                                    W, rows, dz,
                                    FQ,
                                    0, params.getSeed()
                            );

                            k.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
                            k.execute(Range.create(Wv * rows * dz, 1));

                            final int rowStride = W * rows;
                            for (int z = 0; z < dz; z++) {
                                final int src = z * rowStride;
                                final int dst = (fz + z) * plane + fy * W;
                                System.arraycopy(slab, src, result, dst, rowStride);
                            }
                        }));
                    }
                }

                for (Future<?> ftr : futures) {
                    try {
                        ftr.get();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException(e);
                    } catch (ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }

        @Override
        public void generate3DNoise3DIndexed(float x0, float y0, float z0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD3DParams lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), H = lp.heightLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BY = lp.baseY(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final boolean vec = params.isVectorize();
            final int L = vec ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = vec ? (W + L - 1) / L : W;

            final int slabDepthNow = Math.max(1, Math.min(slabDepth, D));
            final int rowsPerTaskNow = Math.max(1, Math.min(rowsPerTask, H));
            final int plane = W * H;

            if (params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

                for (int zStart = 0; zStart < D; zStart += slabDepthNow) {
                    final int dz = Math.min(slabDepthNow, D - zStart);

                    for (int yStart = 0; yStart < H; yStart += rowsPerTaskNow) {
                        final int rows = Math.min(rowsPerTaskNow, H - yStart);

                        final int need = W * rows * dz;
                        if (cacheSlab == null || cacheSlab.length < need) cacheSlab = new float[need];
                        Arrays.fill(cacheSlab, 0f);

                        kernel.bindOutput(cacheSlab);
                        kernel.setParameters(
                                BX,
                                BY + yStart * FQ,
                                BZ + zStart * FQ,
                                W, rows, dz,
                                FQ,
                                0, params.getSeed()
                        );

                        final Range r3 = Range.create3D(Wv, rows, dz, 1, 1, 1);
                        kernel.execute(r3);
                        kernel.get(cacheSlab);

                        final int rowStride = W * rows;
                        for (int z = 0; z < dz; z++) {
                            final int src = z * rowStride;
                            final int dst = (zStart + z) * plane + yStart * W;
                            System.arraycopy(cacheSlab, src, result, dst, rowStride);
                        }
                    }
                }
            } else {
                final List<Future<?>> futures = new ArrayList<>(1024);

                for (int zStart = 0; zStart < D; zStart += slabDepthNow) {
                    final int dz = Math.min(slabDepthNow, D - zStart);

                    for (int yStart = 0; yStart < H; yStart += rowsPerTaskNow) {
                        final int rows = Math.min(rowsPerTaskNow, H - yStart);

                        final int fz = zStart, fy = yStart;

                        futures.add(cacheOptPool.submit(() -> {
                            AbstractSimplexNoiseKernel k = cacheOptKernels.get();
                            float[] slab = slabsPerThread.get();

                            final int need = W * rows * dz;
                            if (need > slab.length) {
                                slab = new float[need];
                                slabsPerThread.set(slab);
                            }

                            k.bindOutput(slab);
                            k.setParameters(
                                    BX,
                                    BY + fy * FQ,
                                    BZ + fz * FQ,
                                    W, rows, dz,
                                    FQ,
                                    0, params.getSeed()
                            );

                            k.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
                            final Range r3 = Range.create3D(Wv, rows, dz, 1, 1, 1);
                            k.execute(r3);

                            final int rowStride = W * rows;
                            for (int z = 0; z < dz; z++) {
                                final int src = z * rowStride;
                                final int dst = (fz + z) * plane + fy * W;
                                System.arraycopy(slab, src, result, dst, rowStride);
                            }
                        }));
                    }
                }

                for (Future<?> f : futures) {
                    try {
                        f.get();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException(e);
                    } catch (ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }

        @Override
        public void generate2DNoise1DIndexed(float x0, float y0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD2DParams lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final boolean vec = params.isVectorize();
            final int L = vec ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = vec ? (W + L - 1) / L : W;

            final int threads = threads();
            final long l3 = HardwareUtil.readCaches().l3.sizeBytes();
            final int rowsPerTask2D = Math.max(8, pickRowsPerTaskL3Aware(W, Float.BYTES, threads, l3));

            kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

            int maxRows = Math.min(rowsPerTask2D, D);
            final int needCap = W * maxRows;
            if (cacheSlab == null || cacheSlab.length < needCap) cacheSlab = new float[needCap];

            for (int zStart = 0; zStart < D; zStart += rowsPerTask2D) {
                final int rows = Math.min(rowsPerTask2D, D - zStart);

                Arrays.fill(cacheSlab, 0f);
                kernel.setExplicit(true);
                kernel.bindOutput(cacheSlab);
                kernel.setParameters(
                        BX, 0f, BZ + zStart * FQ,
                        W, 1, rows,
                        FQ,
                        0,
                        params.getSeed()
                );

                final int elems = Wv * rows;
                kernel.execute(Range.create(elems, 1));
                kernel.get(cacheSlab);

                System.arraycopy(cacheSlab, 0, result, zStart * W, rows * W);
            }
        }

        @Override
        public void generate2DNoise2DIndexed(float x0, float y0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD2DParams lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final boolean vec = params.isVectorize();
            final int L = vec ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = vec ? (W + L - 1) / L : W;

            final int threads = threads();
            final long l3 = HardwareUtil.readCaches().l3.sizeBytes();
            final int rowsPerTask2D = Math.max(8, pickRowsPerTaskL3Aware(W, Float.BYTES, threads, l3));

            kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

            int maxRows = Math.min(rowsPerTask2D, D);
            final int needCap = W * maxRows;
            if (cacheSlab == null || cacheSlab.length < needCap) cacheSlab = new float[needCap];

            for (int zStart = 0; zStart < D; zStart += rowsPerTask2D) {
                final int rows = Math.min(rowsPerTask2D, D - zStart);

                Arrays.fill(cacheSlab, 0f);
                kernel.bindOutput(cacheSlab);
                kernel.setParameters(
                        BX, 0f, BZ + zStart * FQ,
                        W, 1, rows,
                        FQ,
                        0,
                        params.getSeed()
                );

                final Range r2 = Range.create2D(Wv, rows, 1, 1);
                kernel.execute(r2);
                kernel.get(cacheSlab);

                System.arraycopy(cacheSlab, 0, result, zStart * W, rows * W);
            }
        }

        @Override
        public void logSetup() {

        }
    }

    public static class Simple extends CPUJavaAparapiNoiseBackend {
        public Simple(NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int height, int depth) {
            super(params, result, width, height, depth);
        }

        public Simple(NoiseBackendBuilder.CPUNoiseBackendBuilder params, float[] result, int width, int depth) {
            super(params, result, width, depth);
        }

        @Override
        protected AbstractSimplexNoiseKernel setup() {
            if (use1DIndexing) {
                this.local1D = 0;
                this.slabDepth = params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)
                        ? depth
                        : Math.min(32, depth); // kleiner Z-Block für JTP
            } else {
                // === 3D-Indexing ===
                // Für SEQ: ganzes Volumen; für JTP: L3-bewusste Z-Slabs
                this.slabDepth = params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)
                        ? depth
                        : pickSlabDepthForL3(width, height, depth, Float.BYTES, threads());
                // rowsPerTask nicht nötig in Simple (wir kacheln nur in Z)
            }
            this.kernel = createKernel();
            return this.kernel;
        }

        @Override
        public void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD3DParams lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), H = lp.heightLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BY = lp.baseY(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final int L = params.isVectorize() ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = params.isVectorize() ? (W + L - 1) / L : W;
            final int plane = W * H;

            kernel.bindOutput(result);

            if (params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.setParameters(BX, BY, BZ, W, H, D, FQ, 0, params.getSeed());
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
                final int global = Wv * H * D;
                kernel.execute(Range.create(global, 1));
            } else {
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
                for (int zStart = 0; zStart < D; zStart += slabDepth) {
                    final int dz = Math.min(slabDepth, D - zStart);
                    final int global = Wv * H * dz;

                    kernel.setParameters(
                            BX, BY, BZ + zStart * FQ,
                            W, H, dz,
                            FQ,
                            zStart * plane, params.getSeed()
                    );
                    kernel.execute(Range.create(global));
                }
            }
        }

        @Override
        public void generate3DNoise3DIndexed(float x0, float y0, float z0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD3DParams lp = LODUtil.computeLOD3D(width, height, depth, x0, y0, z0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), H = lp.heightLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BY = lp.baseY(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final boolean vec = params.isVectorize();
            final int L = vec ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = vec ? (W + L - 1) / L : W;

            kernel.bindOutput(result);

            if (params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.setParameters(BX, BY, BZ, W, H, D, FQ, 0, params.getSeed());
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
                final Range r3 = Range.create3D(Wv, H, D, 1, 1, 1);
                kernel.execute(r3);
            } else {
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
                final int plane = W * H;
                for (int zStart = 0; zStart < D; zStart += slabDepth) {
                    final int dz = Math.min(slabDepth, D - zStart);
                    kernel.setParameters(
                            BX, BY, BZ + zStart * FQ,
                            W, H, dz,
                            FQ,
                            zStart * plane, params.getSeed()
                    );
                    final Range r3 = Range.create3D(Wv, H, dz);
                    kernel.execute(r3);
                }
            }
        }

        @Override
        public void generate2DNoise1DIndexed(float x0, float y0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD2DParams lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final boolean vec = params.isVectorize();
            final int L = vec ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = vec ? (W + L - 1) / L : W;

            kernel.bindOutput(result);

            if (params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.setParameters(BX, 0f, BZ, W, 1, D, FQ, 0, params.getSeed());
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
                final int global = Wv * D;
                kernel.execute(Range.create(global, 1));
            } else {
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
                final int dzBlock = Math.min(64, D);
                for (int zStart = 0; zStart < D; zStart += dzBlock) {
                    final int dz = Math.min(dzBlock, D - zStart);
                    kernel.setParameters(BX, 0f, BZ + zStart * FQ,
                            W, 1, dz,
                            FQ,
                            zStart * W,
                            params.getSeed());
                    final int global = Wv * dz;
                    kernel.execute(Range.create(global));
                }
            }
        }

        @Override
        public void generate2DNoise2DIndexed(float x0, float y0, float frequency) {
            final int lod = params.getLodLevel();
            final var lodMode = params.getLodMode();
            final LODUtil.LOD2DParams lp = LODUtil.computeLOD2D(width, depth, x0, y0, frequency, lod, lodMode);

            final int W = lp.widthLOD(), D = lp.depthLOD();
            final float BX = lp.baseX(), BZ = lp.baseZ(), FQ = lp.frequencyLOD();

            final boolean vec = params.isVectorize();
            final int L = vec ? HardwareUtil.getVectorLaneLength() : 1;
            final int Wv = vec ? (W + L - 1) / L : W;

            kernel.bindOutput(result);

            if (params.getParallelismMode().equals(NoiseBackendBuilder.CPUParallelismMode.SEQUENTIAL)) {
                kernel.setParameters(BX, 0f, BZ, W, 1, D, FQ, 0, params.getSeed());
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
                final Range r2 = Range.create2D(Wv, D, 1, 1);
                kernel.execute(r2);
            } else {
                kernel.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
                final int dzBlock = Math.min(64, D);
                for (int zStart = 0; zStart < D; zStart += dzBlock) {
                    final int dz = Math.min(dzBlock, D - zStart);
                    kernel.setParameters(BX, 0f, BZ + zStart * FQ,
                            W, 1, dz,
                            FQ,
                            zStart * W,
                            params.getSeed());
                    final Range r2 = Range.create2D(Wv, dz);
                    kernel.execute(r2);
                }
            }
        }

        @Override
        public void logSetup() {

        }
    }
}
