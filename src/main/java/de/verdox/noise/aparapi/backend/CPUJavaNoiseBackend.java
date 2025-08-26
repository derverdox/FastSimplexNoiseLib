package de.verdox.noise.aparapi.backend;

import com.aparapi.device.Device;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel1D;
import de.verdox.noise.aparapi.kernel.scalar.VectorizedSimplexNoise3DKernel1D;
import de.verdox.util.FormatUtil;
import de.verdox.util.HardwareUtil;

import java.util.concurrent.ExecutorService;

public abstract class CPUJavaNoiseBackend extends AparapiNoiseBackend<AbstractSimplexNoise3DAparapiKernel> {
    protected final boolean vectorized;
    protected final boolean optimizeCache;
    protected int rowsPerTask;
    protected float[] cacheSlab;

    protected ThreadLocal<AbstractSimplexNoise3DAparapiKernel> cacheOptKernels;
    protected ThreadLocal<float[]> slabsPerThread;
    protected ExecutorService cacheOptPool;
    private int maxSlabElems;
    protected static final int physicalProcessors = HardwareUtil.getPhysicalProcessorCount();

    public CPUJavaNoiseBackend(Device preferredDevice, boolean vectorized, boolean optimizeCache, float[] result, int width, int height, int depth) {
        super(preferredDevice, result, width, height, depth);
        this.vectorized = vectorized;
        this.optimizeCache = optimizeCache;
    }

    @Override
    protected AbstractSimplexNoise3DAparapiKernel setup() {
        if (optimizeCache) {
            int threads = physicalProcessors;
            long l3 = HardwareUtil.readCaches().l3.sizeBytes();
            this.slabDepth = pickDzFor1D(width, height, depth, Float.BYTES, threads, l3);
            this.rowsPerTask = pickRowsPerTaskFor1DWithDz(width, slabDepth, Float.BYTES, threads, l3);
            maxSlabElems = width * rowsPerTask * slabDepth;

            cacheSlab = new float[maxSlabElems];
            if(cacheOptPool == null) {
                cacheOptPool = java.util.concurrent.Executors.newFixedThreadPool(threads, r -> {
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
    protected AbstractSimplexNoise3DAparapiKernel createKernel() {
        return vectorized ? new VectorizedSimplexNoise3DKernel1D() : new ScalarSimplexNoise3DKernel1D();
    }

    protected abstract int threadsUsed();

    @Override
    public void logSetup() {

        HardwareUtil.printCPU();
        System.out.println("Allocated: " + FormatUtil.formatBytes2((long) width * height * depth * Float.BYTES));
        if (cacheSlab != null) {
            System.out.println("Cache Optimization Mode: Splitting L3 Cache to "+threadsUsed()+" threads with "+FormatUtil.formatBytes2(cacheSlab.length * Float.BYTES)+" for each");
        }

        System.out.printf("Mode: 1D | local=%d | slabDepth=%d | dims=(%d,%d,%d)%n", local1D, slabDepth, width, height, depth);
        System.out.println("================================");
    }

    @Override
    public void dispose() {
        super.dispose();
        if(cacheOptPool != null) {
            cacheOptPool.close();
            cacheOptPool = null;
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
