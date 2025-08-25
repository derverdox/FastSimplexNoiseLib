package de.verdox.noise;

import com.aparapi.device.OpenCLDevice;
import de.verdox.util.HardwareUtil;

import java.util.ArrayList;
import java.util.List;

public class AparapiBackendUtil {

    public static final int MAX_OPENCL_GROUP_SIZE = 256;

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
        int dz = (int) Math.max(1L, Math.min((long)D, budget / Math.max(1, plane)));

        long minElems = (long) threads * 8192L; // heuristisch
        long elems = (long) W * H * dz;
        if (elems < minElems) {
            long need = (minElems + (W*H) - 1) / (W*H);
            dz = (int) Math.min((long) D, Math.max((long) dz, need));
        }

        int base = 16;
        dz = Math.max(base, (dz / base) * base);
        dz = Math.min(dz, D);

        dz = Math.min(dz, 1024);
        return dz;
    }

    /** Wähle eine sinnvolle Slab-Tiefe (dz) für Range.create3D(W,H,dz). */
    public static int pickSlabDepth3D(
            com.aparapi.device.OpenCLDevice dev,
            int W, int H, int D,
            int localX, int localY, int localZ,
            int bytesPerVoxel // float=4
    ) {
        // 1) Platz im VRAM (konservativ nur Anteil nutzen)
        long vram = dev.getGlobalMemSize();               // gesamt VRAM
        long vramBudget = (long)(vram * 0.50);            // z.B. 50% Budget
        long maxBuffer = Math.max(64L << 20, Math.min(vramBudget, 256L << 20)); // 64–256 MB Ziel

        // 2) Zielthreads: genug Work-Groups
        int cus = dev.getMaxComputeUnits();               // z.B. 68
        int targetWorkgroups = cus * 8;                   // grob: 8 WGs pro CU
        // Mindestanzahl globaler Work-Items (sehr grob):
        long minGlobalItems = (long)targetWorkgroups * (long)(localX*localY*localZ);

        // 3) Erste Abschätzung von dz aus Speicherbudget
        long planeBytes = (long) W * H * bytesPerVoxel;   // bytes pro z-Slice
        int dzByMem = (int)Math.max(1L, Math.min(D, maxBuffer / Math.max(1, planeBytes)));

        // 4) Threads-Check: falls zu wenig Items, dz erhöhen (bis zu D)
        long gItems = (long)W * H * dzByMem;
        if (gItems < minGlobalItems) {
            long need = (minGlobalItems + (W*H) - 1) / (W*H);
            dzByMem = (int)Math.min((long)D, Math.max((long)dzByMem, need));
        }

        // 5) An lokale Größe anpassen (Vielfaches von localZ) und Grenzen
        int dz = Math.max(localZ, (dzByMem / localZ) * localZ);
        dz = Math.min(dz, D);
        // Sicherheitskorridor: nicht zu klein, nicht „riesig“
        dz = clamp(dz, localZ, Math.min(D, 1024)); // 1024 nur als Deckel gegen übergroße Slabs

        return dz;
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    /** Für 1D-CPU-Backend: wie viele Zeilen (Y) pro Task? -> Ziel: ~512KB-1MB Output pro Task. */
    public static int pickRowsPerTask(int W, int bytesPerVoxel) {
        long target = 768L << 10; // ~768 KB
        long rowBytes = (long) W * bytesPerVoxel;
        int rows = (int) Math.max(1L, target / Math.max(1, rowBytes));
        // runde auf 8er-Schritte:
        rows = Math.max(8, (rows / 8) * 8);
        return rows;
    }

    /** 3D-Local-Size zur Laufzeit generieren (warp-aware) & bewerten. */
    public static int[] pickLocal3D(int W, int H, int D, int maxWG, int[] maxIt, int warp) {
        // mögliche XY-Seiten, lieber quadratisch/klein bis mittel (Cache/Koaleszierung)
        int[] sides = {8, 16, 32, 64};
        int[] lzOpts = {1, 2, 4, 8};

        List<int[]> cands = new ArrayList<>();
        for (int lx : sides) {
            if (lx > maxIt[0]) continue;
            if (W % lx != 0) continue;
            for (int ly : sides) {
                if (ly > maxIt[1]) continue;
                if (H % ly != 0) continue;
                for (int lz : lzOpts) {
                    if (lz > maxIt[2]) continue;
                    long prod = 1L * lx * ly * lz;
                    if (prod == 0 || prod > maxWG) continue;
                    if (prod % warp != 0) continue;
                    cands.add(new int[]{lx, ly, lz});
                }
            }
        }
        if (cands.isEmpty()) return null;

        // Sortierung: (1) größeres Produkt, (2) quadratischer in XY, (3) kleineres lz
        cands.sort((a, b) -> {
            long pa = 1L * a[0] * a[1] * a[2];
            long pb = 1L * b[0] * b[1] * b[2];
            if (pa != pb) return Long.compare(pb, pa);
            int sqA = Math.abs(a[0] - a[1]);
            int sqB = Math.abs(b[0] - b[1]);
            if (sqA != sqB) return Integer.compare(sqA, sqB);
            return Integer.compare(a[2], b[2]);
        });

        return cands.get(0); // beste nehmen
    }

    /** 3D: dz so wählen, dass dz % localZ == 0 und genug Work-Groups pro Slice (≥ 4×CUs). */
    public static int pickDzFor3D(int W, int H, int D, int lx, int ly, int lz, int CUs) {
        int targetGroups = Math.max(4 * CUs, 1);
        int[] cand = {256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1};

        for (int dz : cand) {
            if (dz > D) continue;
            if (dz % Math.max(lz, 1) != 0) continue;
            long gx = W / lx, gy = H / ly, gz = dz / Math.max(lz, 1);
            long groups = gx * gy * gz;
            if (groups >= targetGroups) return dz;
        }
        // größtes, das lz teilt
        for (int dz : cand) {
            if (dz <= D && dz % Math.max(lz, 1) == 0) return dz;
        }
        return Math.min(D, 16);
    }

    /** 1D: größtes multiple von warp ≤ maxWG. */
    public static int pickLocal1D(int maxWG, int warp) {
        maxWG = MAX_OPENCL_GROUP_SIZE;
        int start = maxWG - (maxWG % warp);
        for (int s = start; s >= warp; s -= warp) {
            return s; // erstes gültiges (größtes) Multiple
        }
        return 1;
    }

    /** 1D: dz so, dass sliceElems % local1D == 0 und genug Work-Groups entstehen. */
    public static int pickDzFor1D(int W, int H, int D, int local1D, int CUs) {
        int targetGroups = Math.max(4 * CUs, 1);
        int plane = W * H;
        int[] cand = {256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1};

        for (int dz : cand) {
            if (dz > D) continue;
            int slice = plane * dz;
            if (slice % local1D != 0) continue;
            long groups = slice / local1D;
            if (groups >= targetGroups) return dz;
        }
        // größtes, das teilbar ist
        for (int dz : cand) {
            if (dz <= D && (plane * dz) % local1D == 0) return dz;
        }
        return Math.min(D, 16);
    }

    public static int detectPreferredWarp(OpenCLDevice dev) {
        String v = dev.getOpenCLPlatform().getVendor().toLowerCase();
        if (v.contains("nvidia")) return 32;
        if (v.contains("advanced micro") || v.contains("amd")) return 64;
        return 32; // konservativ (Intel/sonstige)
    }

    /** Wählt ein gutes rowsPerTask für (width,height,depth). */
    public static int pickRowsPerTask(int width, int height, int depth, int cores) {
        final int totalRows = Math.max(1, depth * height);
        final long bytesPerRow = (long) width * 4L;

        // 1) Ziel-Chunkgröße (zwischen 256KB und 1MB)
        final long targetBytes = 512L * 1024L; // Mittelwert; 256K..1M sind gute Startpunkte

        // 2) Initiale Wahl: so viele Zeilen, dass wir nahe targetBytes sind
        int rpt = (int) Math.max(1, Math.min(totalRows, targetBytes / Math.max(1, bytesPerRow)));

        // 3) Sicherstellen: genug Tasks für gutes Balancing (≥ 8× Kerne)
        final int minTasks = Math.max(cores * 8, cores + 1);
        int tasks = Math.max(1, totalRows / Math.max(1, rpt));
        if (tasks < minTasks) {
            // verkleinere rpt, aber min 1
            rpt = Math.max(1, totalRows / minTasks);
        }

        // 4) Soft-Caps: nicht absurd groß/klein
        rpt = Math.max(1, Math.min(rpt, 256)); // 1..256 Rows pro Task als vernünftiger Korridor

        return rpt;
    }

    /** Variante mit konfigurierbarem Zielbereich. */
    public static int pickRowsPerTask(int width, int height, int depth, int cores, long minBytes, long maxBytes) {
        final int totalRows = Math.max(1, depth * height);
        final long bytesPerRow = (long) width * 4L;
        long targetBytes = Math.max(minBytes, Math.min(maxBytes, 512L*1024L));
        int rpt = (int) Math.max(1, Math.min(totalRows, targetBytes / Math.max(1, bytesPerRow)));

        final int minTasks = Math.max(cores * 8, cores + 1);
        int tasks = Math.max(1, totalRows / Math.max(1, rpt));
        if (tasks < minTasks) rpt = Math.max(1, totalRows / minTasks);

        rpt = Math.max(1, Math.min(rpt, 256));
        return rpt;
    }
}
