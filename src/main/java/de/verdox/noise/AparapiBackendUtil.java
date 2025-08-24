package de.verdox.noise;

import com.aparapi.device.OpenCLDevice;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class AparapiBackendUtil {

    public static final int MAX_OPENCL_GROUP_SIZE = 256;

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

    public static String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        // Einheiten-Labels für 1024er-Potenzen
        String[] units = {"KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
        // Berechne, welche Potenz von 1024 wir brauchen
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        double value = bytes / Math.pow(1024, exp);
        // Format mit zwei Nachkommastellen
        return String.format(Locale.US, "%.2f %s", value, units[exp - 1]);
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
