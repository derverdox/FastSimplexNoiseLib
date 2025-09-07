package de.verdox.noise;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import de.verdox.noise.AparapiBackendUtil;

/**
 * Plant OpenCL-Launch-Parameter (global/local Sizes) für 1D/2D/3D-Dispatches
 * und liefert optional ein Output-Array in passender Größe zurück.
 *
 * Konventionen:
 * - 3D:   (W, H, D)  → X, Y, Z
 * - 2D:   (W, D)     → X, Z     (height wird implicit 1 gesetzt)
 * - 1D:   über alle Elemente linear
 */
public final class OpenCLTuner {

    private OpenCLTuner() {}

    /**
     * Hauptmethode (bestehend): plant 1D- oder 3D-Launch für ein Volumen (W,H,D).
     *
     * @param dev OpenCL-Device
     * @param W   width  (X)
     * @param H   height (Y)
     * @param D   depth  (Z)
     * @param prefer3D   wenn false → 1D-Plan, sonst 3D-Kandidaten probieren
     * @param alignBaseIndexTo32 Basisindex auf 32 ausrichten (optional; hier nur als Feld weitergereicht)
     */
    public static Plan plan(OpenCLDevice dev, int W, int H, int D,
                            boolean prefer3D,
                            boolean alignBaseIndexTo32) {

        final int warp    = AparapiBackendUtil.detectPreferredWarp(dev);
        final int cu      = dev.getMaxComputeUnits();
        final long devMaxWG = dev.getMaxWorkGroupSize();
        final int[] devMaxWI = dev.getMaxWorkItemSize();

        // Ziel-LocalSize nach Aparapi-Grenze (typ. 256) und Warp runden
        int targetLocal = (devMaxWG >= 256) ? 256 : (devMaxWG >= 128 ? 128 : 64);
        targetLocal = roundDownToMultiple(targetLocal, warp);
        if (targetLocal < warp) targetLocal = warp;

        // Heuristik: 1D häufig optimal; 3D nur bei echter Tile-Wiederverwendung/Shared-Mem sinnvoll
        if (!prefer3D) {
            // ---------- 1D ----------
            long n = (long) W * H * D;
            int lx = targetLocal, ly = 1, lz = 1;
            int gx = (int) roundUpToMultipleLong(n, lx);
            float[] out = new float[(int) n];
            int baseIdxAligned = alignBaseIndexTo32 ? roundUpToMultiple(0, 32) : 0;
            return new Plan(
                    /*rank*/ 1,
                    W, H, D,
                    lx, ly, lz,
                    gx, /*gy*/1, /*gz*/1,
                    warp, cu, out, baseIdxAligned
            );
        }

        // ---------- 3D ----------
        // Kandidaten (max ~256 Threads), X-major bevorzugt
        int[][] candidates = new int[][]{
                {32, 8, 1}, {16, 8, 2}, {8, 8, 4}, {16, 4, 4}, {32, 4, 2}
        };
        int lx = 0, ly = 0, lz = 0;
        for (int[] c : candidates) {
            int cx = c[0], cy = c[1], cz = c[2];
            if (cx <= devMaxWI[0] && cy <= devMaxWI[1] && cz <= devMaxWI[2]
                    && (cx * cy * cz) <= targetLocal
                    && cx <= Math.max(1, W) && cy <= Math.max(1, H) && cz <= Math.max(1, D)) {
                lx = cx; ly = cy; lz = cz;
                break;
            }
        }
        if (lx == 0) {
            // Fallback auf 1D, falls keine 3D-Kombination passt
            return plan(dev, W, H, D, false, alignBaseIndexTo32);
        }

        int gx = (int) roundUpToMultipleLong(W, lx);
        int gy = (int) roundUpToMultipleLong(H, ly);
        int gz = (int) roundUpToMultipleLong(D, lz);
        long n = (long) W * H * D;

        float[] out = new float[(int) n];
        int baseIdxAligned = alignBaseIndexTo32 ? roundUpToMultiple(0, 32) : 0;

        return new Plan(
                /*rank*/ 3,
                W, H, D,
                lx, ly, lz,
                gx, gy, gz,
                warp, cu, out, baseIdxAligned
        );
    }

    /**
     * 2D-Planer: plant 1D- oder 2D-Launch für ein (W x D)-Feld (height=1).
     *
     * @param dev OpenCL-Device
     * @param W   width  (X)
     * @param D   depth  (Z)
     * @param prefer2D   wenn false → 1D-Plan, sonst 2D-Kandidaten probieren
     * @param alignBaseIndexTo32 Basisindex auf 32 ausrichten (optional; hier nur als Feld weitergereicht)
     */
    public static Plan plan2D(OpenCLDevice dev, int W, int D,
                              boolean prefer2D,
                              boolean alignBaseIndexTo32) {

        final int warp    = AparapiBackendUtil.detectPreferredWarp(dev);
        final int cu      = dev.getMaxComputeUnits();
        final long devMaxWG = dev.getMaxWorkGroupSize();
        final int[] devMaxWI = dev.getMaxWorkItemSize();

        // Ziel-LocalSize wie oben
        int targetLocal = (devMaxWG >= 256) ? 256 : (devMaxWG >= 128 ? 128 : 64);
        targetLocal = roundDownToMultiple(targetLocal, warp);
        if (targetLocal < warp) targetLocal = warp;

        final long n = (long) W * D;
        final float[] out = new float[(int) n];
        final int baseIdxAligned = alignBaseIndexTo32 ? roundUpToMultiple(0, 32) : 0;

        if (!prefer2D) {
            // ---------- 1D ----------
            int lx = targetLocal, ly = 1, lz = 1;
            int gx = (int) roundUpToMultipleLong(n, lx);
            return new Plan(
                    /*rank*/ 1,
                    W, /*H*/1, D,
                    lx, ly, lz,
                    gx, /*gy*/1, /*gz*/1,
                    warp, cu, out, baseIdxAligned
            );
        }

        // ---------- 2D (X,Z) ----------
        // Kandidaten <= targetLocal; X-major bevorzugt
        int[][] cands = new int[][]{
                {64, 4}, {32, 8}, {16, 16}, {128, 2}, {8, 32}, {4, 64}
        };

        int lx = 0, lz = 0;
        for (int[] c : cands) {
            int cx = c[0], cz = c[1];
            if (cx <= devMaxWI[0] && cz <= devMaxWI[1]
                    && (cx * cz) <= targetLocal
                    && cx <= Math.max(1, W) && cz <= Math.max(1, D)) {
                lx = cx; lz = cz;
                break;
            }
        }

        if (lx == 0) {
            // Fallback 1D
            int l1 = targetLocal;
            int g1 = (int) roundUpToMultipleLong(n, l1);
            return new Plan(
                    /*rank*/ 1,
                    W, /*H*/1, D,
                    l1, 1, 1,
                    g1, 1, 1,
                    warp, cu, out, baseIdxAligned
            );
        }

        int gx = (int) roundUpToMultipleLong(W, lx);
        int gz = (int) roundUpToMultipleLong(D, lz);

        return new Plan(
                /*rank*/ 2,
                W, /*H*/1, D,
                lx, /*ly*/1, lz,
                gx, /*gy*/1, gz,
                warp, cu, out, baseIdxAligned
        );
    }

    // ============================ Helpers ============================

    private static int roundDownToMultiple(int v, int m) {
        return (m <= 0) ? v : (v / m) * m;
    }

    private static int roundUpToMultiple(int v, int m) {
        if (m <= 0) return v;
        int r = v % m;
        return (r == 0) ? v : (v + (m - r));
    }

    private static long roundUpToMultipleLong(long v, long m) {
        if (m <= 0) return v;
        long r = v % m;
        return (r == 0) ? v : (v + (m - r));
    }

    // ============================= Plan ==============================

    public static final class Plan {
        /** 1 = 1D (Range.create), 2 = 2D (Range.create2D), 3 = 3D (Range.create3D) */
        public final int rank;
        public final int W, H, D;
        public final int lx, ly, lz;  // local sizes
        public final int gx, gy, gz;  // global sizes (gepaddet)
        public final int warp, cu;
        public final float[] out;
        public final int baseIdxAligned;

        public Plan(int rank, int W, int H, int D,
                    int lx, int ly, int lz,
                    int gx, int gy, int gz,
                    int warp, int cu,
                    float[] out, int baseIdxAligned) {
            this.rank = rank;
            this.W = W; this.H = H; this.D = D;
            this.lx = lx; this.ly = ly; this.lz = lz;
            this.gx = gx; this.gy = gy; this.gz = gz;
            this.warp = warp; this.cu = cu;
            this.out = out; this.baseIdxAligned = baseIdxAligned;
        }

        /** Erzeugt die passende Aparapi-Range für den Plan. */
        public Range toRange() {
            return switch (rank) {
                case 3 -> Range.create3D(
                        Math.max(gx, 1), Math.max(gy, 1), Math.max(gz, 1),
                        Math.max(lx, 1), Math.max(ly, 1), Math.max(lz, 1));
                case 2 -> Range.create2D(
                        Math.max(gx, 1), Math.max(gz, 1),
                        Math.max(lx, 1), Math.max(lz, 1)); // (X,Z)
                default -> Range.create(
                        Math.max(gx, 1),
                        Math.max(lx, 1));
            };
        }

        @Override
        public String toString() {
            String dim = switch (rank) {
                case 3 -> "3D";
                case 2 -> "2D";
                default -> "1D";
            };
            return String.format("Plan{rank=%s, local=(%d,%d,%d), global=(%d,%d,%d), dims=(W=%d,H=%d,D=%d), warp=%d, CUs=%d}",
                    dim, lx, ly, lz, gx, gy, gz, W, H, D, warp, cu);
        }
    }
}
