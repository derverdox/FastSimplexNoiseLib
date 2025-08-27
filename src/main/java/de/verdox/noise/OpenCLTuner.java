package de.verdox.noise;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;

import java.util.Arrays;

public class OpenCLTuner {

    public static final class Plan {
        public final boolean use3D;
        public final int W, H, D;
        public final int lx, ly, lz;          // local size
        public final int gx, gy, gz;          // global size (padded)
        public final int localSize, globalItems;
        public final int warpSize;            // 32 (NV) / 64 (AMD), fallback 32
        public final int computeUnits;
        public final float[] out;             // length = W*H*D
        public final int alignedBaseIndex;    // optional: 32-float Alignment

        public Plan(boolean use3D, int W, int H, int D,
                    int lx, int ly, int lz,
                    int gx, int gy, int gz,
                    int warpSize, int computeUnits,
                    float[] out, int alignedBaseIndex) {
            this.use3D = use3D;
            this.W = W;
            this.H = H;
            this.D = D;
            this.lx = lx;
            this.ly = ly;
            this.lz = lz;
            this.gx = gx;
            this.gy = gy;
            this.gz = gz;
            this.localSize = lx * ly * lz;
            this.globalItems = use3D ? (gx * gy * gz) : gx;
            this.warpSize = warpSize;
            this.computeUnits = computeUnits;
            this.out = out;
            this.alignedBaseIndex = alignedBaseIndex;
        }

        public Range toRange() {
            if (use3D) return Range.create3D(gx, gy, gz, lx, ly, lz);
            return Range.create(gx, lx); // 1D
        }

        @Override
        public String toString() {
            return "Plan{" +
                    "use3D=" + use3D +
                    ", W=" + W +
                    ", H=" + H +
                    ", D=" + D +
                    ", lx=" + lx +
                    ", ly=" + ly +
                    ", lz=" + lz +
                    ", gx=" + gx +
                    ", gy=" + gy +
                    ", gz=" + gz +
                    ", localSize=" + localSize +
                    ", globalItems=" + globalItems +
                    ", warpSize=" + warpSize +
                    ", computeUnits=" + computeUnits +
                    ", Array length=" + out.length +
                    ", alignedBaseIndex=" + alignedBaseIndex +
                    '}';
        }
    }

    /**
     * Hauptmethode: liefert Local/Global Sizes + Output-Array.
     */
    public static Plan plan(OpenCLDevice dev, int W, int H, int D,
                            boolean prefer3D,
                            boolean alignBaseIndexTo32) {

        final int warp = AparapiBackendUtil.detectPreferredWarp(dev);
        final int cu = dev.getMaxComputeUnits();
        final long devMaxWG = dev.getMaxWorkGroupSize();
        final int[] devMaxWI = dev.getMaxWorkItemSize();

        // Aparapi-Grenze: 256
        int targetLocal = (devMaxWG >= 256) ? 256 : (devMaxWG >= 128 ? 128 : 64);
        // runde auf Vielfaches des warp
        targetLocal = roundDownToMultiple(targetLocal, warp);
        if (targetLocal < warp) targetLocal = warp;

        // Heuristik: 1D ist für euren Simplex meist ideal; 3D nur wenn echte Tile-Wiederverwendung geplant
        if (!prefer3D) {
            // 1D: local = targetLocal; global = ceil(W*H*D / local) * local
            long n = (long) W * H * D;
            long global = roundUpToMultipleLong(n, targetLocal);
            // Output-Array exakt (ohne Padding)
            float[] out = new float[(int) n];
            int baseIdxAligned = alignBaseIndexTo32 ? roundUpToMultiple((int) 0, 32) : 0;
            return new Plan(false, W, H, D,
                    targetLocal, 1, 1,
                    (int) global, 1, 1,
                    warp, cu, out, baseIdxAligned);
        }

        // 3D: Kandidaten (alle = 256 Threads) in x-major Reihenfolge
        int[][] candidates = new int[][]{
                {32, 8, 1}, {16, 8, 2}, {8, 8, 4}, {16, 4, 4}, {32, 4, 2}
        };
        int lx = 0, ly = 0, lz = 0;
        for (int[] c : candidates) {
            int cx = c[0], cy = c[1], cz = c[2];
            // jede Dimension muss device limits UND Volumendimension respektieren
            if (cx <= devMaxWI[0] && cy <= devMaxWI[1] && cz <= devMaxWI[2]
                    && (cx * cy * cz) <= targetLocal
                    && cx <= Math.max(1, W) && cy <= Math.max(1, H) && cz <= Math.max(1, D)) {
                lx = cx;
                ly = cy;
                lz = cz;
                break;
            }
        }
        if (lx == 0) { // Fallback auf 1D, falls keine 3D-Kombi passt
            return plan(dev, W, H, D, false, alignBaseIndexTo32);
        }

        // Global 3D = pro Achse aufrunden
        int gx = (int) roundUpToMultipleLong(W, lx);
        int gy = (int) roundUpToMultipleLong(H, ly);
        int gz = (int) roundUpToMultipleLong(D, lz);
        long n = (long) W * H * D;

        float[] out = new float[(int) n];
        int baseIdxAligned = alignBaseIndexTo32 ? roundUpToMultiple((int) 0, 32) : 0;

        return new Plan(true, W, H, D, lx, ly, lz, gx, gy, gz, warp, cu, out, baseIdxAligned);
    }

    // ===== Helpers =====
    private static int roundUpToMultiple(int x, int m) {
        return ((x + m - 1) / m) * m;
    }

    private static int roundDownToMultiple(int x, int m) {
        return (x / m) * m;
    }

    private static long roundUpToMultipleLong(long x, int m) {
        long mm = m;
        return ((x + mm - 1) / mm) * mm;
    }
}
