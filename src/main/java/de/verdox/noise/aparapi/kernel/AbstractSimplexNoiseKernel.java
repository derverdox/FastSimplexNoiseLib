package de.verdox.noise.aparapi.kernel;

import com.aparapi.Kernel;
import de.verdox.noise.NoiseBackendBuilder;

public abstract class AbstractSimplexNoiseKernel extends Kernel {
    // === 3D Simplex Constants (wie gehabt) ===
    public static final float SKEWNESS_FACTOR   = 0.3333333333333333f;   // 1/3
    public static final float UNSKEWNESS_FACTOR = 0.16666666666666666f;  // 1/6
    public static final float UNSKEWNESS_FACTOR_2 = 2 * UNSKEWNESS_FACTOR;
    public static final float UNSKEWNESS_FACTOR_3 = 3 * UNSKEWNESS_FACTOR;
    public static final float ATTENUATION = 0.6f;

    // === 2D Simplex Constants ===
    public static final float SKEWNESS_FACTOR_2D   = 0.3660254037844386f;   // F2 = (sqrt(3)-1)/2
    public static final float UNSKEWNESS_FACTOR_2D = 0.21132486540518713f;  // G2 = (3-sqrt(3))/6

    public float baseX, baseY, baseZ, frequency;
    public int gridWidth, gridHeight, gridDepth, baseIndex;

    public final int noiseCalcMode;

    public float[] noiseResult = {0};

    // === Seed & Permutations ===
    public int seed = 1337; // default

    // Diese Arrays werden vor Ausführung gefüllt (Host-Seite), dann als @Constant ans Device kopiert.
    @Constant
    public short[] perm = new short[512];

    @Constant
    public short[] permMod12 = new short[512];

    // Mod-12 Lookup (konstant)
    @Constant
    public short[] MOD12 = new short[] {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3
    };

    // 12 Gradienten (3D), in 2D verwenden wir nur die ersten beiden Komponenten
    @Constant
    public final float[] grad3 = {
            1, 1, 0,  -1, 1, 0,   1, -1, 0,   -1, -1, 0,
            1, 0, 1,  -1, 0, 1,   1, 0, -1,  -1, 0, -1,
            0, 1, 1,   0, -1, 1,  0, 1, -1,   0, -1, -1
    };

    // Klassische Basistabelle (für deterministischen Default ohne setSeed())
    private static final short[] PERM_BASE = new short[] {
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,
            69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
            35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,
            48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,
            244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,
            116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,
            126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,
            152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,
            224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
            81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,
            45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };

    public AbstractSimplexNoiseKernel(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
        this.noiseCalcMode = noiseCalculationMode.ordinal();
        // Default-Permutation initialisieren (entspricht klassischer Implementierung)
        initDefaultPermutation();
    }

    // ===================== Setup / Binding =====================

    public void bindOutput(float[] out) { this.noiseResult = out; }

    public float[] getResult() { return noiseResult; }

    public void setParameters(float x0, float y0, float z0,
                              int width, int height, int depth,
                              float frequency, int baseIndex, long seed) {
        this.baseX = x0; this.baseY = y0; this.baseZ = z0;
        this.gridWidth = width; this.gridHeight = height; this.gridDepth = depth;
        this.frequency = frequency; this.baseIndex = baseIndex;
        this.setSeed(seed);
    }

    // ===================== Seeding =====================

    public void setSeed(long newSeed) {
        this.seed = (int) newSeed;

        // Basistabelle 0..255
        short[] p = new short[256];
        for (short i = 0; i < 256; i++) p[i] = i;

        // Fisher–Yates Shuffle mit einfachem, stabilem PRNG
        int s = (int) newSeed;
        for (int i = 255; i > 0; i--) {
            s = splitMix32(s);
            // (s >>> 1) um negatives mod zu vermeiden
            int j = (s >>> 1) % (i + 1);
            short tmp = p[i];
            p[i] = p[j];
            p[j] = tmp;
        }

        // Duplizieren & Mod12
        for (int i = 0; i < 512; i++) {
            int v = p[i & 255] & 0xFF;
            perm[i] = (short) v;
            permMod12[i] = (short) (v % 12);
        }
    }

    private void initDefaultPermutation() {
        for (int i = 0; i < 512; i++) {
            int v = PERM_BASE[i & 255] & 0xFF;
            perm[i] = (short) v;
            permMod12[i] = (short) (v % 12);
        }
    }

    private static int splitMix32(int x) {
        x += 0x9E3779B9;
        x = (x ^ (x >>> 15)) * 0x85EBCA6B;
        x = (x ^ (x >>> 13)) * 0xC2B2AE35;
        x = x ^ (x >>> 16);
        return x;
    }

    // ===================== 3D Simplex (Permutationspfad) =====================

    public float cpuScalarNoiseLookup(float xin, float yin, float zin) {
        float corner0, corner1, corner2, corner3;

        float skewFactor = (xin + yin + zin) * SKEWNESS_FACTOR;
        int skewedX = fastfloor(xin + skewFactor);
        int skewedY = fastfloor(yin + skewFactor);
        int skewedZ = fastfloor(zin + skewFactor);

        float unskewFactor = (skewedX + skewedY + skewedZ) * UNSKEWNESS_FACTOR;

        float cellOriginX = skewedX - unskewFactor;
        float cellOriginY = skewedY - unskewFactor;
        float cellOriginZ = skewedZ - unskewFactor;

        float x0 = xin - cellOriginX;
        float y0 = yin - cellOriginY;
        float z0 = zin - cellOriginZ;

        int rankX = 0, rankY = 0, rankZ = 0;
        if (x0 > y0) rankX++; else rankY++;
        if (x0 > z0) rankX++; else rankZ++;
        if (y0 > z0) rankY++; else rankZ++;

        int offset1X = (rankX >= 2) ? 1 : 0;
        int offset1Y = (rankY >= 2) ? 1 : 0;
        int offset1Z = (rankZ >= 2) ? 1 : 0;

        int offset2X = (rankX >= 1) ? 1 : 0;
        int offset2Y = (rankY >= 1) ? 1 : 0;
        int offset2Z = (rankZ >= 1) ? 1 : 0;

        float x1 = x0 - offset1X + UNSKEWNESS_FACTOR;
        float y1 = y0 - offset1Y + UNSKEWNESS_FACTOR;
        float z1 = z0 - offset1Z + UNSKEWNESS_FACTOR;

        float x2 = x0 - offset2X + UNSKEWNESS_FACTOR_2;
        float y2 = y0 - offset2Y + UNSKEWNESS_FACTOR_2;
        float z2 = z0 - offset2Z + UNSKEWNESS_FACTOR_2;

        float x3 = x0 - 1.0f + UNSKEWNESS_FACTOR_3;
        float y3 = y0 - 1.0f + UNSKEWNESS_FACTOR_3;
        float z3 = z0 - 1.0f + UNSKEWNESS_FACTOR_3;

        int ii = skewedX & 255;
        int jj = skewedY & 255;
        int kk = skewedZ & 255;

        int gradientIndex0 = permMod12[ii + p(jj + p(kk))] & 0xFF;
        int gradientIndex1 = permMod12[ii + offset1X + p(jj + offset1Y + p(kk + offset1Z))] & 0xFF;
        int gradientIndex2 = permMod12[ii + offset2X + p(jj + offset2Y + p(kk + offset2Z))] & 0xFF;
        int gradientIndex3 = permMod12[ii + 1 + p(jj + 1 + p(kk + 1))] & 0xFF;

        int b0 = 3 * gradientIndex0;
        int b1 = 3 * gradientIndex1;
        int b2 = 3 * gradientIndex2;
        int b3 = 3 * gradientIndex3;

        corner0 = corner(x0, y0, z0, b0);
        corner1 = corner(x1, y1, z1, b1);
        corner2 = corner(x2, y2, z2, b2);
        corner3 = corner(x3, y3, z3, b3);

        return 32.0f * (corner0 + corner1 + corner2 + corner3);
    }

    // ===================== 3D Simplex (ALU-only, Seeded Hash) =====================

    public float scalarNoiseAluOnly(float xin, float yin, float zin) {
        float corner0, corner1, corner2, corner3;

        float skewFactor = (xin + yin + zin) * SKEWNESS_FACTOR;
        int skewedX = fastfloor(xin + skewFactor);
        int skewedY = fastfloor(yin + skewFactor);
        int skewedZ = fastfloor(zin + skewFactor);

        float unskewFactor = (skewedX + skewedY + skewedZ) * UNSKEWNESS_FACTOR;

        float cellOriginX = skewedX - unskewFactor;
        float cellOriginY = skewedY - unskewFactor;
        float cellOriginZ = skewedZ - unskewFactor;

        float x0 = xin - cellOriginX;
        float y0 = yin - cellOriginY;
        float z0 = zin - cellOriginZ;

        int rankX = 0, rankY = 0, rankZ = 0;
        if (x0 > y0) rankX++; else rankY++;
        if (x0 > z0) rankX++; else rankZ++;
        if (y0 > z0) rankY++; else rankZ++;

        int offset1X = (rankX >= 2) ? 1 : 0;
        int offset1Y = (rankY >= 2) ? 1 : 0;
        int offset1Z = (rankZ >= 2) ? 1 : 0;

        int offset2X = (rankX >= 1) ? 1 : 0;
        int offset2Y = (rankY >= 1) ? 1 : 0;
        int offset2Z = (rankZ >= 1) ? 1 : 0;

        float x1 = x0 - offset1X + UNSKEWNESS_FACTOR;
        float y1 = y0 - offset1Y + UNSKEWNESS_FACTOR;
        float z1 = z0 - offset1Z + UNSKEWNESS_FACTOR;
        float x2 = x0 - offset2X + UNSKEWNESS_FACTOR_2;

        float y2 = y0 - offset2Y + UNSKEWNESS_FACTOR_2;
        float z2 = z0 - offset2Z + UNSKEWNESS_FACTOR_2;
        float x3 = x0 - 1.0f + UNSKEWNESS_FACTOR_3;

        float y3 = y0 - 1.0f + UNSKEWNESS_FACTOR_3;
        float z3 = z0 - 1.0f + UNSKEWNESS_FACTOR_3;

        int gradientIndex0 = MOD12[intNoiseSeeded(skewedX + intNoiseSeeded(skewedY + intNoiseSeeded(skewedZ, seed), seed), seed)];
        int gradientIndex1 = MOD12[intNoiseSeeded(skewedX + offset1X + intNoiseSeeded(skewedY + offset1Y + intNoiseSeeded(skewedZ + offset1Z, seed), seed), seed)];
        int gradientIndex2 = MOD12[intNoiseSeeded(skewedX + offset2X + intNoiseSeeded(skewedY + offset2Y + intNoiseSeeded(skewedZ + offset2Z, seed), seed), seed)];
        int gradientIndex3 = MOD12[intNoiseSeeded(skewedX + 1 + intNoiseSeeded(skewedY + 1 + intNoiseSeeded(skewedZ + 1, seed), seed), seed)];

        int b0 = 3 * gradientIndex0;
        int b1 = 3 * gradientIndex1;
        int b2 = 3 * gradientIndex2;
        int b3 = 3 * gradientIndex3;

        corner0 = corner(x0, y0, z0, b0);
        corner1 = corner(x1, y1, z1, b1);
        corner2 = corner(x2, y2, z2, b2);
        corner3 = corner(x3, y3, z3, b3);

        return 32.0f * (corner0 + corner1 + corner2 + corner3);
    }

    // ===================== 2D Simplex (Permutationstabellen-Pfad) =====================

    public float cpuScalarNoiseLookup2D(float xin, float yin) {
        float n0 = 0f, n1 = 0f, n2 = 0f;

        float s = (xin + yin) * SKEWNESS_FACTOR_2D;
        int i = fastfloor(xin + s);
        int j = fastfloor(yin + s);

        float t = (i + j) * UNSKEWNESS_FACTOR_2D;
        float X0 = i - t;
        float Y0 = j - t;

        float x0 = xin - X0;
        float y0 = yin - Y0;

        int i1 = 0, j1 = 0;
        if (x0 > y0) { i1 = 1; j1 = 0; } else { i1 = 0; j1 = 1; }

        float x1 = x0 - i1 + UNSKEWNESS_FACTOR_2D;
        float y1 = y0 - j1 + UNSKEWNESS_FACTOR_2D;
        float x2 = x0 - 1f + 2f * UNSKEWNESS_FACTOR_2D;
        float y2 = y0 - 1f + 2f * UNSKEWNESS_FACTOR_2D;

        int ii = i & 255;
        int jj = j & 255;

        int gi0 = (perm[ii + p(jj)] % 12) & 0xFF;
        int gi1 = (perm[ii + i1 + p(jj + j1)] % 12) & 0xFF;
        int gi2 = (perm[ii + 1 + p(jj + 1)] % 12) & 0xFF;

        float t0 = 0.5f - x0*x0 - y0*y0;
        if (t0 > 0f) { t0 *= t0; n0 = t0 * t0 * (grad3[3*gi0] * x0 + grad3[3*gi0 + 1] * y0); }

        float t1 = 0.5f - x1*x1 - y1*y1;
        if (t1 > 0f) { t1 *= t1; n1 = t1 * t1 * (grad3[3*gi1] * x1 + grad3[3*gi1 + 1] * y1); }

        float t2 = 0.5f - x2*x2 - y2*y2;
        if (t2 > 0f) { t2 *= t2; n2 = t2 * t2 * (grad3[3*gi2] * x2 + grad3[3*gi2 + 1] * y2); }

        return 70.0f * (n0 + n1 + n2);
    }

    // ===================== 2D Simplex (ALU-only, Seeded Hash) =====================

    public float scalarNoiseAluOnly2D(float xin, float yin) {
        float n0 = 0f, n1 = 0f, n2 = 0f;

        float s = (xin + yin) * SKEWNESS_FACTOR_2D;
        int i = fastfloor(xin + s);
        int j = fastfloor(yin + s);

        float t = (i + j) * UNSKEWNESS_FACTOR_2D;
        float X0 = i - t;
        float Y0 = j - t;

        float x0 = xin - X0;
        float y0 = yin - Y0;

        int i1 = 0, j1 = 0;
        if (x0 > y0) { i1 = 1; j1 = 0; } else { i1 = 0; j1 = 1; }

        float x1 = x0 - i1 + UNSKEWNESS_FACTOR_2D;
        float y1 = y0 - j1 + UNSKEWNESS_FACTOR_2D;
        float x2 = x0 - 1f + 2f * UNSKEWNESS_FACTOR_2D;
        float y2 = y0 - 1f + 2f * UNSKEWNESS_FACTOR_2D;

        int gi0 = MOD12[intNoiseSeeded(i     + intNoiseSeeded(j    , seed), seed)];
        int gi1 = MOD12[intNoiseSeeded(i+i1  + intNoiseSeeded(j+j1 , seed), seed)];
        int gi2 = MOD12[intNoiseSeeded(i+1   + intNoiseSeeded(j+1  , seed), seed)];

        float t0 = 0.5f - x0*x0 - y0*y0;
        if (t0 > 0f) { t0 *= t0; n0 = t0 * t0 * (grad3[3*gi0] * x0 + grad3[3*gi0 + 1] * y0); }

        float t1 = 0.5f - x1*x1 - y1*y1;
        if (t1 > 0f) { t1 *= t1; n1 = t1 * t1 * (grad3[3*gi1] * x1 + grad3[3*gi1 + 1] * y1); }

        float t2 = 0.5f - x2*x2 - y2*y2;
        if (t2 > 0f) { t2 *= t2; n2 = t2 * t2 * (grad3[3*gi2] * x2 + grad3[3*gi2 + 1] * y2); }

        return 70.0f * (n0 + n1 + n2);
    }

    // ===================== Utils =====================

    // Perm-Lookup (512 gespiegelt)
    private int p(int idx) { return perm[idx & 511] & 255; }

    private float corner(float x, float y, float z, int b) {
        float t = ATTENUATION - x * x - y * y - z * z;
        t = t > 0f ? t * t : 0f;
        float dot = grad3[b] * x + grad3[b + 1] * y + grad3[b + 2] * z;
        return (t * t) * dot;
    }

    // Seeded Hash, 0..255
    public int intNoiseSeeded(int n, int seed) {
        int x = n ^ seed;
        x = (x + 463856334) ^ (x >>> 13);
        x = x * (x * x * 60493 + 19990303) + 1376312589;
        return (x & 0x7fffffff) & 255;
    }

    // (Optional) Alte, unseeded Variante – falls noch irgendwo gebraucht:
    public int intNoise(int n) {
        n = (n + 463856334) >> 13 ^ (n + 575656768);
        return (n * (n * n * 60493 + 19990303) + 1376312589) & 0x7fffffff & 255;
    }

    public static int fastfloor(float x) {
        int xi = (int) x;
        return x < xi ? xi - 1 : xi;
    }

    public static float dot(float gx, float gy, float gz, float x, float y, float z) {
        return gx * x + gy * y + gz * z;
    }
}
