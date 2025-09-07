package de.verdox.noise.aparapi.kernel.cpu;

import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoiseKernel;
import jdk.incubator.vector.*;

public abstract class CPUVectorSimplexNoiseKernel extends AbstractSimplexNoiseKernel {
    private static final VectorSpecies<Float> SF = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SI = IntVector.SPECIES_PREFERRED;

    // ===== Float-Konstanten =====
    private static final FloatVector V0 = FloatVector.zero(SF);
    private static final FloatVector V1 = FloatVector.broadcast(SF, 1f);
    private static final FloatVector VN1 = FloatVector.broadcast(SF, -1f);
    private static final FloatVector V_1_3 = FloatVector.broadcast(SF, 1f / 3f);
    private static final FloatVector V_1_6 = FloatVector.broadcast(SF, 1f / 6f);
    private static final FloatVector V_2_6 = FloatVector.broadcast(SF, 2f / 6f);
    private static final FloatVector V_3_6 = FloatVector.broadcast(SF, 3f / 6f);
    private static final FloatVector V_0_6 = FloatVector.broadcast(SF, 0.6f);
    private static final FloatVector V32 = FloatVector.broadcast(SF, 32f);

    // 2D Simplex: F2/G2 & Attenuation
    private static final FloatVector V_F2 = FloatVector.broadcast(SF, 0.3660254037844386f);   // (√3-1)/2
    private static final FloatVector V_G2 = FloatVector.broadcast(SF, 0.21132486540518713f);  // (3-√3)/6
    private static final FloatVector V_0_5 = FloatVector.broadcast(SF, 0.5f);
    private static final FloatVector V70  = FloatVector.broadcast(SF, 70f);

    // ===== Int-Konstanten =====
    private static final IntVector I0 = IntVector.zero(SI);
    private static final IntVector I1 = IntVector.broadcast(SI, 1);
    private static final IntVector I2 = IntVector.broadcast(SI, 2);
    private static final IntVector I12 = IntVector.broadcast(SI, 12);
    private static final IntVector I255 = IntVector.broadcast(SI, 255);
    private static final IntVector I2731 = IntVector.broadcast(SI, 2731); // für /12

    // Lane-Index 0..L-1
    private static final float[] LANE;
    static {
        LANE = new float[SF.length()];
        for (int i = 0; i < LANE.length; i++) LANE[i] = i;
    }

    // Broadcast-Parameter
    protected FloatVector V_FREQ;
    protected FloatVector V_X0;
    protected FloatVector V_Y0;
    protected FloatVector V_Z0;
    protected FloatVector V_LANE;
    protected IntVector  I_SEED;  // neu: Seed für Hash

    public CPUVectorSimplexNoiseKernel(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
        super(noiseCalculationMode);
        V_LANE = FloatVector.fromArray(SF, LANE, 0);
    }

    @Override
    public void setParameters(float x0, float y0, float z0, int width, int height, int depth,
                              float frequency, int baseIndex, long seed) {
        super.setParameters(x0, y0, z0, width, height, depth, frequency, baseIndex, seed);
        V_FREQ = FloatVector.broadcast(SF, frequency);
        V_X0   = FloatVector.broadcast(SF, baseX);
        V_Y0   = FloatVector.broadcast(SF, baseY);
        V_Z0   = FloatVector.broadcast(SF, baseZ);
        I_SEED = IntVector.broadcast(SI, (int) seed); // neu
    }

    // ========================= Simple (vektorisiert) =========================
    public abstract static class Simple extends CPUVectorSimplexNoiseKernel {
        public Simple(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
            super(noiseCalculationMode);
        }

        // ------------------------------ 3D ------------------------------
        public static class Noise3DIndexing1D extends CPUVectorSimplexNoiseKernel.Simple {
            public Noise3DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                final int L = SF.length();
                final int W = gridWidth, H = gridHeight, D = gridDepth;
                final int Wv = (W + L - 1) / L;

                final int gid = getGlobalId(0);
                final int total = Wv * H * D;
                if (gid >= total) return;

                // 1D -> (z,y,xBlock)
                int tmp = gid;
                final int z = tmp / (Wv * H);
                tmp -= z * (Wv * H);
                final int y = tmp / Wv;
                final int xb = tmp - y * Wv;
                final int x = xb * L;

                // Basisindex der Zeile (x-major)
                final int base = baseIndex + (z * H + y) * W;

                final FloatVector vXin = V_X0.add(FloatVector.broadcast(SF, (float) x).add(V_LANE).mul(V_FREQ));
                final FloatVector vYin = V_Y0.add(FloatVector.broadcast(SF, (float) y).mul(V_FREQ));
                final FloatVector vZin = V_Z0.add(FloatVector.broadcast(SF, (float) z).mul(V_FREQ));

                // Skew / Unskew (3D)
                final FloatVector s = vXin.add(vYin).add(vZin).mul(V_1_3);
                final FloatVector xiS = vXin.add(s);
                final FloatVector yiS = vYin.add(s);
                final FloatVector ziS = vZin.add(s);

                final IntVector i = floorV(xiS);
                final IntVector j = floorV(yiS);
                final IntVector k = floorV(ziS);

                final FloatVector iF = (FloatVector) i.convert(VectorOperators.I2F, 0);
                final FloatVector jF = (FloatVector) j.convert(VectorOperators.I2F, 0);
                final FloatVector kF = (FloatVector) k.convert(VectorOperators.I2F, 0);

                final FloatVector t = iF.add(jF).add(kF).mul(V_1_6);
                final FloatVector x0 = vXin.sub(iF).add(t);
                final FloatVector y0 = vYin.sub(jF).add(t);
                final FloatVector z0 = vZin.sub(kF).add(t);

                // Eckenwahl (6 Fälle)
                VectorMask<Float> m_x_ge_y = x0.compare(VectorOperators.GE, y0);
                VectorMask<Float> m_y_ge_z = y0.compare(VectorOperators.GE, z0);
                VectorMask<Float> m_x_ge_z = x0.compare(VectorOperators.GE, z0);
                VectorMask<Float> m_y_lt_z = y0.compare(VectorOperators.LT, z0);
                VectorMask<Float> m_x_lt_z = x0.compare(VectorOperators.LT, z0);

                IntVector i1 = I0, j1 = I0, k1 = I0;
                IntVector i2 = I0, j2 = I0, k2 = I0;

                VectorMask<Float> c1 = m_x_ge_y.and(m_y_ge_z);
                i1 = i1.blend(I1, c1.cast(SI)); i2 = i2.blend(I1, c1.cast(SI)); j2 = j2.blend(I1, c1.cast(SI));

                VectorMask<Float> c2 = m_x_ge_y.and(m_x_ge_z).and(m_y_ge_z.not());
                i1 = i1.blend(I1, c2.cast(SI)); i2 = i2.blend(I1, c2.cast(SI)); k2 = k2.blend(I1, c2.cast(SI));

                VectorMask<Float> c3 = m_x_ge_y.and(m_x_ge_z.not());
                k1 = k1.blend(I1, c3.cast(SI)); i2 = i2.blend(I1, c3.cast(SI)); k2 = k2.blend(I1, c3.cast(SI));

                VectorMask<Float> c4 = m_x_ge_y.not().and(m_y_lt_z);
                k1 = k1.blend(I1, c4.cast(SI)); j2 = j2.blend(I1, c4.cast(SI)); k2 = k2.blend(I1, c4.cast(SI));

                VectorMask<Float> c5 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z);
                j1 = j1.blend(I1, c5.cast(SI)); j2 = j2.blend(I1, c5.cast(SI)); k2 = k2.blend(I1, c5.cast(SI));

                VectorMask<Float> c6 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z.not());
                j1 = j1.blend(I1, c6.cast(SI)); i2 = i2.blend(I1, c6.cast(SI)); j2 = j2.blend(I1, c6.cast(SI));

                final FloatVector i1F = (FloatVector) i1.convert(VectorOperators.I2F, 0);
                final FloatVector j1F = (FloatVector) j1.convert(VectorOperators.I2F, 0);
                final FloatVector k1F = (FloatVector) k1.convert(VectorOperators.I2F, 0);
                final FloatVector i2F = (FloatVector) i2.convert(VectorOperators.I2F, 0);
                final FloatVector j2F = (FloatVector) j2.convert(VectorOperators.I2F, 0);
                final FloatVector k2F = (FloatVector) k2.convert(VectorOperators.I2F, 0);

                final FloatVector x1 = x0.sub(i1F).add(V_1_6);
                final FloatVector y1 = y0.sub(j1F).add(V_1_6);
                final FloatVector z1 = z0.sub(k1F).add(V_1_6);

                final FloatVector x2 = x0.sub(i2F).add(V_2_6);
                final FloatVector y2 = y0.sub(j2F).add(V_2_6);
                final FloatVector z2 = z0.sub(k2F).add(V_2_6);

                final FloatVector x3 = x0.sub(V1).add(V_3_6);
                final FloatVector y3 = y0.sub(V1).add(V_3_6);
                final FloatVector z3 = z0.sub(V1).add(V_3_6);

                final IntVector ii = i.and(I255), jj = j.and(I255), kk = k.and(I255);

                // Seeded Hash-Kaskade (3D)
                final IntVector nk0 = intNoiseSeededV(kk, I_SEED);
                final IntVector nk1 = intNoiseSeededV(kk.add(k1), I_SEED);
                final IntVector nk2 = intNoiseSeededV(kk.add(k2), I_SEED);
                final IntVector nk3 = intNoiseSeededV(kk.add(1),   I_SEED);

                final IntVector nj0 = intNoiseSeededV(jj.add(nk0), I_SEED);
                final IntVector nj1 = intNoiseSeededV(jj.add(j1).add(nk1), I_SEED);
                final IntVector nj2 = intNoiseSeededV(jj.add(j2).add(nk2), I_SEED);
                final IntVector nj3 = intNoiseSeededV(jj.add(1).add(nk3), I_SEED);

                IntVector gi0 = mod12Fast(intNoiseSeededV(ii.add(nj0), I_SEED));
                IntVector gi1 = mod12Fast(intNoiseSeededV(ii.add(i1).add(nj1), I_SEED));
                IntVector gi2 = mod12Fast(intNoiseSeededV(ii.add(i2).add(nj2), I_SEED));
                IntVector gi3 = mod12Fast(intNoiseSeededV(ii.add(1).add(nj3), I_SEED));

                final FloatVector r0 = x0.fma(x0, y0.fma(y0, z0.mul(z0)));
                final FloatVector r1 = x1.fma(x1, y1.fma(y1, z1.mul(z1)));
                final FloatVector r2 = x2.fma(x2, y2.fma(y2, z2.mul(z2)));
                final FloatVector r3 = x3.fma(x3, y3.fma(y3, z3.mul(z3)));

                final FloatVector t0 = V_0_6.sub(r0).max(V0);
                final FloatVector t1 = V_0_6.sub(r1).max(V0);
                final FloatVector t2 = V_0_6.sub(r2).max(V0);
                final FloatVector t3 = V_0_6.sub(r3).max(V0);

                final FloatVector tt0_4 = t0.mul(t0).mul(t0.mul(t0));
                final FloatVector tt1_4 = t1.mul(t1).mul(t1.mul(t1));
                final FloatVector tt2_4 = t2.mul(t2).mul(t2.mul(t2));
                final FloatVector tt3_4 = t3.mul(t3).mul(t3.mul(t3));

                FloatVector n0 = tt0_4.mul(dotFromHashCorner(gi0, x0, y0, z0));
                FloatVector n1 = tt1_4.mul(dotFromHashCorner(gi1, x1, y1, z1));
                FloatVector n2 = tt2_4.mul(dotFromHashCorner(gi2, x2, y2, z2));
                FloatVector n3 = tt3_4.mul(dotFromHashCorner(gi3, x3, y3, z3));

                final FloatVector vOut = V32.mul(n0.add(n1).add(n2).add(n3));

                if (x + L <= W) {
                    vOut.intoArray(noiseResult, base + x);
                } else {
                    VectorMask<Float> m = SF.indexInRange(0, W - x);
                    vOut.intoArray(noiseResult, base + x, m);
                }
            }
        }

        public static class Noise3DIndexing3D extends CPUVectorSimplexNoiseKernel.Simple {
            public Noise3DIndexing3D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                final int L  = SF.length();
                final int W  = gridWidth, H = gridHeight, D = gridDepth;
                final int xb = getGlobalId(0); // Vektor-Kachel entlang X
                final int y  = getGlobalId(1);
                final int z  = getGlobalId(2);

                if (y >= H || z >= D) return;

                final int x = xb * L;
                if (x >= W) return;

                final int base = baseIndex + (z * H + y) * W;

                final FloatVector vXin = V_X0.add(FloatVector.broadcast(SF, (float) x).add(V_LANE).mul(V_FREQ));
                final FloatVector vYin = V_Y0.add(FloatVector.broadcast(SF, (float) y).mul(V_FREQ));
                final FloatVector vZin = V_Z0.add(FloatVector.broadcast(SF, (float) z).mul(V_FREQ));

                final FloatVector s  = vXin.add(vYin).add(vZin).mul(V_1_3);
                final FloatVector xiS = vXin.add(s);
                final FloatVector yiS = vYin.add(s);
                final FloatVector ziS = vZin.add(s);

                final IntVector i = floorV(xiS);
                final IntVector j = floorV(yiS);
                final IntVector k = floorV(ziS);

                final FloatVector iF = (FloatVector) i.convert(VectorOperators.I2F, 0);
                final FloatVector jF = (FloatVector) j.convert(VectorOperators.I2F, 0);
                final FloatVector kF = (FloatVector) k.convert(VectorOperators.I2F, 0);

                final FloatVector t  = iF.add(jF).add(kF).mul(V_1_6);
                final FloatVector x0 = vXin.sub(iF).add(t);
                final FloatVector y0 = vYin.sub(jF).add(t);
                final FloatVector z0 = vZin.sub(kF).add(t);

                VectorMask<Float> m_x_ge_y = x0.compare(VectorOperators.GE, y0);
                VectorMask<Float> m_y_ge_z = y0.compare(VectorOperators.GE, z0);
                VectorMask<Float> m_x_ge_z = x0.compare(VectorOperators.GE, z0);
                VectorMask<Float> m_y_lt_z = y0.compare(VectorOperators.LT, z0);
                VectorMask<Float> m_x_lt_z = x0.compare(VectorOperators.LT, z0);

                IntVector i1 = I0, j1 = I0, k1 = I0;
                IntVector i2 = I0, j2 = I0, k2 = I0;

                VectorMask<Float> c1 = m_x_ge_y.and(m_y_ge_z);
                i1 = i1.blend(I1, c1.cast(SI)); i2 = i2.blend(I1, c1.cast(SI)); j2 = j2.blend(I1, c1.cast(SI));

                VectorMask<Float> c2 = m_x_ge_y.and(m_x_ge_z).and(m_y_ge_z.not());
                i1 = i1.blend(I1, c2.cast(SI)); i2 = i2.blend(I1, c2.cast(SI)); k2 = k2.blend(I1, c2.cast(SI));

                VectorMask<Float> c3 = m_x_ge_y.and(m_x_ge_z.not());
                k1 = k1.blend(I1, c3.cast(SI)); i2 = i2.blend(I1, c3.cast(SI)); k2 = k2.blend(I1, c3.cast(SI));

                VectorMask<Float> c4 = m_x_ge_y.not().and(m_y_lt_z);
                k1 = k1.blend(I1, c4.cast(SI)); j2 = j2.blend(I1, c4.cast(SI)); k2 = k2.blend(I1, c4.cast(SI));

                VectorMask<Float> c5 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z);
                j1 = j1.blend(I1, c5.cast(SI)); j2 = j2.blend(I1, c5.cast(SI)); k2 = k2.blend(I1, c5.cast(SI));

                VectorMask<Float> c6 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z.not());
                j1 = j1.blend(I1, c6.cast(SI)); i2 = i2.blend(I1, c6.cast(SI)); j2 = j2.blend(I1, c6.cast(SI));

                final FloatVector i1F = (FloatVector) i1.convert(VectorOperators.I2F, 0);
                final FloatVector j1F = (FloatVector) j1.convert(VectorOperators.I2F, 0);
                final FloatVector k1F = (FloatVector) k1.convert(VectorOperators.I2F, 0);
                final FloatVector i2F = (FloatVector) i2.convert(VectorOperators.I2F, 0);
                final FloatVector j2F = (FloatVector) j2.convert(VectorOperators.I2F, 0);
                final FloatVector k2F = (FloatVector) k2.convert(VectorOperators.I2F, 0);

                final FloatVector x1 = x0.sub(i1F).add(V_1_6);
                final FloatVector y1 = y0.sub(j1F).add(V_1_6);
                final FloatVector z1 = z0.sub(k1F).add(V_1_6);

                final FloatVector x2 = x0.sub(i2F).add(V_2_6);
                final FloatVector y2 = y0.sub(j2F).add(V_2_6);
                final FloatVector z2 = z0.sub(k2F).add(V_2_6);

                final FloatVector x3 = x0.sub(V1).add(V_3_6);
                final FloatVector y3 = y0.sub(V1).add(V_3_6);
                final FloatVector z3 = z0.sub(V1).add(V_3_6);

                final IntVector ii = i.and(I255), jj = j.and(I255), kk = k.and(I255);

                final IntVector nk0 = intNoiseSeededV(kk, I_SEED);
                final IntVector nk1 = intNoiseSeededV(kk.add(k1), I_SEED);
                final IntVector nk2 = intNoiseSeededV(kk.add(k2), I_SEED);
                final IntVector nk3 = intNoiseSeededV(kk.add(1),   I_SEED);

                final IntVector nj0 = intNoiseSeededV(jj.add(nk0), I_SEED);
                final IntVector nj1 = intNoiseSeededV(jj.add(j1).add(nk1), I_SEED);
                final IntVector nj2 = intNoiseSeededV(jj.add(j2).add(nk2), I_SEED);
                final IntVector nj3 = intNoiseSeededV(jj.add(1).add(nk3), I_SEED);

                IntVector gi0 = mod12Fast(intNoiseSeededV(ii.add(nj0), I_SEED));
                IntVector gi1 = mod12Fast(intNoiseSeededV(ii.add(i1).add(nj1), I_SEED));
                IntVector gi2 = mod12Fast(intNoiseSeededV(ii.add(i2).add(nj2), I_SEED));
                IntVector gi3 = mod12Fast(intNoiseSeededV(ii.add(1).add(nj3), I_SEED));

                final FloatVector r0 = x0.fma(x0, y0.fma(y0, z0.mul(z0)));
                final FloatVector r1 = x1.fma(x1, y1.fma(y1, z1.mul(z1)));
                final FloatVector r2 = x2.fma(x2, y2.fma(y2, z2.mul(z2)));
                final FloatVector r3 = x3.fma(x3, y3.fma(y3, z3.mul(z3)));

                final FloatVector t0 = V_0_6.sub(r0).max(V0);
                final FloatVector t1 = V_0_6.sub(r1).max(V0);
                final FloatVector t2 = V_0_6.sub(r2).max(V0);
                final FloatVector t3 = V_0_6.sub(r3).max(V0);

                final FloatVector tt0_4 = t0.mul(t0).mul(t0.mul(t0));
                final FloatVector tt1_4 = t1.mul(t1).mul(t1.mul(t1));
                final FloatVector tt2_4 = t2.mul(t2).mul(t2.mul(t2));
                final FloatVector tt3_4 = t3.mul(t3).mul(t3.mul(t3));

                FloatVector n0 = tt0_4.mul(dotFromHashCorner(gi0, x0, y0, z0));
                FloatVector n1 = tt1_4.mul(dotFromHashCorner(gi1, x1, y1, z1));
                FloatVector n2 = tt2_4.mul(dotFromHashCorner(gi2, x2, y2, z2));
                FloatVector n3 = tt3_4.mul(dotFromHashCorner(gi3, x3, y3, z3));

                final FloatVector vOut = V32.mul(n0.add(n1).add(n2).add(n3));

                if (x + L <= W) {
                    vOut.intoArray(noiseResult, base + x);
                } else {
                    VectorMask<Float> m = SF.indexInRange(0, W - x);
                    vOut.intoArray(noiseResult, base + x, m);
                }
            }
        }

        // ------------------------------ 2D (x,z) ------------------------------
        /** 1D-Launch → (xBlock,z), verarbeitet X in Vektor-Kacheln (Lanes) */
        public static class Noise2DIndexing1D extends CPUVectorSimplexNoiseKernel.Simple {
            public Noise2DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                final int L = SF.length();
                final int W = gridWidth, D = gridDepth;
                final int Wv = (W + L - 1) / L;

                final int gid = getGlobalId(0);
                final int total = Wv * D;
                if (gid >= total) return;

                final int z  = gid / Wv;
                final int xb = gid - z * Wv;
                final int x  = xb * L;

                final int base = baseIndex + z * W;

                final FloatVector vXin = V_X0.add(FloatVector.broadcast(SF, (float) x).add(V_LANE).mul(V_FREQ));
                final FloatVector vZin = V_Z0.add(FloatVector.broadcast(SF, (float) z).mul(V_FREQ));

                // 2D Skew/Unskew (X,Z)
                final FloatVector s  = vXin.add(vZin).mul(V_F2);
                final IntVector i    = floorV(vXin.add(s));
                final IntVector k    = floorV(vZin.add(s)); // "k" = z-Index

                final FloatVector iF = (FloatVector) i.convert(VectorOperators.I2F, 0);
                final FloatVector kF = (FloatVector) k.convert(VectorOperators.I2F, 0);

                final FloatVector t  = iF.add(kF).mul(V_G2);
                final FloatVector x0 = vXin.sub(iF).add(t);
                final FloatVector z0 = vZin.sub(kF).add(t);

                // Eckenwahl (2D): i1/j1 -> hier i1/k1
                VectorMask<Float> m_x_gt_z = x0.compare(VectorOperators.GT, z0);
                IntVector i1 = I0.blend(I1, m_x_gt_z.cast(SI));
                IntVector k1 = I0.blend(I1, m_x_gt_z.not().cast(SI));

                final FloatVector i1F = (FloatVector) i1.convert(VectorOperators.I2F, 0);
                final FloatVector k1F = (FloatVector) k1.convert(VectorOperators.I2F, 0);

                final FloatVector x1 = x0.sub(i1F).add(V_G2);
                final FloatVector z1 = z0.sub(k1F).add(V_G2);
                final FloatVector x2 = x0.sub(V1).add(V_G2.add(V_G2)); // x0 - 1 + 2*G2
                final FloatVector z2 = z0.sub(V1).add(V_G2.add(V_G2));

                final IntVector ii = i.and(I255);
                final IntVector kk = k.and(I255);

                // Seeded Hash-Kaskade (2D)
                final IntVector nk0 = intNoiseSeededV(kk, I_SEED);
                final IntVector nk1 = intNoiseSeededV(kk.add(k1), I_SEED);
                final IntVector nk2 = intNoiseSeededV(kk.add(1),   I_SEED);

                IntVector gi0 = mod12Fast(intNoiseSeededV(ii.add(nk0), I_SEED));
                IntVector gi1 = mod12Fast(intNoiseSeededV(ii.add(i1).add(nk1), I_SEED));
                IntVector gi2 = mod12Fast(intNoiseSeededV(ii.add(1).add(nk2),   I_SEED));

                // t = 0.5 - (x^2+z^2)
                final FloatVector r0 = x0.fma(x0, z0.mul(z0));
                final FloatVector r1 = x1.fma(x1, z1.mul(z1));
                final FloatVector r2 = x2.fma(x2, z2.mul(z2));

                final FloatVector t0 = V_0_5.sub(r0).max(V0);
                final FloatVector t1 = V_0_5.sub(r1).max(V0);
                final FloatVector t2 = V_0_5.sub(r2).max(V0);

                final FloatVector tt0_4 = t0.mul(t0).mul(t0.mul(t0));
                final FloatVector tt1_4 = t1.mul(t1).mul(t1.mul(t1));
                final FloatVector tt2_4 = t2.mul(t2).mul(t2.mul(t2));

                FloatVector n0 = tt0_4.mul(dotFromHash2D_XZ(gi0, x0, z0));
                FloatVector n1 = tt1_4.mul(dotFromHash2D_XZ(gi1, x1, z1));
                FloatVector n2 = tt2_4.mul(dotFromHash2D_XZ(gi2, x2, z2));

                final FloatVector vOut = V70.mul(n0.add(n1).add(n2));

                if (x + L <= W) {
                    vOut.intoArray(noiseResult, base + x);
                } else {
                    VectorMask<Float> m = SF.indexInRange(0, W - x);
                    vOut.intoArray(noiseResult, base + x, m);
                }
            }
        }

        /** 2D-Launch (xb,z) – X in Kacheln, Z als Zeilenindex */
        public static class Noise2DIndexing2D extends CPUVectorSimplexNoiseKernel.Simple {
            public Noise2DIndexing2D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                final int L  = SF.length();
                final int W  = gridWidth, D = gridDepth;
                final int xb = getGlobalId(0);
                final int z  = getGlobalId(1);

                if (z >= D) return;

                final int x = xb * L;
                if (x >= W) return;

                final int base = baseIndex + z * W;

                final FloatVector vXin = V_X0.add(FloatVector.broadcast(SF, (float) x).add(V_LANE).mul(V_FREQ));
                final FloatVector vZin = V_Z0.add(FloatVector.broadcast(SF, (float) z).mul(V_FREQ));

                // 2D Skew/Unskew
                final FloatVector s  = vXin.add(vZin).mul(V_F2);
                final IntVector i    = floorV(vXin.add(s));
                final IntVector k    = floorV(vZin.add(s));

                final FloatVector iF = (FloatVector) i.convert(VectorOperators.I2F, 0);
                final FloatVector kF = (FloatVector) k.convert(VectorOperators.I2F, 0);

                final FloatVector t  = iF.add(kF).mul(V_G2);
                final FloatVector x0 = vXin.sub(iF).add(t);
                final FloatVector z0 = vZin.sub(kF).add(t);

                VectorMask<Float> m_x_gt_z = x0.compare(VectorOperators.GT, z0);
                IntVector i1 = I0.blend(I1, m_x_gt_z.cast(SI));
                IntVector k1 = I0.blend(I1, m_x_gt_z.not().cast(SI));

                final FloatVector i1F = (FloatVector) i1.convert(VectorOperators.I2F, 0);
                final FloatVector k1F = (FloatVector) k1.convert(VectorOperators.I2F, 0);

                final FloatVector x1 = x0.sub(i1F).add(V_G2);
                final FloatVector z1 = z0.sub(k1F).add(V_G2);
                final FloatVector x2 = x0.sub(V1).add(V_G2.add(V_G2));
                final FloatVector z2 = z0.sub(V1).add(V_G2.add(V_G2));

                final IntVector ii = i.and(I255);
                final IntVector kk = k.and(I255);

                final IntVector nk0 = intNoiseSeededV(kk, I_SEED);
                final IntVector nk1 = intNoiseSeededV(kk.add(k1), I_SEED);
                final IntVector nk2 = intNoiseSeededV(kk.add(1),   I_SEED);

                IntVector gi0 = mod12Fast(intNoiseSeededV(ii.add(nk0), I_SEED));
                IntVector gi1 = mod12Fast(intNoiseSeededV(ii.add(i1).add(nk1), I_SEED));
                IntVector gi2 = mod12Fast(intNoiseSeededV(ii.add(1).add(nk2),   I_SEED));

                final FloatVector r0 = x0.fma(x0, z0.mul(z0));
                final FloatVector r1 = x1.fma(x1, z1.mul(z1));
                final FloatVector r2 = x2.fma(x2, z2.mul(z2));

                final FloatVector t0 = V_0_5.sub(r0).max(V0);
                final FloatVector t1 = V_0_5.sub(r1).max(V0);
                final FloatVector t2 = V_0_5.sub(r2).max(V0);

                final FloatVector tt0_4 = t0.mul(t0).mul(t0.mul(t0));
                final FloatVector tt1_4 = t1.mul(t1).mul(t1.mul(t1));
                final FloatVector tt2_4 = t2.mul(t2).mul(t2.mul(t2));

                FloatVector n0 = tt0_4.mul(dotFromHash2D_XZ(gi0, x0, z0));
                FloatVector n1 = tt1_4.mul(dotFromHash2D_XZ(gi1, x1, z1));
                FloatVector n2 = tt2_4.mul(dotFromHash2D_XZ(gi2, x2, z2));

                final FloatVector vOut = V70.mul(n0.add(n1).add(n2));

                if (x + L <= W) {
                    vOut.intoArray(noiseResult, base + x);
                } else {
                    VectorMask<Float> m = SF.indexInRange(0, W - x);
                    vOut.intoArray(noiseResult, base + x, m);
                }
            }
        }
    }

    // ========================= Helpers =========================

    private static IntVector floorV(FloatVector x) {
        IntVector t = (IntVector) x.convert(VectorOperators.F2I, 0);
        FloatVector tf = (FloatVector) t.convert(VectorOperators.I2F, 0);
        VectorMask<Float> needDec = x.compare(VectorOperators.LT, tf);
        IntVector adj = I0.blend(I1, needDec.cast(SI));
        return t.sub(adj);
    }

    /** v % 12 via Multiply+Shift (exakt für 0..255). */
    private static IntVector mod12Fast(IntVector v) {
        IntVector q = v.mul(I2731).lanewise(VectorOperators.LSHR, 15);
        return v.sub(q.mul(I12));
    }

    /** Seeded Hash 0..255 (vektorisiert). */
    private static IntVector intNoiseSeededV(IntVector nIn, IntVector seedV) {
        // x = (n ^ seed)
        IntVector x = nIn.lanewise(VectorOperators.XOR, seedV);
        x = x.add(463856334).lanewise(VectorOperators.ASHR, 13)
                .lanewise(VectorOperators.XOR, nIn.add(575656768)); // beibehaltene Struktur, aber „gesalted“
        IntVector n2 = x.mul(x);
        IntVector t  = n2.mul(60493).add(19990303);
        return x.mul(t).add(1376312589).lanewise(VectorOperators.AND, I255);
    }

    /**
     * dot(grad(h), (x,y,z)) – gleiche Kodierung wie in deiner 3D-Skalaren Version:
     * Gruppen 0..3:(x,y), 4..7:(x,z), 8..11:(y,z), Vorzeichen aus Bit0/Bit1.
     */
    private static FloatVector dotFromHashCorner(IntVector h, FloatVector x, FloatVector y, FloatVector z) {
        IntVector grp = h.lanewise(VectorOperators.LSHR, 2);

        VectorMask<Integer> b0 = h.and(1).compare(VectorOperators.NE, I0);
        VectorMask<Integer> b1 = h.and(2).compare(VectorOperators.NE, I0);
        FloatVector s0 = V1.blend(VN1, b0.cast(SF));
        FloatVector s1 = V1.blend(VN1, b1.cast(SF));

        FloatVector dXY = s0.mul(x).add(s1.mul(y));
        FloatVector dXZ = s0.mul(x).add(s1.mul(z));
        FloatVector dYZ = s0.mul(y).add(s1.mul(z));

        VectorMask<Integer> mXZ = grp.compare(VectorOperators.EQ, I1);
        VectorMask<Integer> mYZ = grp.compare(VectorOperators.EQ, IntVector.broadcast(SI, 2));

        return dXY.blend(dXZ, mXZ.cast(SF)).blend(dYZ, mYZ.cast(SF));
    }

    /** 2D-Variante (X,Z): nutzt nur Vorzeichenbits und immer die XZ-Achse. */
    private static FloatVector dotFromHash2D_XZ(IntVector h, FloatVector x, FloatVector z) {
        VectorMask<Integer> b0 = h.and(1).compare(VectorOperators.NE, I0);
        VectorMask<Integer> b1 = h.and(2).compare(VectorOperators.NE, I0);
        FloatVector s0 = V1.blend(VN1, b0.cast(SF));
        FloatVector s1 = V1.blend(VN1, b1.cast(SF));
        return s0.mul(x).add(s1.mul(z));
    }
}
