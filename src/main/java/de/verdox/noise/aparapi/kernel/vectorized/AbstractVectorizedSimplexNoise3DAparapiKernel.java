package de.verdox.noise.aparapi.kernel.vectorized;

import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import jdk.incubator.vector.*;

public abstract class AbstractVectorizedSimplexNoise3DAparapiKernel extends AbstractSimplexNoise3DAparapiKernel {
    private static final VectorSpecies<Float> SF = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SI = IntVector.SPECIES_PREFERRED;

    // Floats
    private static final FloatVector V0 = FloatVector.zero(SF);
    private static final FloatVector V1 = FloatVector.broadcast(SF, 1f);
    private static final FloatVector VN1 = FloatVector.broadcast(SF, -1f);
    private static final FloatVector V_1_3 = FloatVector.broadcast(SF, 1f / 3f);
    private static final FloatVector V_1_6 = FloatVector.broadcast(SF, 1f / 6f);
    private static final FloatVector V_2_6 = FloatVector.broadcast(SF, 2f / 6f);
    private static final FloatVector V_3_6 = FloatVector.broadcast(SF, 3f / 6f);
    private static final FloatVector V_0_6 = FloatVector.broadcast(SF, 0.6f);
    private static final FloatVector V32 = FloatVector.broadcast(SF, 32f);

    // Ints
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

        final int base = baseIndex + (z * H + y) * W;

        // Inputs
        final FloatVector V_FREQ = FloatVector.broadcast(SF, frequency);
        final FloatVector V_X0 = FloatVector.broadcast(SF, baseX);
        final FloatVector V_Y0 = FloatVector.broadcast(SF, baseY);
        final FloatVector V_Z0 = FloatVector.broadcast(SF, baseZ);
        final FloatVector V_LANE = FloatVector.fromArray(SF, LANE, 0);

        final FloatVector vXin = V_X0.add(FloatVector.broadcast(SF, (float) x).add(V_LANE).mul(V_FREQ));
        final FloatVector vYin = V_Y0.add(FloatVector.broadcast(SF, (float) y).mul(V_FREQ));
        final FloatVector vZin = V_Z0.add(FloatVector.broadcast(SF, (float) z).mul(V_FREQ));

        // Skew / Unskew
        final FloatVector s = vXin.add(vYin).add(vZin).mul(V_1_3);
        final FloatVector xiS = vXin.add(s);
        final FloatVector yiS = vYin.add(s);
        final FloatVector ziS = vZin.add(s);

        final IntVector i = floorV(xiS);
        final IntVector j = floorV(yiS);
        final IntVector k = floorV(ziS);

        // I2F nur EINMAL:
        final FloatVector iF = (FloatVector) i.convert(VectorOperators.I2F, 0);
        final FloatVector jF = (FloatVector) j.convert(VectorOperators.I2F, 0);
        final FloatVector kF = (FloatVector) k.convert(VectorOperators.I2F, 0);

        final FloatVector t = iF.add(jF).add(kF).mul(V_1_6);
        final FloatVector x0 = vXin.sub(iF).add(t);
        final FloatVector y0 = vYin.sub(jF).add(t);
        final FloatVector z0 = vZin.sub(kF).add(t);

        // ==== Simplex-Corner-Wahl (6 Fälle) -> **IntVector** 0/1 für i1/j1/k1 und i2/j2/k2 ====
        VectorMask<Float> m_x_ge_y = x0.compare(VectorOperators.GE, y0);
        VectorMask<Float> m_y_ge_z = y0.compare(VectorOperators.GE, z0);
        VectorMask<Float> m_x_ge_z = x0.compare(VectorOperators.GE, z0);
        VectorMask<Float> m_y_lt_z = y0.compare(VectorOperators.LT, z0);
        VectorMask<Float> m_x_lt_z = x0.compare(VectorOperators.LT, z0);

        IntVector i1 = I0, j1 = I0, k1 = I0;
        IntVector i2 = I0, j2 = I0, k2 = I0;

        // Case 1: x>=y && y>=z
        VectorMask<Float> c1 = m_x_ge_y.and(m_y_ge_z);
        i1 = i1.blend(I1, c1.cast(SI));
        i2 = i2.blend(I1, c1.cast(SI));
        j2 = j2.blend(I1, c1.cast(SI));

        // Case 2: x>=y && x>=z && !(y>=z)
        VectorMask<Float> c2 = m_x_ge_y.and(m_x_ge_z).and(m_y_ge_z.not());
        i1 = i1.blend(I1, c2.cast(SI));
        i2 = i2.blend(I1, c2.cast(SI));
        k2 = k2.blend(I1, c2.cast(SI));

        // Case 3: x>=y && !(x>=z)
        VectorMask<Float> c3 = m_x_ge_y.and(m_x_ge_z.not());
        k1 = k1.blend(I1, c3.cast(SI));
        i2 = i2.blend(I1, c3.cast(SI));
        k2 = k2.blend(I1, c3.cast(SI));

        // Case 4: !(x>=y) && y<z
        VectorMask<Float> c4 = m_x_ge_y.not().and(m_y_lt_z);
        k1 = k1.blend(I1, c4.cast(SI));
        j2 = j2.blend(I1, c4.cast(SI));
        k2 = k2.blend(I1, c4.cast(SI));

        // Case 5: !(x>=y) && !(y<z) && x<z
        VectorMask<Float> c5 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z);
        j1 = j1.blend(I1, c5.cast(SI));
        j2 = j2.blend(I1, c5.cast(SI));
        k2 = k2.blend(I1, c5.cast(SI));

        // Case 6: sonst
        VectorMask<Float> c6 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z.not());
        j1 = j1.blend(I1, c6.cast(SI));
        i2 = i2.blend(I1, c6.cast(SI));
        j2 = j2.blend(I1, c6.cast(SI));

        // Einmalige I2F für Ecken
        final FloatVector i1F = (FloatVector) i1.convert(VectorOperators.I2F, 0);
        final FloatVector j1F = (FloatVector) j1.convert(VectorOperators.I2F, 0);
        final FloatVector k1F = (FloatVector) k1.convert(VectorOperators.I2F, 0);
        final FloatVector i2F = (FloatVector) i2.convert(VectorOperators.I2F, 0);
        final FloatVector j2F = (FloatVector) j2.convert(VectorOperators.I2F, 0);
        final FloatVector k2F = (FloatVector) k2.convert(VectorOperators.I2F, 0);

        // Ecken (wie Skalar)
        final FloatVector x1 = x0.sub(i1F).add(V_1_6);
        final FloatVector y1 = y0.sub(j1F).add(V_1_6);
        final FloatVector z1 = z0.sub(k1F).add(V_1_6);

        final FloatVector x2 = x0.sub(i2F).add(V_2_6);
        final FloatVector y2 = y0.sub(j2F).add(V_2_6);
        final FloatVector z2 = z0.sub(k2F).add(V_2_6);

        final FloatVector x3 = x0.sub(V1).add(V_3_6);
        final FloatVector y3 = y0.sub(V1).add(V_3_6);
        final FloatVector z3 = z0.sub(V1).add(V_3_6);

        // Hashes 0..255 → %12
        final IntVector ii = i.and(I255), jj = j.and(I255), kk = k.and(I255);

        // Vorstufen (klarere Datenabhängigkeiten, gleich viele intNoise-Calls, aber besser planbar)
        final IntVector nk0 = intNoiseV(kk);
        final IntVector nk1 = intNoiseV(kk.add(k1));
        final IntVector nk2 = intNoiseV(kk.add(k2));
        final IntVector nk3 = intNoiseV(kk.add(1));

        final IntVector nj0 = intNoiseV(jj.add(nk0));
        final IntVector nj1 = intNoiseV(jj.add(j1).add(nk1));
        final IntVector nj2 = intNoiseV(jj.add(j2).add(nk2));
        final IntVector nj3 = intNoiseV(jj.add(1).add(nk3));

        IntVector gi0 = mod12Fast(intNoiseV(ii.add(nj0)));
        IntVector gi1 = mod12Fast(intNoiseV(ii.add(i1).add(nj1)));
        IntVector gi2 = mod12Fast(intNoiseV(ii.add(i2).add(nj2)));
        IntVector gi3 = mod12Fast(intNoiseV(ii.add(1).add(nj3)));

        // t = 0.6 - (x^2+y^2+z^2)  (FMA nutzt)
        final FloatVector r0 = x0.fma(x0, y0.fma(y0, z0.mul(z0)));
        final FloatVector r1 = x1.fma(x1, y1.fma(y1, z1.mul(z1)));
        final FloatVector r2 = x2.fma(x2, y2.fma(y2, z2.mul(z2)));
        final FloatVector r3 = x3.fma(x3, y3.fma(y3, z3.mul(z3)));

        final FloatVector t0 = V_0_6.sub(r0).max(V0);
        final FloatVector t1 = V_0_6.sub(r1).max(V0);
        final FloatVector t2 = V_0_6.sub(r2).max(V0);
        final FloatVector t3 = V_0_6.sub(r3).max(V0);

        final FloatVector tt0 = t0.mul(t0);
        final FloatVector tt0_4 = tt0.mul(tt0);
        final FloatVector tt1 = t1.mul(t1);
        final FloatVector tt1_4 = tt1.mul(tt1);
        final FloatVector tt2 = t2.mul(t2);
        final FloatVector tt2_4 = tt2.mul(tt2);
        final FloatVector tt3 = t3.mul(t3);
        final FloatVector tt3_4 = tt3.mul(tt3);

        // dot(grad, x*)
        FloatVector n0 = tt0_4.mul(dotFromHashCorner(gi0, x0, y0, z0));
        FloatVector n1 = tt1_4.mul(dotFromHashCorner(gi1, x1, y1, z1));
        FloatVector n2 = tt2_4.mul(dotFromHashCorner(gi2, x2, y2, z2));
        FloatVector n3 = tt3_4.mul(dotFromHashCorner(gi3, x3, y3, z3));

        final FloatVector vOut = V32.mul(n0.add(n1).add(n2).add(n3));

        // Store (Tail mit Mask, wenn unterstützt)
        if (x + L <= W) {
            vOut.intoArray(noiseResult, base + x);
        } else {
            VectorMask<Float> m = SF.indexInRange(0, W - x);
            vOut.intoArray(noiseResult, base + x, m);
        }
    }

    // ===== Helpers =====

    private static FloatVector sq(FloatVector v) {
        return v.mul(v);
    }

// ===== Helpers =====

    private static IntVector floorV(FloatVector x) {
        IntVector t = (IntVector) x.convert(VectorOperators.F2I, 0);
        FloatVector tf = (FloatVector) t.convert(VectorOperators.I2F, 0);
        VectorMask<Float> needDec = x.compare(VectorOperators.LT, tf);
        IntVector adj = I0.blend(I1, needDec.cast(SI));
        return t.sub(adj);
    }

    /**
     * v % 12 via Multiply+Shift (exakt für 0..255).
     */
    private static IntVector mod12Fast(IntVector v) {
        IntVector q = v.mul(I2731).lanewise(VectorOperators.LSHR, 15);
        return v.sub(q.mul(I12));
    }

    /**
     * Hash 0..255 (wie skalar, aber vektorisiert).
     */
    private static IntVector intNoiseV(IntVector nIn) {
        IntVector a = nIn.add(463856334).lanewise(VectorOperators.ASHR, 13);
        IntVector b = nIn.add(575656768);
        IntVector v = a.lanewise(VectorOperators.XOR, b);
        IntVector n2 = v.mul(v);
        IntVector t = n2.mul(60493).add(19990303);
        return v.mul(t).add(1376312589).lanewise(VectorOperators.AND, I255);
    }

    /**
     * dot(grad(h), (x,y,z)) ohne Gather: Gruppen 0..3:(x,y), 4..7:(x,z), 8..11:(y,z) – Vorzeichen aus Bit0/Bit1
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

        FloatVector dot = dXY.blend(dXZ, mXZ.cast(SF)).blend(dYZ, mYZ.cast(SF));
        return dot;
    }
}
