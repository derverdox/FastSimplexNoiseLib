package de.verdox.noise.aparapi.kernel.vectorized;

import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import jdk.incubator.vector.*;

public abstract class AbstractVectorizedSimplexNoise3DAparapiKernel extends AbstractSimplexNoise3DAparapiKernel {
    private static final VectorSpecies<Float>   SF = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SI = IntVector.SPECIES_PREFERRED;

    // Floats
    private static final FloatVector V0    = FloatVector.zero(SF);
    private static final FloatVector V1    = FloatVector.broadcast(SF, 1f);
    private static final FloatVector VN1   = FloatVector.broadcast(SF, -1f);
    private static final FloatVector V_1_3 = FloatVector.broadcast(SF, 1f/3f);
    private static final FloatVector V_1_6 = FloatVector.broadcast(SF, 1f/6f);
    private static final FloatVector V_2_6 = FloatVector.broadcast(SF, 2f/6f);
    private static final FloatVector V_3_6 = FloatVector.broadcast(SF, 3f/6f);
    private static final FloatVector V_0_6 = FloatVector.broadcast(SF, 0.6f);
    private static final FloatVector V32   = FloatVector.broadcast(SF, 32f);

    // Ints
    private static final IntVector I0    = IntVector.zero(SI);
    private static final IntVector I1    = IntVector.broadcast(SI, 1);
    private static final IntVector I2    = IntVector.broadcast(SI, 2);
    private static final IntVector I12   = IntVector.broadcast(SI, 12);
    private static final IntVector I255  = IntVector.broadcast(SI, 255);
    private static final IntVector I2731 = IntVector.broadcast(SI, 2731); // für /12

    // Lane-Index 0..L-1
    private static final float[] LANE;
    static {
        LANE = new float[SF.length()];
        for (int i = 0; i < LANE.length; i++) LANE[i] = i;
    }

    @Override public void run() {
        final int L = SF.length();
        final int W = argWidth, H = argHeight, D = argDepth;
        final int Wv = (W + L - 1) / L;

        final int gid = getGlobalId(0);
        final int total = Wv * H * D;
        if (gid >= total) return;

        // 1D -> (z,y,xBlock)
        int tmp = gid;
        final int z  = tmp / (Wv * H);
        tmp         -= z * (Wv * H);
        final int y  = tmp / Wv;
        final int xb = tmp - y * Wv;
        final int x  = xb * L;

        final int base = argBase + (z * H + y) * W;

        // Inputs
        final FloatVector V_FREQ = FloatVector.broadcast(SF, argFrequency);
        final FloatVector V_X0   = FloatVector.broadcast(SF, argX);
        final FloatVector V_Y0   = FloatVector.broadcast(SF, argY);
        final FloatVector V_Z0   = FloatVector.broadcast(SF, argZ);
        final FloatVector V_LANE = FloatVector.fromArray(SF, LANE, 0);

        final FloatVector vXin = V_X0.add(FloatVector.broadcast(SF, (float)x).add(V_LANE).mul(V_FREQ));
        final FloatVector vYin = V_Y0.add(FloatVector.broadcast(SF, (float)y).mul(V_FREQ));
        final FloatVector vZin = V_Z0.add(FloatVector.broadcast(SF, (float)z).mul(V_FREQ));

        // Skew / Unskew
        final FloatVector s   = vXin.add(vYin).add(vZin).mul(V_1_3);
        final FloatVector xiS = vXin.add(s);
        final FloatVector yiS = vYin.add(s);
        final FloatVector ziS = vZin.add(s);

        final IntVector i = floorV(xiS);
        final IntVector j = floorV(yiS);
        final IntVector k = floorV(ziS);

        final FloatVector t  = i.add(j).add(k).convert(VectorOperators.I2F, 0).mul(V_1_6).reinterpretAsFloats();
        final FloatVector x0 = vXin.sub(i.convert(VectorOperators.I2F,0)).add(t);
        final FloatVector y0 = vYin.sub(j.convert(VectorOperators.I2F,0)).add(t);
        final FloatVector z0 = vZin.sub(k.convert(VectorOperators.I2F,0)).add(t);

        // ==== Simplex-Corner-Wahl: exakt wie Skalarcode (6 Fälle) ====
        VectorMask<Float> m_x_ge_y = x0.compare(VectorOperators.GE, y0);
        VectorMask<Float> m_y_ge_z = y0.compare(VectorOperators.GE, z0);
        VectorMask<Float> m_x_ge_z = x0.compare(VectorOperators.GE, z0);
        VectorMask<Float> m_y_lt_z = y0.compare(VectorOperators.LT, z0);
        VectorMask<Float> m_x_lt_z = x0.compare(VectorOperators.LT, z0);

        FloatVector i1f = V0, j1f = V0, k1f = V0;
        FloatVector i2f = V0, j2f = V0, k2f = V0;

        // Case 1: x>=y && y>=z
        VectorMask<Float> c1 = m_x_ge_y.and(m_y_ge_z);
        i1f = i1f.blend(V1, c1); i2f = i2f.blend(V1, c1); j2f = j2f.blend(V1, c1);

        // Case 2: x>=y && x>=z && !(y>=z)  <=> x>=y && y<z
        VectorMask<Float> c2 = m_x_ge_y.and(m_x_ge_z).and(m_y_ge_z.not());
        i1f = i1f.blend(V1, c2); i2f = i2f.blend(V1, c2); k2f = k2f.blend(V1, c2);

        // Case 3: x>=y && !(x>=z)  <=> x>=y && x<z
        VectorMask<Float> c3 = m_x_ge_y.and(m_x_ge_z.not());
        k1f = k1f.blend(V1, c3); i2f = i2f.blend(V1, c3); k2f = k2f.blend(V1, c3);

        // Case 4: !(x>=y) && y<z
        VectorMask<Float> c4 = m_x_ge_y.not().and(m_y_lt_z);
        k1f = k1f.blend(V1, c4); j2f = j2f.blend(V1, c4); k2f = k2f.blend(V1, c4);

        // Case 5: !(x>=y) && !(y<z) && x<z  <=> x<y && y>=z && x<z
        VectorMask<Float> c5 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z);
        j1f = j1f.blend(V1, c5); j2f = j2f.blend(V1, c5); k2f = k2f.blend(V1, c5);

        // Case 6: sonst
        VectorMask<Float> c6 = m_x_ge_y.not().and(m_y_lt_z.not()).and(m_x_lt_z.not());
        j1f = j1f.blend(V1, c6); i2f = i2f.blend(V1, c6); j2f = j2f.blend(V1, c6);

        // Ecken (wie Skalar: -i1+1/6, -i2+2/6, -1+3/6)
        final FloatVector x1 = x0.sub(i1f).add(V_1_6);
        final FloatVector y1 = y0.sub(j1f).add(V_1_6);
        final FloatVector z1 = z0.sub(k1f).add(V_1_6);

        final FloatVector x2 = x0.sub(i2f).add(V_2_6);
        final FloatVector y2 = y0.sub(j2f).add(V_2_6);
        final FloatVector z2 = z0.sub(k2f).add(V_2_6);

        final FloatVector x3 = x0.sub(V1).add(V_3_6);
        final FloatVector y3 = y0.sub(V1).add(V_3_6);
        final FloatVector z3 = z0.sub(V1).add(V_3_6);

        // Hashes 0..255 → %12
        final IntVector ii = i.and(I255), jj = j.and(I255), kk = k.and(I255);

        IntVector gi0 = intNoiseV( ii.add( intNoiseV( jj.add( intNoiseV(kk) ) ) ) );
        IntVector gi1 = intNoiseV( ii.add(i1f.convert(VectorOperators.F2I,0))
                                     .add( intNoiseV( jj.add(j1f.convert(VectorOperators.F2I,0))
                                                        .add( intNoiseV( kk.add(k1f.convert(VectorOperators.F2I,0)) ) ) ) ) );
        IntVector gi2 = intNoiseV( ii.add(i2f.convert(VectorOperators.F2I,0))
                                     .add( intNoiseV( jj.add(j2f.convert(VectorOperators.F2I,0))
                                                        .add( intNoiseV( kk.add(k2f.convert(VectorOperators.F2I,0)) ) ) ) ) );
        IntVector gi3 = intNoiseV( ii.add(1)
                                     .add( intNoiseV( jj.add(1).add( intNoiseV( kk.add(1) ) ) ) ) );

        gi0 = mod12Fast(gi0); gi1 = mod12Fast(gi1); gi2 = mod12Fast(gi2); gi3 = mod12Fast(gi3);

        // t und t^4
        final FloatVector t0 = V_0_6.sub(sq(x0).add(sq(y0)).add(sq(z0))).max(V0);
        final FloatVector t1 = V_0_6.sub(sq(x1).add(sq(y1)).add(sq(z1))).max(V0);
        final FloatVector t2 = V_0_6.sub(sq(x2).add(sq(y2)).add(sq(z2))).max(V0);
        final FloatVector t3 = V_0_6.sub(sq(x3).add(sq(y3)).add(sq(z3))).max(V0);

        final FloatVector tt0 = sq(sq(t0));
        final FloatVector tt1 = sq(sq(t1));
        final FloatVector tt2 = sq(sq(t2));
        final FloatVector tt3 = sq(sq(t3));

        // dot(grad, x*)
        FloatVector n0 = tt0.mul( dotFromHashCorner(gi0, x0, y0, z0) );
        FloatVector n1 = tt1.mul( dotFromHashCorner(gi1, x1, y1, z1) );
        FloatVector n2 = tt2.mul( dotFromHashCorner(gi2, x2, y2, z2) );
        FloatVector n3 = tt3.mul( dotFromHashCorner(gi3, x3, y3, z3) );

        final FloatVector vOut = V32.mul(n0.add(n1).add(n2).add(n3));

        // Store (Tail)
        if (x + L <= W) {
            vOut.intoArray(result, base + x);
        } else {
            int remain = W - x;
            for (int lane = 0; lane < remain; lane++) {
                result[base + x + lane] = vOut.lane(lane);
            }
        }
    }

    // ===== Helpers =====

    private static FloatVector sq(FloatVector v){ return v.mul(v); }

    /** floorf vektorisiert (truncate->fix). */
    private static IntVector floorV(FloatVector x) {
        IntVector t    = x.convert(VectorOperators.F2I, 0).reinterpretAsInts();
        FloatVector tf = t.convert(VectorOperators.I2F, 0).reinterpretAsFloats();
        VectorMask<Float> needDec = x.compare(VectorOperators.LT, tf);
        IntVector adj = I0.blend(I1, needDec.cast(SI));
        return t.sub(adj);
    }

    /** v % 12 via Multiply+Shift (exakt für 0..255). */
    private static IntVector mod12Fast(IntVector v) {
        IntVector q = v.mul(I2731).lanewise(VectorOperators.LSHR, 15);
        return v.sub(q.mul(I12));
    }

    /** Hash 0..255 (wie skalar, aber vektorisiert). */
    private static IntVector intNoiseV(IntVector nIn) {
        IntVector a  = nIn.add(463856334).lanewise(VectorOperators.ASHR, 13);
        IntVector b  = nIn.add(575656768);
        IntVector v  = a.lanewise(VectorOperators.XOR, b);
        IntVector n2 = v.mul(v);
        IntVector t  = n2.mul(60493).add(19990303);
        return v.mul(t).add(1376312589).lanewise(VectorOperators.AND, I255);
    }

    /**
     * dot(grad(h), (x,y,z)) ohne Gather:
     * Gruppen 0..3: (x,y), 4..7: (x,z), 8..11: (y,z) – Vorzeichen aus Bit0/Bit1
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

        VectorMask<Integer> mXY = grp.compare(VectorOperators.EQ, I0);
        VectorMask<Integer> mXZ = grp.compare(VectorOperators.EQ, I1);
        VectorMask<Integer> mYZ = grp.compare(VectorOperators.EQ, IntVector.broadcast(SI, 2));

        FloatVector dot = dXY.blend(dXZ, mXZ.cast(SF));
        dot = dot.blend(dYZ, mYZ.cast(SF));
        return dot;
    }
}
