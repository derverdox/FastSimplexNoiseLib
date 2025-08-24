package de.verdox.noise.aparapi.kernel.vectorized;

import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import jdk.incubator.vector.*;

public abstract class AbstractVectorizedSimplexNoise3DAparapiKernel extends AbstractSimplexNoise3DAparapiKernel {
    private static final VectorSpecies<Float> SF = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SI = IntVector.SPECIES_PREFERRED;

    private static final FloatVector V_ONE_THIRD = FloatVector.broadcast(SF, 1f / 3f);
    private static final FloatVector V_ONE_SIXTH = FloatVector.broadcast(SF, 1f / 6f);
    private static final FloatVector V_TWO_SIXTH = FloatVector.broadcast(SF, 2f / 6f);
    private static final FloatVector V_THREE_SIXTH = FloatVector.broadcast(SF, 3f / 6f);
    private static final FloatVector V_POINT6 = FloatVector.broadcast(SF, 0.6f);
    private static final FloatVector V_32 = FloatVector.broadcast(SF, 32f);

    private static final FloatVector V_ZERO = FloatVector.zero(SF);
    private static final FloatVector V_ONE = FloatVector.broadcast(SF, 1f);
    private static final IntVector I_255 = IntVector.broadcast(SI, 255);
    private static final IntVector I_12 = IntVector.broadcast(SI, 12);

    private static final float[] LANE_IDX;
    static {
        LANE_IDX = new float[SF.length()];
        for (int i = 0; i < LANE_IDX.length; i++) LANE_IDX[i] = i;
    }

    /** Achtung: ignoriert getGlobalId – rechnet die gesamte Slab argDepth × argHeight × argWidth. */
    @Override public void run() {
        final int W = argWidth, H = argHeight, D = argDepth;
        final int L = SF.length();
        final FloatVector V_FREQ = FloatVector.broadcast(SF, argFrequency);
        final FloatVector V_X0   = FloatVector.broadcast(SF, argX);
        final FloatVector V_Y0   = FloatVector.broadcast(SF, argY);
        final FloatVector V_Z0   = FloatVector.broadcast(SF, argZ);
        final FloatVector V_LANE = FloatVector.fromArray(SF, LANE_IDX, 0);

        // Gradienten getrennt (12) für schnellen Zugriff
        final float[] GX = new float[12], GY = new float[12], GZ = new float[12];
        for (int g = 0; g < 12; g++) {
            GX[g] = grad3[3*g  ];
            GY[g] = grad3[3*g+1];
            GZ[g] = grad3[3*g+2];
        }

        for (int z = 0; z < D; z++) {
            FloatVector vZin = V_Z0.add(FloatVector.broadcast(SF, z).mul(V_FREQ));
            for (int y = 0; y < H; y++) {
                FloatVector vYin = V_Y0.add(FloatVector.broadcast(SF, y).mul(V_FREQ));
                int base = argBase + (z * H + y) * W;

                int x = 0;
                for (; x + L <= W; x += L) {
                    // xin lanes = argX + (x + lane) * freq
                    FloatVector vXin = V_X0.add(FloatVector.broadcast(SF, (float)x).add(V_LANE).mul(V_FREQ));

                    // ---- Skewing ----
                    FloatVector vS = vXin.add(vYin).add(vZin).mul(V_ONE_THIRD);
                    FloatVector vXi_s = vXin.add(vS);
                    FloatVector vYi_s = vYin.add(vS);
                    FloatVector vZi_s = vZin.add(vS);

                    // fastfloor vektorisiert
                    IntVector vI = fastfloorV(vXi_s);
                    IntVector vJ = fastfloorV(vYi_s);
                    IntVector vK = fastfloorV(vZi_s);

                    // Unskew
                    FloatVector vT = vI.add(vJ).add(vK).convert(VectorOperators.I2F,0).mul(V_ONE_SIXTH)
                                       .reinterpretAsFloats();
                    FloatVector vX0 = vXin.sub( vI.convert(VectorOperators.I2F,0).sub(vT) );
                    FloatVector vY0 = vYin.sub( vJ.convert(VectorOperators.I2F,0).sub(vT) );
                    FloatVector vZ0 = vZin.sub( vK.convert(VectorOperators.I2F,0).sub(vT) );

                    // Simplex Rangfolge -> i1/j1/k1 und i2/j2/k2
                    VectorMask<Float> m_x_ge_y = vX0.compare(VectorOperators.GE, vY0);
                    VectorMask<Float> m_y_ge_z = vY0.compare(VectorOperators.GE, vZ0);
                    VectorMask<Float> m_x_ge_z = vX0.compare(VectorOperators.GE, vZ0);
                    VectorMask<Float> m_y_lt_z = vY0.compare(VectorOperators.LT, vZ0);
                    VectorMask<Float> m_x_lt_z = vX0.compare(VectorOperators.LT, vZ0);

                    FloatVector i1 = V_ZERO, j1 = V_ZERO, k1 = V_ZERO;
                    FloatVector i2 = V_ZERO, j2 = V_ZERO, k2 = V_ZERO;

                    // A: x>=y>=z
                    VectorMask<Float> ca = m_x_ge_y.and(m_y_ge_z);
                    i1 = i1.blend(V_ONE, ca); i2 = i2.blend(V_ONE, ca); j2 = j2.blend(V_ONE, ca);
                    // B: x>=z>y
                    VectorMask<Float> cb = m_x_ge_y.and(m_y_lt_z).and(m_x_ge_z);
                    i1 = i1.blend(V_ONE, cb); i2 = i2.blend(V_ONE, cb); k2 = k2.blend(V_ONE, cb);
                    // C: z>x>=y
                    VectorMask<Float> cc = m_x_lt_z.and(m_x_ge_y);
                    k1 = k1.blend(V_ONE, cc); i2 = i2.blend(V_ONE, cc); k2 = k2.blend(V_ONE, cc);
                    // D: y<z & !(x>=y)
                    VectorMask<Float> cd = m_y_lt_z.and(m_x_ge_y.not());
                    k1 = k1.blend(V_ONE, cd); j2 = j2.blend(V_ONE, cd); k2 = k2.blend(V_ONE, cd);
                    // E: x<z & !(y>=z)
                    VectorMask<Float> ce = m_x_lt_z.and(m_y_ge_z.not());
                    j1 = j1.blend(V_ONE, ce); j2 = j2.blend(V_ONE, ce); k2 = k2.blend(V_ONE, ce);
                    // F: else
                    VectorMask<Float> cf = m_x_ge_z.and(m_y_ge_z).not().and(m_x_ge_y.not());
                    j1 = j1.blend(V_ONE, cf); i2 = i2.blend(V_ONE, cf); j2 = j2.blend(V_ONE, cf);

                    // Posis
                    FloatVector vX1 = vX0.sub(i1.mul(V_ONE_SIXTH));
                    FloatVector vY1 = vY0.sub(j1.mul(V_ONE_SIXTH));
                    FloatVector vZ1 = vZ0.sub(k1.mul(V_ONE_SIXTH));
                    FloatVector vX2 = vX0.sub(i2.mul(V_TWO_SIXTH));
                    FloatVector vY2 = vY0.sub(j2.mul(V_TWO_SIXTH));
                    FloatVector vZ2 = vZ0.sub(k2.mul(V_TWO_SIXTH));
                    FloatVector vX3 = vX0.sub(V_THREE_SIXTH);
                    FloatVector vY3 = vY0.sub(V_THREE_SIXTH);
                    FloatVector vZ3 = vZ0.sub(V_THREE_SIXTH);

                    // Hashes (intNoise) → 0..255 → %12
                    IntVector vII = vI.and(I_255), vJJ = vJ.and(I_255), vKK = vK.and(I_255);

                    IntVector gi0 = intNoiseV( vII.add( intNoiseV( vJJ.add( intNoiseV(vKK) ) ) ) );
                    IntVector gi1 = intNoiseV( vII.add(i1.convert(VectorOperators.F2I,0))
                                                  .add( intNoiseV( vJJ.add(j1.convert(VectorOperators.F2I,0))
                                                                      .add( intNoiseV( vKK.add(k1.convert(VectorOperators.F2I,0)) ) ) ) ) );
                    IntVector gi2 = intNoiseV( vII.add(i2.convert(VectorOperators.F2I,0))
                                                  .add( intNoiseV( vJJ.add(j2.convert(VectorOperators.F2I,0))
                                                                      .add( intNoiseV( vKK.add(k2.convert(VectorOperators.F2I,0)) ) ) ) ) );
                    IntVector gi3 = intNoiseV( vII.add(1)
                                                  .add( intNoiseV( vJJ.add(1).add( intNoiseV(vKK.add(1)) ) ) ) );
                    gi0 = mod12(gi0); gi1 = mod12(gi1); gi2 = mod12(gi2); gi3 = mod12(gi3);

                    // t0..t3, t^4
                    FloatVector t0 = V_POINT6.sub( sq(vX0).add(sq(vY0)).add(sq(vZ0)) ).max(V_ZERO);
                    FloatVector t1 = V_POINT6.sub( sq(vX1).add(sq(vY1)).add(sq(vZ1)) ).max(V_ZERO);
                    FloatVector t2 = V_POINT6.sub( sq(vX2).add(sq(vY2)).add(sq(vZ2)) ).max(V_ZERO);
                    FloatVector t3 = V_POINT6.sub( sq(vX3).add(sq(vY3)).add(sq(vZ3)) ).max(V_ZERO);

                    FloatVector tt0 = sq(sq(t0));
                    FloatVector tt1 = sq(sq(t1));
                    FloatVector tt2 = sq(sq(t2));
                    FloatVector tt3 = sq(sq(t3));

                    // dot(grad, x*)
                    FloatVector g0x = selectGrad(GX, gi0), g0y = selectGrad(GY, gi0), g0z = selectGrad(GZ, gi0);
                    FloatVector g1x = selectGrad(GX, gi1), g1y = selectGrad(GY, gi1), g1z = selectGrad(GZ, gi1);
                    FloatVector g2x = selectGrad(GX, gi2), g2y = selectGrad(GY, gi2), g2z = selectGrad(GZ, gi2);
                    FloatVector g3x = selectGrad(GX, gi3), g3y = selectGrad(GY, gi3), g3z = selectGrad(GZ, gi3);

                    FloatVector n0 = tt0.mul( g0x.mul(vX0).add(g0y.mul(vY0)).add(g0z.mul(vZ0)) );
                    FloatVector n1 = tt1.mul( g1x.mul(vX1).add(g1y.mul(vY1)).add(g1z.mul(vZ1)) );
                    FloatVector n2 = tt2.mul( g2x.mul(vX2).add(g2y.mul(vY2)).add(g2z.mul(vZ2)) );
                    FloatVector n3 = tt3.mul( g3x.mul(vX3).add(g3y.mul(vY3)).add(g3z.mul(vZ3)) );

                    FloatVector vOut = V_32.mul( n0.add(n1).add(n2).add(n3) );
                    vOut.intoArray(result, base + x);
                }

                // Tail skalar
                for (; x < W; x++) {
                    int idx = base + x;
                    float xin = argX + x * argFrequency;
                    float yin = argY + y * argFrequency;
                    float zin = argZ + z * argFrequency;
                    result[idx] = scalarNoise(xin, yin, zin);
                }
            }
        }
    }

    // === Vektor-Helpers ===
    private static FloatVector sq(FloatVector v){ return v.mul(v); }

    private static IntVector fastfloorV(FloatVector x) {
        IntVector trunc = x.convert(VectorOperators.F2I, 0).reinterpretAsInts();             // truncate toward 0
        FloatVector truncF = trunc.convert(VectorOperators.I2F, 0).reinterpretAsFloats();
        var negFrac = x.compare(VectorOperators.LT, truncF);
        IntVector adj = IntVector.zero(SI).blend(IntVector.broadcast(SI, 1), negFrac.cast(SI));
        return trunc.sub(adj); // floor
    }

    private static IntVector intNoiseV(IntVector nIn) {
        IntVector a = nIn.add(463856334).lanewise(VectorOperators.ASHR, 13);
        IntVector b = nIn.add(575656768);
        IntVector v = a.lanewise(VectorOperators.XOR, b);
        IntVector n2 = v.mul(v);
        IntVector t  = n2.mul(60493).add(19990303);
        return v.mul(t).add(1376312589)
                .lanewise(VectorOperators.AND, IntVector.broadcast(SI, 0x7fffffff))
                .lanewise(VectorOperators.AND, IntVector.broadcast(SI, 255));
    }

    private static IntVector mod12(IntVector v) {
        IntVector q = v.lanewise(VectorOperators.DIV, I_12);
        return v.sub(q.mul(12));
    }

    private static FloatVector selectGrad(float[] table, IntVector idx) {
        FloatVector out = V_ZERO;
        for (int g = 0; g < 12; g++) {
            var m = idx.compare(VectorOperators.EQ, IntVector.broadcast(SI, g));
            out = out.blend(FloatVector.broadcast(SF, table[g]), m.cast(SF));
        }
        return out;
    }
}
