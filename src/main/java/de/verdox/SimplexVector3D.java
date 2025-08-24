package de.verdox;

import jdk.incubator.vector.*;

public final class SimplexVector3D {
    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final float F3 = 1f/3f, G3 = 1f/6f;

    // Gradienten + Perm wie im Scalar-Backend
    private static final int[][] GRAD = SimplexScalar3DGradient.GRAD;
    private static final int[] PERM = SimplexScalar3DGradient.PERM;
    private static int p(int x){ return PERM[x & 255]; }

    // Hilfsklasse, nur um Duplikat zu vermeiden:
    static final class SimplexScalar3DGradient {
        static final int[][] GRAD = {
                {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
                {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
                {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
        };
        static final int[] PERM = new int[512];
        static {
            int[] P = new int[256];
            for (int i=0;i<256;i++) P[i]=i;
            for (int i=0;i<256;i++) PERM[i]=PERM[i+256]=P[i];
        }
    }

    private static FloatVector gatherGrad(int[] gi, int comp, int l) {
        float[] tmp = new float[F.length()];
        for (int i = 0; i < l; i++) {
            tmp[i] = GRAD[gi[i]][comp];
        }
        // Rest-Lanes bleiben 0; sie werden ohnehin maskiert
        return FloatVector.fromArray(F, tmp, 0);
    }
    private static float[] pad(float[] src, int used) {
        if (used == F.length()) return src;
        float[] dst = new float[F.length()];
        System.arraycopy(src, 0, dst, 0, used);
        return dst;
    }

    /** vektorisiert eine X-Zeile (nx Werte) bei festem Y, Z in {@code out[off..off+nx)}. */
    public static void noiseLine(float[] xRamp, float Y, float Z, float[] out, int off, int nx) {
        int x = 0;
        while (x < nx) {
            int lanes = F.length();
            int l = Math.min(lanes, nx - x);
            var m = F.indexInRange(0, l);

            // Vektoren laden (mit Offset + Maske)
            var vx = FloatVector.fromArray(F, xRamp, x, m);
            var vy = FloatVector.broadcast(F, Y);
            var vz = FloatVector.broadcast(F, Z);

            var s = vx.add(vy).add(vz).mul(F3);

            // floor lane-weise (skalar) – nur l Elemente
            float[] tmp = new float[l];
            vx.add(s).intoArray(tmp, 0, m);
            int[] ii = new int[l];
            for (int i = 0; i < l; i++) ii[i] = (int) Math.floor(tmp[i]);

            vy.add(s).intoArray(tmp, 0, m);
            int[] jj = new int[l];
            for (int i = 0; i < l; i++) jj[i] = (int) Math.floor(tmp[i]);

            vz.add(s).intoArray(tmp, 0, m);
            int[] kk = new int[l];
            for (int i = 0; i < l; i++) kk[i] = (int) Math.floor(tmp[i]);

            // Unskew + lokale Deltas
            float[] X0 = new float[l], Y0 = new float[l], Z0 = new float[l];
            float[] x0a = new float[l], y0a = new float[l], z0a = new float[l];
            float[] xs = new float[l];
            // xs aus xRamp kopieren (nur l Werte)
            System.arraycopy(xRamp, x, xs, 0, l);

            for (int i = 0; i < l; i++) {
                float t = (ii[i] + jj[i] + kk[i]) * G3;
                X0[i] = ii[i] - t; Y0[i] = jj[i] - t; Z0[i] = kk[i] - t;
                x0a[i] = xs[i] - X0[i];
                y0a[i] = Y     - Y0[i];
                z0a[i] = Z     - Z0[i];
            }

            // Corner ranking
            int[] i1 = new int[l], j1 = new int[l], k1 = new int[l];
            int[] i2 = new int[l], j2 = new int[l], k2 = new int[l];
            for (int i = 0; i < l; i++) {
                float x0v = x0a[i], y0v = y0a[i], z0v = z0a[i];
                if (x0v >= y0v) {
                    if (y0v >= z0v) { i1[i]=1; j1[i]=0; k1[i]=0; i2[i]=1; j2[i]=1; k2[i]=0; }
                    else if (x0v >= z0v) { i1[i]=1; j1[i]=0; k1[i]=0; i2[i]=1; j2[i]=0; k2[i]=1; }
                    else { i1[i]=0; j1[i]=0; k1[i]=1; i2[i]=1; j2[i]=0; k2[i]=1; }
                } else {
                    if (y0v < z0v) { i1[i]=0; j1[i]=0; k1[i]=1; i2[i]=0; j2[i]=1; k2[i]=1; }
                    else if (x0v < z0v) { i1[i]=0; j1[i]=1; k1[i]=0; i2[i]=0; j2[i]=1; k2[i]=1; }
                    else { i1[i]=0; j1[i]=1; k1[i]=0; i2[i]=1; j2[i]=1; k2[i]=0; }
                }
            }

            float[] x1 = new float[l], y1 = new float[l], z1 = new float[l];
            float[] x2 = new float[l], y2 = new float[l], z2 = new float[l];
            float[] x3 = new float[l], y3 = new float[l], z3 = new float[l];
            for (int i = 0; i < l; i++) {
                x1[i]=x0a[i]-i1[i]+G3; y1[i]=y0a[i]-j1[i]+G3; z1[i]=z0a[i]-k1[i]+G3;
                x2[i]=x0a[i]-i2[i]+2*G3; y2[i]=y0a[i]-j2[i]+2*G3; z2[i]=z0a[i]-k2[i]+2*G3;
                x3[i]=x0a[i]-1+3*G3;     y3[i]=y0a[i]-1+3*G3;     z3[i]=z0a[i]-1+3*G3;
            }

            int[] gi0 = new int[l], gi1a = new int[l], gi2a = new int[l], gi3a = new int[l];
            for (int i = 0; i < l; i++) {
                int I = ii[i] & 255, J = jj[i] & 255, K = kk[i] & 255;
                gi0[i]  = p(I +     p(J +     p(K    ))) % 12;
                gi1a[i] = p(I + i1[i]+p(J + j1[i]+p(K+k1[i]))) % 12;
                gi2a[i] = p(I + i2[i]+p(J + j2[i]+p(K+k2[i]))) % 12;
                gi3a[i] = p(I + 1   +p(J + 1   +p(K + 1   ))) % 12;
            }

            // Vektorisiert: Ecke 0..3, mit Mask m und l-basierten Arrays
            var vx0 = FloatVector.fromArray(F, x0a, 0, m);
            var vy0 = FloatVector.fromArray(F, y0a, 0, m);
            var vz0 = FloatVector.fromArray(F, z0a, 0, m);
            var t0  = FloatVector.broadcast(F, 0.6f)
                                 .sub(vx0.mul(vx0)).sub(vy0.mul(vy0)).sub(vz0.mul(vz0));
            var m0  = t0.compare(VectorOperators.GT, 0f);
            var tt0 = t0.mul(t0).mul(t0).mul(t0);
            var n0  = tt0.mul(
                    gatherGrad(gi0,0,l).mul(vx0)
                                       .add(gatherGrad(gi0,1,l).mul(vy0))
                                       .add(gatherGrad(gi0,2,l).mul(vz0))
            ).blend(FloatVector.zero(F), m0.not());

            var vx1v = FloatVector.fromArray(F, x1, 0, m);
            var vy1v = FloatVector.fromArray(F, y1, 0, m);
            var vz1v = FloatVector.fromArray(F, z1, 0, m);
            var t1v  = FloatVector.broadcast(F, 0.6f)
                                  .sub(vx1v.mul(vx1v)).sub(vy1v.mul(vy1v)).sub(vz1v.mul(vz1v));
            var m1v  = t1v.compare(VectorOperators.GT, 0f);
            var tt1v = t1v.mul(t1v).mul(t1v).mul(t1v);
            var n1v  = tt1v.mul(
                    gatherGrad(gi1a,0,l).mul(vx1v)
                                        .add(gatherGrad(gi1a,1,l).mul(vy1v))
                                        .add(gatherGrad(gi1a,2,l).mul(vz1v))
            ).blend(FloatVector.zero(F), m1v.not());

            var vx2v = FloatVector.fromArray(F, x2, 0, m);
            var vy2v = FloatVector.fromArray(F, y2, 0, m);
            var vz2v = FloatVector.fromArray(F, z2, 0, m);
            var t2v  = FloatVector.broadcast(F, 0.6f)
                                  .sub(vx2v.mul(vx2v)).sub(vy2v.mul(vy2v)).sub(vz2v.mul(vz2v));
            var m2v  = t2v.compare(VectorOperators.GT, 0f);
            var tt2v = t2v.mul(t2v).mul(t2v).mul(t2v);
            var n2v  = tt2v.mul(
                    gatherGrad(gi2a,0,l).mul(vx2v)
                                        .add(gatherGrad(gi2a,1,l).mul(vy2v))
                                        .add(gatherGrad(gi2a,2,l).mul(vz2v))
            ).blend(FloatVector.zero(F), m2v.not());

            var vx3v = FloatVector.fromArray(F, x3, 0, m);
            var vy3v = FloatVector.fromArray(F, y3, 0, m);
            var vz3v = FloatVector.fromArray(F, z3, 0, m);
            var t3v  = FloatVector.broadcast(F, 0.6f)
                                  .sub(vx3v.mul(vx3v)).sub(vy3v.mul(vy3v)).sub(vz3v.mul(vz3v));
            var m3v  = t3v.compare(VectorOperators.GT, 0f);
            var tt3v = t3v.mul(t3v).mul(t3v).mul(t3v);
            var n3v  = tt3v.mul(
                    gatherGrad(gi3a,0,l).mul(vx3v)
                                        .add(gatherGrad(gi3a,1,l).mul(vy3v))
                                        .add(gatherGrad(gi3a,2,l).mul(vz3v))
            ).blend(FloatVector.zero(F), m3v.not());

            var sum = n0.add(n1v).add(n2v).add(n3v).mul(32f);
            sum.intoArray(out, off + x, m);

            x += l;
        }
    }

}
