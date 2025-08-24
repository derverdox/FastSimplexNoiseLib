package de.verdox;

import micycle.jsimplex.noise.cpu.SimplexNoiseCpu;

import java.util.stream.IntStream;

public final class NoiseEngine {

    private NoiseEngine() {
    }

    /**
     * Samplet Simplex-Noise in einem 3D-Block und liefert ein 1d-Array in Z-Major Reihenfolge.
     *
     * @param backend  SCALAR | VECTOR | GPU
     * @param parallel true = parallelisiere über Z-Slices
     * @param x0,y0,z0 Startkoordinate
     * @param nx,ny,nz Längen in X,Y,Z (Anzahl Punkte)
     * @param dx,dy,dz Schrittweiten in X,Y,Z (z.B. 1.0f)
     * @return float[nx*ny*nz] in Z-Major: idx = (z*ny + y)*nx + x
     */
    public static float[] generate(
            Backend backend, boolean parallel,
            float x0, float y0, float z0,
            int nx, int ny, int nz,
            float dx, float dy, float dz
    ) {
        float[] out = new float[nx * ny * nz];

        switch (backend) {
            case SCALAR -> runScalar(out, x0, y0, z0, nx, ny, nz, dx, dy, dz, parallel);
            case VECTOR -> runVector(out, x0, y0, z0, nx, ny, nz, dx, dy, dz, parallel);
            case GPU -> runGpu(out, x0, y0, z0, nx, ny, nz, dx, dy, dz);
            default -> throw new IllegalArgumentException("Unsupported backend: " + backend);
        }
        return out;
    }

    private static void runScalar(float[] out, float x0, float y0, float z0,
                                  int nx, int ny, int nz, float dx, float dy, float dz, boolean parallel) {
        var range = IntStream.range(0, nz);
        if (parallel) range = range.parallel();

        range.forEach(z -> {
            float Z = z0 + z * dz;
            int baseZ = z * nx * ny;
            for (int y = 0; y < ny; y++) {
                float Y = y0 + y * dy;
                int baseY = baseZ + y * nx;
                for (int x = 0; x < nx; x++) {
                    float X = x0 + x * dx;
                    out[baseY + x] = (float) SimplexNoiseCpu.noise(X, Y, Z);
                }
            }
        });
    }

    private static void runVector(float[] out, float x0, float y0, float z0,
                                  int nx, int ny, int nz, float dx, float dy, float dz, boolean parallel) {
        var range = IntStream.range(0, nz);
        if (parallel) range = range.parallel();

        range.forEach(z -> {
            float Z = z0 + z * dz;
            int baseZ = z * nx * ny;
            for (int y = 0; y < ny; y++) {
                float Y = y0 + y * dy;
                int off = baseZ + y * nx;
                // vektorisiert eine ganze X-Zeile schreiben
                SimplexVector3D.noiseLine(XRamp(x0, dx, nx), Y, Z, out, off, nx);
            }
        });
    }

    private static void runGpu(float[] out, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz) {
        SimplexGpuAdapter.generate3D(out, x0, y0, z0, nx, ny, nz, dx, dy, dz);
    }

    private static float[] XRamp(float x0, float dx, int nx) {
        float[] xs = new float[nx];
        for (int i = 0; i < nx; i++) xs[i] = x0 + i * dx;
        return xs;
    }
}
