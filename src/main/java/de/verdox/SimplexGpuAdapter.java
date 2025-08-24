package de.verdox;

import com.aparapi.Range;
import micycle.jsimplex.generator.NoiseSurface;
import micycle.jsimplex.noise.gpu.SimplexNoiseGpu2D;
import micycle.jsimplex.noise.gpu.SimplexNoiseGpu3D;
import micycle.jsimplex.noise.gpu.SimplexNoiseGpu3DKernelIntNoised;

/**
 * Dünner Wrapper: erzeugt pro Z-Slice ein 2D-Noise-Field via jSimplex.
 * Ersetze den Aufruf in generate2D(...) durch die passende jSimplex-API aus dem Repo.
 */
public final class SimplexGpuAdapter {


    private static final int MAX_LOCAL = 256;   // Gerätegrenze
    private static final int SLAB_DEPTH = 64;   // Z-Slice-Höhe pro Launch (anpassbar)

    private SimplexGpuAdapter() {
    }

    public static void generate3D(float[] out, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz) {
        generate3dRaw(out, x0, y0, z0, nx, ny, nz, 1, true);
    }

    /**
     * Generates a 3d surface of noise in a 1D array representation. Array is laid
     * out as consecutive rows.
     *
     * @param x         X coordinate of the surface in noise space.
     * @param y         Y coordinate of the surface in noise space.
     * @param z         Z coordinate of the surface in noise space.
     * @param width     Width of the surface.
     * @param height    Height of the surface.
     * @param depth     Depth of the surface.
     * @param frequency Frequency of the noise wave.
     * @param fast      Whether memory bandwidth optimizations are used.
     * @return 3D surface of noise in a 1D array representation.
     */

    private static void generate3dRaw(float[] out,
                                      float x, float y, float z,
                                      int width, int height, int depth,
                                      float frequency,
                                      boolean fast) {

        // Kernel wiederverwenden (unverändert)
        SimplexNoiseGpu3DKernelIntNoised kernel = new SimplexNoiseGpu3DKernelIntNoised();

        final int planeSize = width * height;
        int base = 0; // Schreiboffset im 'out' (Z-major)

        for (int z0 = 0; z0 < depth; z0 += SLAB_DEPTH) {
            final int dz = Math.min(SLAB_DEPTH, depth - z0);
            final int sliceElems = planeSize * dz;

            // temporäres Slice-Array: Kernel schreibt immer ab Index 0
            float[] slice = new float[sliceElems];

            // Z-Start wird über Koordinate realisiert (z + z0*frequency)
            kernel.setParameters(slice, x, y, z + z0 * frequency, width, height, dz, frequency);

            // lokale Größe so wählen, dass sie sliceElems teilt (≤ 256)
            int local = pickLocalSize(sliceElems);

            // Range: exakt sliceElems global (kein Aufrunden nötig)
            Range range = Range.create(sliceElems, local);

            // optional: "fast" könnte hier eine andere Kernelvariante wählen;
            // da der Kernel unverändert ist, behandeln wir 'fast' identisch.
            kernel.execute(range);

            // in das Ziel-Array kopieren (Z-major)
            System.arraycopy(slice, 0, out, base, sliceElems);
            base += sliceElems;
        }

        kernel.dispose();
    }

    /** Wählt die größte lokale Größe ≤ MAX_LOCAL, die 'n' teilt. Fällt notfalls bis 1 zurück. */
    private static int pickLocalSize(int n) {
        // bevorzugte Kandidaten absteigend (typisch gute Größen)
        int[] candidates = {256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1};
        for (int c : candidates) {
            if (c <= MAX_LOCAL && (n % c) == 0) return c;
        }
        // falls keiner passt (sollte praktisch nie passieren), nutze 1
        return 1;
    }
}
