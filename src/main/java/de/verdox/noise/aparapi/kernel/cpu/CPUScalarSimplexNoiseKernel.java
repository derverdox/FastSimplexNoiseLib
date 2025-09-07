package de.verdox.noise.aparapi.kernel.cpu;

import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoiseKernel;

public abstract class CPUScalarSimplexNoiseKernel extends AbstractSimplexNoiseKernel {
    public CPUScalarSimplexNoiseKernel(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
        super(noiseCalculationMode);
    }

    public void calculate3DNoise(int index, float xin, float yin, float zin) {
        noiseResult[index] = noiseCalcMode == 0
                ? scalarNoiseAluOnly(xin, yin, zin)
                : cpuScalarNoiseLookup(xin, yin, zin);
    }

    public void calculate2DNoise(int index, float xin, float zin) {
        noiseResult[index] = (noiseCalcMode == 0)
                ? scalarNoiseAluOnly2D(xin, zin)
                : cpuScalarNoiseLookup2D(xin, zin);
    }

    // ==================== Batched ====================
    public abstract static class Batched extends CPUScalarSimplexNoiseKernel {
        public int globalWidth, globalHeight;

        public Batched(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
            super(noiseCalculationMode);
        }

        public static class Noise3DIndexing1D extends Batched {
            public Noise3DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int gid = getGlobalId(); // 1D
                int n = gridWidth * gridHeight * gridDepth;
                if (gid >= n) return;

                int x = gid % gridWidth;
                int y = (gid / gridWidth) % gridHeight;
                int z = gid / (gridWidth * gridHeight);

                int idx = baseIndex + x + y * globalWidth + z * globalWidth * globalHeight; // x-major

                float xin = (baseX + x) * frequency;
                float yin = (baseY + y) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate3DNoise(idx, xin, yin, zin);
            }
        }

        public static class Noise3DIndexing3D extends Batched {
            public Noise3DIndexing3D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int x = getGlobalId(0);
                int y = getGlobalId(1);
                int z = getGlobalId(2);

                // robust gegen gepaddete Ranges
                if (x >= gridWidth || y >= gridHeight || z >= gridDepth) return;

                int idx = baseIndex + x + y * globalWidth + z * globalWidth * globalHeight; // x-major

                float xin = (baseX + x) * frequency;
                float yin = (baseY + y) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate3DNoise(idx, xin, yin, zin);
            }
        }

        // -------------------- 2D (x,z) --------------------
        /** 1D-globales Launch-Grid → (x,z) Mapping */
        public static class Noise2DIndexing1D extends Batched {
            public Noise2DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int gid = getGlobalId(); // 1D
                int n = gridWidth * gridDepth;
                if (gid >= n) return;

                int x = gid % gridWidth;
                int z = gid / gridWidth;

                int idx = baseIndex + x + z * globalWidth; // x-major (Zeilen = depth)

                float xin = (baseX + x) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate2DNoise(idx, xin, zin);
            }
        }

        /** 2D-globales Launch-Grid: (x,z) */
        public static class Noise2DIndexing2D extends Batched {
            public Noise2DIndexing2D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int x = getGlobalId(0);
                int z = getGlobalId(1);

                if (x >= gridWidth || z >= gridDepth) return;

                int idx = baseIndex + x + z * globalWidth;

                float xin = (baseX + x) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate2DNoise(idx, xin, zin);
            }
        }
    }

    // ==================== Simple ====================
    public abstract static class Simple extends CPUScalarSimplexNoiseKernel {
        public Simple(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
            super(noiseCalculationMode);
        }

        public static class Noise3DIndexing1D extends Simple {
            public Noise3DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int i = getGlobalId(0);

                int x = i % gridWidth;
                int y = (i / gridWidth) % gridHeight;
                int z = i / (gridWidth * gridHeight);

                int idx = baseIndex + i;

                float xin = (baseX + x) * frequency;
                float yin = (baseY + y) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate3DNoise(idx, xin, yin, zin);
            }
        }

        public static class Noise3DIndexing3D extends Simple {
            public Noise3DIndexing3D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int x = getGlobalId(0);
                int y = getGlobalId(1);
                int z = getGlobalId(2);

                int li = (z * gridHeight + y) * gridWidth + x; // lokal, dicht gepackt
                int idx = baseIndex + li;

                float xin = (baseX + x) * frequency;
                float yin = (baseY + y) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate3DNoise(idx, xin, yin, zin);
            }
        }

        // -------------------- 2D (x,z) --------------------
        /** 1D-globales Launch-Grid → (x,z), lokal dicht gepackt */
        public static class Noise2DIndexing1D extends Simple {
            public Noise2DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int i = getGlobalId(0);
                int n = gridWidth * gridDepth;
                if (i >= n) return;

                int x = i % gridWidth;
                int z = i / gridWidth;

                int li = z * gridWidth + x; // lokal: x-major
                int idx = baseIndex + li;

                float xin = (baseX + x) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate2DNoise(idx, xin, zin);
            }
        }

        /** 2D-globales Launch-Grid: (x,z), lokal dicht gepackt */
        public static class Noise2DIndexing2D extends Simple {
            public Noise2DIndexing2D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int x = getGlobalId(0);
                int z = getGlobalId(1);

                if (x >= gridWidth || z >= gridDepth) return;

                int li = z * gridWidth + x; // lokal: x-major
                int idx = baseIndex + li;

                float xin = (baseX + x) * frequency;
                float zin = (baseZ + z) * frequency;

                calculate2DNoise(idx, xin, zin);
            }
        }
    }
}
