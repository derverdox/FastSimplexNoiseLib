package de.verdox.noise.aparapi.kernel.cpu;

import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoiseKernel;

public abstract class CPUScalarSimplexNoiseKernel extends AbstractSimplexNoiseKernel {
    public CPUScalarSimplexNoiseKernel(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
        super(noiseCalculationMode);
    }

    public void calculate3DNoise(int index, float xin, float yin, float zin) {
        noiseResult[index] = noiseCalcMode == 0 ? scalarNoiseAluOnly(
                xin,
                yin,
                zin
        ) : cpuScalarNoiseLookup(
                xin,
                yin,
                zin
        );
    }

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
                float xin = baseX + x * frequency, yin = baseY + y * frequency, zin = baseZ + z * frequency;
                calculate3DNoise(idx, xin, yin, zin);
            }
        }

        public static class Noise3DIndexing3D extends Batched {

            public Noise3DIndexing3D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }

            @Override
            public void run() {
                int gid = getGlobalId(); // 1D
                int n = gridWidth * gridHeight * gridDepth;
                if (gid >= n) return; // Guard, falls global gepaddet

                int x = getGlobalId(0), y = getGlobalId(1), z = getGlobalId(2);

                int idx = baseIndex + x + y * globalWidth + z * globalWidth * globalHeight; // x-major
                float xin = baseX + x * frequency, yin = baseY + y * frequency, zin = baseZ + z * frequency;
                calculate3DNoise(idx, xin, yin, zin);
            }
        }
    }

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

                calculate3DNoise(baseIndex + i, baseX + x * frequency, baseY + y * frequency, baseZ + z * frequency);
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
                int li = (z * gridHeight + y) * gridWidth + x;
                int idx = baseIndex + li;

                calculate3DNoise(idx, baseX + x * frequency, baseY + y * frequency, baseZ + z * frequency);
            }
        }
    }
}
