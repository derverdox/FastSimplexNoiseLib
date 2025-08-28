package de.verdox.noise.aparapi.kernel.gpu;

import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.aparapi.kernel.cpu.CPUScalarSimplexNoiseKernel;

@Deprecated
public abstract class GPUScalarSimplexNoiseKernel extends CPUScalarSimplexNoiseKernel {
    public GPUScalarSimplexNoiseKernel(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
        super(noiseCalculationMode);
    }

    public abstract static class Batched extends CPUScalarSimplexNoiseKernel.Batched {
        public Batched(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
            super(noiseCalculationMode);
        }

        public static class Noise3DIndexing1D extends CPUScalarSimplexNoiseKernel.Batched.Noise3DIndexing1D {
            public Noise3DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }
        }

        public static class Noise3DIndexing3D extends CPUScalarSimplexNoiseKernel.Batched.Noise3DIndexing3D {

            public Noise3DIndexing3D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }
        }
    }

    public abstract static class Simple extends CPUScalarSimplexNoiseKernel.Simple {
        public Simple(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
            super(noiseCalculationMode);
        }

        public static class Noise3DIndexing1D extends CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing1D {
            public Noise3DIndexing1D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }
        }

        public static class Noise3DIndexing3D extends CPUScalarSimplexNoiseKernel.Simple.Noise3DIndexing3D {
            public Noise3DIndexing3D(NoiseBackendBuilder.NoiseCalculationMode noiseCalculationMode) {
                super(noiseCalculationMode);
            }
        }
    }
}
