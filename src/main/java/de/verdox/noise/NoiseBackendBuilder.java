package de.verdox.noise;

import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import de.verdox.noise.aparapi.backend.gpu.GPUAparapiNoiseBackend;

public abstract class NoiseBackendBuilder<BUILDER extends NoiseBackendBuilder<BUILDER>> {

    public static CPUNoiseBackendBuilder cpu() {
        return new CPUNoiseBackendBuilder();
    }

    public static GPUNoiseBackendBuilder gpu() {
        return new GPUNoiseBackendBuilder();
    }

    protected int size = 16;
    protected boolean is3D = false;
    protected boolean oneDimensionalIndexing = true;
    protected float[] result = new float[size * size * size];
    protected NoiseCalculationMode noiseCalculationMode = NoiseCalculationMode.ALU_ONLY;

    private NoiseBackendBuilder() {
    }

    public BUILDER withSize2D(int size) {
        if (size <= 0 || (size & (size - 1)) != 0) {
            throw new IllegalArgumentException("Size must be 2^x = size");
        }
        this.size = size;
        this.is3D = false;
        this.result = new float[size * size];
        return (BUILDER) this;
    }

    public BUILDER with1DIndexing(boolean oneDimensionalIndexing) {
        this.oneDimensionalIndexing = oneDimensionalIndexing;
        return (BUILDER) this;
    }

    public BUILDER withNoiseCalculationMode(NoiseCalculationMode noiseCalculationMode) {
        this.noiseCalculationMode = noiseCalculationMode;
        return (BUILDER) this;
    }

    public BUILDER withSize3D(int size) {
        if (size <= 0 || (size & (size - 1)) != 0) {
            throw new IllegalArgumentException("Size must be 2^x = size");
        }
        this.size = size;
        this.is3D = true;
        this.result = new float[size * size * size];
        return (BUILDER) this;
    }

    public NoiseCalculationMode getNoiseCalculationMode() {
        return noiseCalculationMode;
    }

    public abstract NoiseBackend build();

    public static class CPUNoiseBackendBuilder extends NoiseBackendBuilder<CPUNoiseBackendBuilder> {
        private boolean preventRamUsage;
        private boolean vectorize;
        private CPUParallelismMode parallelismMode;

        private CPUNoiseBackendBuilder() {
        }

        /**
         * Uses a kernel that tries to hold all memory in L1 and L2 caches of processor cores
         */
        public CPUNoiseBackendBuilder preventRamUsage(boolean preventRamUsage) {
            this.preventRamUsage = preventRamUsage;
            return this;
        }

        public CPUNoiseBackendBuilder vectorize(boolean vectorize) {
            this.vectorize = vectorize;
            return this;
        }

        public CPUNoiseBackendBuilder withParallelismMode(CPUParallelismMode parallelismMode) {
            this.parallelismMode = parallelismMode;
            return this;
        }

        @Override
        public NoiseBackend build() {
            return switch (parallelismMode) {
                case SEQUENTIAL ->
                        new CPUJavaSeqBackend(null, noiseCalculationMode, vectorize, preventRamUsage, result, size, size, size);
                case PARALLELISM_CORES ->
                        new CPUJavaJtpBackend(null, noiseCalculationMode, false, vectorize, preventRamUsage, result, size, size, size);
                case PARALLELISM_THREADS ->
                        new CPUJavaJtpBackend(null, noiseCalculationMode, true, vectorize, preventRamUsage, result, size, size, size);
            };
        }

        public boolean isPreventRamUsage() {
            return preventRamUsage;
        }

        public boolean isVectorize() {
            return vectorize;
        }

        public CPUParallelismMode getParallelismMode() {
            return parallelismMode;
        }
    }

    public static class GPUNoiseBackendBuilder extends NoiseBackendBuilder<GPUNoiseBackendBuilder> {
        private boolean useBatching;
        private OpenCLDevice preferredDevice = (OpenCLDevice) KernelManager.DeprecatedMethods.bestGPU();

        private GPUNoiseBackendBuilder() {
        }

        public GPUNoiseBackendBuilder withBatching(boolean useBatching) {
            this.useBatching = useBatching;
            return this;
        }

        public GPUNoiseBackendBuilder withPreferredDevice(OpenCLDevice preferredDevice) {
            this.preferredDevice = preferredDevice;
            return this;
        }

        @Override
        public NoiseBackend build() {
            if (useBatching) {
                return new GPUAparapiNoiseBackend.Batched(preferredDevice, this, result, size, size, size);
            } else {
                return new GPUAparapiNoiseBackend.Simple(preferredDevice, this, result, size, size, size);
            }
        }

        public boolean isUseBatching() {
            return useBatching;
        }
    }

    public enum CPUParallelismMode {
        SEQUENTIAL,
        PARALLELISM_CORES,
        PARALLELISM_THREADS,
    }

    public enum NoiseCalculationMode {
        ALU_ONLY,
        LOOKUP
    }
}
