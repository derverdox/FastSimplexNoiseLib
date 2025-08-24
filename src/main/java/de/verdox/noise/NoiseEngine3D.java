package de.verdox.noise;

import jdk.incubator.vector.FloatVector;

public class NoiseEngine3D {
    public static int CPU_CORES = Runtime.getRuntime().availableProcessors();
    public static int VECTOR_REGISTER_BIT_SIZE = FloatVector.SPECIES_PREFERRED.vectorBitSize();
    public static int VECTOR_AMOUNT_LANES = FloatVector.SPECIES_PREFERRED.length();
    private final NoiseBackend noiseBackend;

    /**
     * Creates an Engine with a pre-defined width, height, and depth per job.
     */
    public NoiseEngine3D(NoiseBackend noiseBackend) {
        this.noiseBackend = noiseBackend;
    }

    public void computeNoise(float startX, float startY, float startZ, float frequency) {
        this.noiseBackend.generate(startX, startY, startZ, frequency);
    }
}
