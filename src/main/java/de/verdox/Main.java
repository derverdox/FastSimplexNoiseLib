package de.verdox;

import de.verdox.noise.NoiseBackend;
import de.verdox.noise.NoiseBackendFactory;
import de.verdox.noise.NoiseEngine3D;

import java.io.IOException;

public class Main {

    private static final int size = 1024;

    public static void main(String[] args) throws IOException {
        float[] result = new float[size * size * size];

        NoiseBackend gpu = NoiseBackendFactory.firstGPU(result, size, size, size);
        NoiseBackend cpuParallelScalar = NoiseBackendFactory.cpuScalarParallel(result, size, size, size);
        NoiseBackend cpuSeqScalar = NoiseBackendFactory.cpuScalarSeq(result, size, size, size);

        NoiseBackend cpuParallelVectorized = NoiseBackendFactory.cpuVectorizedParallel(result, size, size, size);
        NoiseBackend cpuSeqScalarVectorized = NoiseBackendFactory.cpuVectorizedSeq(result, size, size, size);


        NoiseBackend backend = gpu;

        NoiseEngine3D engine = new NoiseEngine3D(backend);
        backend.logSetup();
        long start = System.nanoTime();
        engine.computeNoise(0, 0, 0, 0.009f);
        long end = System.nanoTime();


        javax.imageio.ImageIO.write(backend.topLayerToGrayscale(), "png", new java.io.File("heightmap.png"));
    }
}
