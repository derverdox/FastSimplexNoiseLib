package de.verdox;

import de.verdox.noise.NoiseBackend;
import de.verdox.noise.NoiseBackendFactory;
import de.verdox.noise.NoiseEngine3D;
import de.verdox.util.HardwareUtil;

import java.io.IOException;

public class Main {

    private static final int size = 512;

    public static void main(String[] args) throws IOException {
        float[] result = new float[size * size * size];
        boolean optimizeCache = false;

        NoiseBackend gpu = NoiseBackendFactory.firstGPU(true, result, size, size, size);
        NoiseBackend cpuParallelScalar = NoiseBackendFactory.cpuScalarParallel(optimizeCache, result, size, size, size);
        NoiseBackend cpuSeqScalar = NoiseBackendFactory.cpuScalarSeq(optimizeCache, result, size, size, size);

        NoiseBackend cpuParallelVectorized = NoiseBackendFactory.cpuVectorizedParallel(optimizeCache, result, size, size, size);
        NoiseBackend cpuSeqScalarVectorized = NoiseBackendFactory.cpuVectorizedSeq(optimizeCache, result, size, size, size);

        HardwareUtil.printCPU();


        NoiseBackend backend = gpu;

        NoiseEngine3D engine = new NoiseEngine3D(backend);
        backend.logSetup();
        long start = System.nanoTime();
        engine.computeNoise(0, 0, 0, 0.009f);
        long end = System.nanoTime();


        javax.imageio.ImageIO.write(backend.topLayerToGrayscale(), "png", new java.io.File("heightmap.png"));
    }
}
