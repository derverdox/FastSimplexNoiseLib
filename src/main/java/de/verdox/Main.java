package de.verdox;

import de.verdox.noise.NoiseBackend;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.NoiseEngine3D;
import de.verdox.util.HardwareUtil;

import java.io.IOException;

public class Main {

    private static final int size = 512;

    public static void main(String[] args) throws IOException {
        NoiseBackend backend = NoiseBackendBuilder.cpu()
                .withSize3D(512)
                .withParallelismMode(NoiseBackendBuilder.CPUParallelismMode.PARALLELISM_CORES)
                .build();

        HardwareUtil.printCPU();

        NoiseEngine3D engine = new NoiseEngine3D(backend);
        backend.logSetup();
        long start = System.nanoTime();
        engine.computeNoise(0, 0, 0, 0.009f);
        long end = System.nanoTime();


        javax.imageio.ImageIO.write(backend.topLayerToGrayscale(), "png", new java.io.File("heightmap.png"));
    }
}
