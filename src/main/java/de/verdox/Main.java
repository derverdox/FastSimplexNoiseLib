package de.verdox;

import de.verdox.noise.NoiseBackend;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.NoiseEngine3D;
import de.verdox.util.HardwareUtil;
import de.verdox.util.LODUtil;

import java.io.IOException;
import java.util.Arrays;

public class Main {

    private static final int size = 16;

    public static void main(String[] args) throws IOException {
        NoiseBackend backend = NoiseBackendBuilder.gpu()
                .withSeed(1239179847938347234L)
                .withSize2D(size, (byte) 0, LODUtil.LODMode.TILE_PYRAMID)
                //.vectorize(false)
                //.withParallelismMode(NoiseBackendBuilder.CPUParallelismMode.PARALLELISM_CORES)
                .build();

        HardwareUtil.printCPU();

        NoiseEngine3D engine = new NoiseEngine3D(backend);
        backend.logSetup();


        if(backend.is3D) {
            engine.computeNoise(0, 0, 0, 0.009f);
            javax.imageio.ImageIO.write(backend.topLayer3DToGrayscale(), "png", new java.io.File("heightmap.png"));
        }
        else {
            engine.computeNoise(0, 0, 0.009f);
            javax.imageio.ImageIO.write(backend.noise2DToGrayscale(), "png", new java.io.File("heightmap.png"));
        }
    }
}
