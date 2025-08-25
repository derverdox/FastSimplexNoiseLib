package de.verdox;

import de.verdox.noise.NoiseBackend;
import de.verdox.noise.NoiseBackendFactory;
import de.verdox.noise.NoiseEngine3D;
import org.openjdk.jmh.annotations.*;

import static java.util.concurrent.TimeUnit.SECONDS;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(SECONDS)
@State(Scope.Benchmark)
@Threads(1)
@Warmup(iterations = 5, time = 1, timeUnit = SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = SECONDS)
@Fork(2)
public class CPUNoiseBenchJmh {

    // --- Backend- und Lauf-Parameter ---
    @Param({"SEQUENTIAL", "PARALLEL"})
    public String backend;

    // --- Variante A: Größe als Tripel-String (am bequemsten) ---
    @Param({"16x16x16", "32x32x32", "64x64x64", "128x128x128", "256x256x256", "512x512x512", "1024x1024x1024"})
    public String shape; // wird in @Setup geparst

    @Param({"true", "false"})
    public boolean vectorized;

    @Param({"true", "false"})
    public boolean cacheOptimized;

    // Interne, aus 'shape' geparste Dimensionen:
    private int nx, ny, nz;
    private float[] result;
    private NoiseBackend noiseBackend;
    private NoiseEngine3D engine;

    @Setup(Level.Trial)
    public void setup() {
        String[] parts = shape.toLowerCase().split("x");
        if (parts.length != 3) {
            throw new IllegalArgumentException("shape must be like '16x16x16'");
        }
        try {
            nx = Integer.parseInt(parts[0].trim());
            ny = Integer.parseInt(parts[1].trim());
            nz = Integer.parseInt(parts[2].trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("shape must contain integers: " + shape, e);
        }



        result = new float[nx * ny * nz];
        noiseBackend = switch (CPUBackend.valueOf(backend)) {
            case PARALLEL -> NoiseBackendFactory.cpuParallel(vectorized, cacheOptimized, result, nx, ny, nz);
            case SEQUENTIAL -> NoiseBackendFactory.cpuSeq(vectorized, cacheOptimized, result, nx, ny, nz);
        };
        engine = new NoiseEngine3D(noiseBackend);
        noiseBackend.logSetup();
    }

    @Benchmark
    public float[] benchNoise() {
        engine.computeNoise(0, 0, 0, 0.009f);
        return result;
    }

    @TearDown(Level.Trial)
    public void cleanUp() {
        noiseBackend.dispose();
    }
}