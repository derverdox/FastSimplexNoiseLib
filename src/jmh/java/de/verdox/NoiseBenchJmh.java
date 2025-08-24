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
public class NoiseBenchJmh {

    // --- Backend- und Lauf-Parameter ---
    @Param({"CPU_SCALAR_SEQ", "CPU_SCALAR_PARALLEL", "CPU_VECTORIZED_SEQ", "CPU_VECTORIZED_PARALLEL", "GPU"})
    public String backend;

    // --- Variante A: Größe als Tripel-String (am bequemsten) ---
    @Param({"16x16x16", "32x32x32", "64x64x64", "128x128x128", "256x256x256", "512x512x512", "1024x1024x1024"})
    public String shape; // wird in @Setup geparst

    // Interne, aus 'shape' geparste Dimensionen:
    private int nx, ny, nz;

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
    }

    @Benchmark
    public float[] benchChunk() {
        float[] result = new float[nx * ny * nz];
        NoiseBackend noiseBackend = switch (Backend.valueOf(backend)) {
            case CPU_SCALAR_SEQ -> NoiseBackendFactory.cpuScalarSeq(result, nx, ny, nz);
            case CPU_SCALAR_PARALLEL -> NoiseBackendFactory.cpuScalarParallel(result, nx, ny, nz);
            case CPU_VECTORIZED_SEQ -> NoiseBackendFactory.cpuVectorizedParallel(result, nx, ny, nz);
            case CPU_VECTORIZED_PARALLEL -> NoiseBackendFactory.cpuVectorizedSeq(result, nx, ny, nz);
            case GPU -> NoiseBackendFactory.firstGPU(result, nx, ny, nz);
        };
        //noiseBackend.logSetup();
        NoiseEngine3D engine = new NoiseEngine3D(noiseBackend);
        engine.computeNoise(0, 0, 0, 0.009f);
        return result;
    }
}