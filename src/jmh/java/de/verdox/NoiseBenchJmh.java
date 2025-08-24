package de.verdox;

import org.openjdk.jmh.annotations.*;

import static java.util.concurrent.TimeUnit.SECONDS;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(SECONDS)
@State(Scope.Benchmark)
@Threads(1)
@Warmup(iterations=5, time=1, timeUnit=SECONDS)
@Measurement(iterations=10, time=1, timeUnit=SECONDS)
@Fork(2)
public class NoiseBenchJmh {

    // --- Backend- und Lauf-Parameter ---
    @Param({"GPU","VECTOR","SCALAR"})
    public String backend;

    @Param({"true", "false"})
    public boolean parallel;

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

        // Warm Up for GPU
        if(Backend.valueOf(backend).equals(Backend.GPU)) {
            NoiseEngine.generate(Backend.valueOf(backend), parallel, 0,0,0, nx,ny,nz, 1,1,1);
        }
    }

    @Benchmark
    public float[] benchChunk() {
        return NoiseEngine.generate(
                Backend.valueOf(backend), parallel,
                0, 0, 0,
                nx, ny, nz,
                1, 1, 1
        );
    }
}