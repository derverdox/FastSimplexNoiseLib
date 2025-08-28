package de.verdox.noise;

import com.aparapi.internal.kernel.KernelManager;
import de.verdox.noise.NoiseBackend;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.NoiseBackendBuilder.CPUParallelismMode;
import de.verdox.noise.NoiseBackendBuilder.NoiseCalculationMode;
import de.verdox.noise.aparapi.backend.AparapiNoiseBackend;
import com.aparapi.device.OpenCLDevice;
import org.junit.jupiter.api.*;

import java.util.*;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class NoiseBackendConsistencyTest {

    private static final float X0 = 0f, Y0 = 0f, Z0 = 0f;
    private static final float FREQ = 0.01f; // stabiler Test-Frequenzwert

    // Toleranzen: CPU↔CPU sehr streng, CPU↔GPU minimal lockerer
    private static final float ABS_EPS_CPU = 1e-6f;
    private static final float REL_EPS_CPU = 1e-6f;
    private static final float ABS_EPS_GPU = 1e-5f;
    private static final float REL_EPS_GPU = 1e-5f;

    @TestFactory
    Iterable<DynamicNode> allPermutations_3D_size16_and_32_agree() {
        List<Integer> sizes = List.of(16, 32);
        List<Boolean> indexings = List.of(true, false); // 1D & 3D
        List<NoiseCalculationMode> modes = List.of(NoiseCalculationMode.ALU_ONLY, NoiseCalculationMode.LOOKUP);

        boolean gpuAvailable = KernelManager.DeprecatedMethods.bestGPU() != null;

        List<DynamicNode> suites = new ArrayList<>();
        for (int size : sizes) {
            for (boolean oneD : indexings) {
                for (NoiseCalculationMode mode : modes) {
                    String display = String.format("size=%d, %s, mode=%s",
                            size, oneD ? "indexing=1D" : "indexing=3D", mode);

                    suites.add(DynamicContainer.dynamicContainer(display,
                            buildAndCompareGroup(size, oneD, mode, gpuAvailable)));
                }
            }
        }
        return suites;
    }

    private List<DynamicTest> buildAndCompareGroup(int size, boolean oneD, NoiseCalculationMode mode, boolean gpuAvailable) {
        // --- alle Konfigurationen der Backends aufbauen ---
        List<Case> cases = new ArrayList<>();

        // CPU: Simple vs CacheOnly; Vektor an/aus; Parallelismus
        for (boolean preventRam : List.of(false, true)) {
            for (boolean vectorize : List.of(false, true)) {
                for (CPUParallelismMode pm : CPUParallelismMode.values()) {
                    NoiseBackendBuilder.CPUNoiseBackendBuilder b =
                            NoiseBackendBuilder.cpu()
                                    .withSize3D(size)
                                    .with1DIndexing(oneD)
                                    .withNoiseCalculationMode(mode)
                                    .preventRamUsage(preventRam)
                                    .vectorize(vectorize)
                                    .withParallelismMode(pm);
                    NoiseBackend backend = b.build();
                    float[] out = compute3D(backend, oneD);
                    cases.add(new Case(
                            String.format("CPU[%s, vec=%s, ram=%s]", pm, vectorize, preventRam),
                            out, false));
                }
            }
        }

        // GPU: Simple vs Batched (falls verfügbar)
        if (gpuAvailable) {
            for (boolean batching : List.of(false, true)) {
                // Achtung: bevorzugtes Device kann Null sein → Assumption behandeln
                OpenCLDevice dev = (OpenCLDevice) KernelManager.DeprecatedMethods.bestGPU();
                assumeTrue(dev != null, "Kein OpenCL-GPU-Device gefunden – GPU-Fälle übersprungen.");

                NoiseBackendBuilder.GPUNoiseBackendBuilder g =
                        NoiseBackendBuilder.gpu()
                                .withSize3D(size)
                                .with1DIndexing(oneD)
                                .withNoiseCalculationMode(mode)
                                .withBatching(batching)
                                .withPreferredDevice(dev);
                NoiseBackend backend = g.build();
                float[] out = compute3D(backend, oneD);
                cases.add(new Case(
                        String.format("GPU[%s]", batching ? "batched" : "direct"),
                        out, true));
            }
        }

        // --- Paarweise Vergleiche innerhalb der Gruppe ---
        List<DynamicTest> tests = new ArrayList<>();
        for (int i = 0; i < cases.size(); i++) {
            for (int j = i + 1; j < cases.size(); j++) {
                Case a = cases.get(i), b = cases.get(j);
                String name = a.name + " == " + b.name;
                tests.add(DynamicTest.dynamicTest(name, () -> {
                    assertSameShape(a.data, b.data);
                    boolean involvesGPU = a.gpu || b.gpu;
                    float abs = involvesGPU ? ABS_EPS_GPU : ABS_EPS_CPU;
                    float rel = involvesGPU ? REL_EPS_GPU : REL_EPS_CPU;
                    assertArraysAlmostEqual(a.data, b.data, abs, rel);
                }));
            }
        }
        return tests;
    }

    /** ruft die passende 3D-Generate-Methode auf und gibt eine frische Kopie des Ergebnisses zurück */
    private float[] compute3D(NoiseBackend backend, boolean oneD) {
        // Wir erwarten hier Aparapi-Backends mit 3D-API
        if (!(backend instanceof AparapiNoiseBackend<?> apa)) {
            throw new IllegalStateException("Erwarte AparapiNoiseBackend, bekam: " + backend.getClass());
        }
        if (oneD) {
            apa.generate3DNoise1DIndexed(X0, Y0, Z0, FREQ);
        } else {
            apa.generate3DNoise3DIndexed(X0, Y0, Z0, FREQ);
        }
        float[] res = apa.getResult();
        return Arrays.copyOf(res, res.length);
    }

    private static void assertSameShape(float[] a, float[] b) {
        assertEquals(a.length, b.length, "Array-Längen unterschiedlich");
    }

    private static void assertArraysAlmostEqual(float[] a, float[] b, float absEps, float relEps) {
        assertSameShape(a, b);
        for (int i = 0; i < a.length; i++) {
            float x = a[i], y = b[i];
            float diff = Math.abs(x - y);
            float tol = Math.max(absEps, relEps * Math.max(Math.abs(x), Math.abs(y)));
            if (diff > tol) {
                fail(String.format(Locale.ROOT,
                        "Abweichung bei index=%d: a=%.9f, b=%.9f, |Δ|=%.9g > tol=%.9g",
                        i, x, y, diff, tol));
            }
        }
    }

    private record Case(String name, float[] data, boolean gpu) {}
}

