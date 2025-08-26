package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import de.verdox.util.FormatUtil;
import de.verdox.util.HardwareUtil;

import java.util.Arrays;

public class CPUJavaSeqBackend extends CPUJavaNoiseBackend {

    public CPUJavaSeqBackend(Device preferred, boolean vectorized, boolean optimizeCache, float[] result, int w, int h, int d) {
        super(preferred, vectorized, optimizeCache, result, w, h, d);
        this.executionMode = Kernel.EXECUTION_MODE.SEQ;
    }

    protected AbstractSimplexNoise3DAparapiKernel setup() {
        this.use3DRange = false;
        this.local1D = 0;
        this.slabDepth = depth;
        return super.setup();
    }

    @Override
    protected int threadsUsed() {
        return 1;
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float f) {
        final int plane = width * height;
        final int L = vectorized ? HardwareUtil.getVectorLaneLength() : 1;
        final int Wv = vectorized ? (width + L - 1) / L : width;

        if (optimizeCache) {
            kernel.setExecutionMode(executionMode);

            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                final int dz = Math.min(slabDepth, depth - zStart);

                for (int yStart = 0; yStart < height; yStart += rowsPerTask) {
                    final int rows = Math.min(rowsPerTask, height - yStart);

                    // Slab vorbereiten
                    Arrays.fill(cacheSlab, 0);
                    kernel.bindOutput(cacheSlab);

                    // Lokale Dimensionen an den Kernel geben
                    kernel.setParameters(
                            x0,
                            y0 + yStart * f,      // Y-Offset in Weltkoords
                            z0 + zStart * f,      // Z-Offset in Weltkoords
                            width, rows, dz,      // lokale Slab-Dims (Breite bleibt UNGEPUFFERT)
                            f,
                            0                     // base=0 (Slab-only)
                    );

                    // WICHTIG: bei vectorized = Wv * rows * dz  (sonst: width * rows * dz)
                    final int elems = Wv * rows * dz;
                    kernel.execute(Range.create(elems, 1)); // 1D-NDRange

                    // Slab zurückkopieren (nur die realen 'width' Elemente je Zeile)
                    final int rowStride = width * rows; // Elemente pro z-Schicht im Slab
                    for (int z = 0; z < dz; z++) {
                        final int src = z * rowStride;
                        final int dst = (zStart + z) * plane + yStart * width;
                        System.arraycopy(cacheSlab, src, result, dst, rowStride);
                    }
                }
            }
        } else {
            // Direkt in das Zielarray schreiben
            kernel.bindOutput(result);
            kernel.setParameters(x0, y0, z0, width, height, depth, f, 0);
            kernel.setExecutionMode(executionMode);

            // global size: vectorized => Wv * height * depth ; sonst plane * depth
            final int global = Wv * height * depth;
            System.out.println("W: " + Wv + ", H: " + height + ", D: " + depth+" -> Range: "+Range.create(global, 1));
            kernel.execute(Range.create(global, 1));
        }
    }

    @Override
    public void logSetup() {
        System.out.println();
        System.out.println("=== CPU-Seq Backend ===");
        super.logSetup();
    }
}