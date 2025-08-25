package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import de.verdox.util.FormatUtil;

import java.util.Arrays;

public class CPUJavaJtpBackend extends CPUJavaNoiseBackend {
    public CPUJavaJtpBackend(Device preferred, boolean vectorized, boolean optimizeCache, float[] result, int w, int h, int d) {
        super(preferred, vectorized, optimizeCache, result, w, h, d);
        this.executionMode = Kernel.EXECUTION_MODE.JTP;
    }

    protected AbstractSimplexNoise3DAparapiKernel setup() {
        this.use3DRange = false;
        this.local1D = 0;
        this.slabDepth = Math.min(32, depth);
        return super.setup();
    }

    @Override
    protected int threadsUsed() {
        return Runtime.getRuntime().availableProcessors();
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float f) {
        final int plane = width * height;
        if (optimizeCache) {
            kernel.setExecutionMode(executionMode);

            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                final int dz = Math.min(slabDepth, depth - zStart);

                for (int yStart = 0; yStart < height; yStart += rowsPerTask) {
                    final int rows = Math.min(rowsPerTask, height - yStart);

                    // 3) Kernel auf den kleinen Ausschnitt loslassen
                    //    Achtung: 1D-Kernel decodiert i -> x,y,z aus width/height/depth,
                    //    also geben wir height=rows, depth=dz und base=0.
                    kernel.bindOutput(cacheSlab);
                    kernel.setParameters(
                            x0,
                            y0 + yStart * f,       // Y-Offset in Weltkoords
                            z0 + zStart * f,       // Z-Offset in Weltkoords
                            width, rows, dz,       // lokale Dimensionen des Slabs
                            f,
                            0                      // base=0, weil slab nur diesen Block enthält
                    );

                    final int elems = width * rows * dz;
                    kernel.execute(Range.create(elems));

                    final int rowStride = width * rows;
                    for (int z = 0; z < dz; z++) {
                        final int src = z * rowStride;
                        final int dst = (zStart + z) * plane + yStart * width;
                        System.arraycopy(cacheSlab, src, result, dst, rowStride);
                        Arrays.fill(cacheSlab, 0);
                    }
                }
            }
        } else {
            kernel.bindOutput(result);
            kernel.setExecutionMode(executionMode);
            for (int zStart = 0; zStart < depth; zStart += slabDepth) {
                int dz = Math.min(slabDepth, depth - zStart);
                int slice = plane * dz;
                Range range = Range.create(slice); // keine local size im JTP
                kernel.setParameters(x0, y0, z0 + zStart * f, width, height, dz, f, zStart * plane);
                kernel.setExecutionMode(executionMode);
                kernel.execute(range);
            }
        }
    }

    @Override
    public void logSetup() {
        System.out.println();
        System.out.println("=== CPU-Parallel Backend ===");
        super.logSetup();
    }
}