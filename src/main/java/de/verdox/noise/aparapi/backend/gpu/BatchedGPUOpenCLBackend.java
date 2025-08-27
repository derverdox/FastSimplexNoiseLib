package de.verdox.noise.aparapi.backend.gpu;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import de.verdox.noise.OpenCLTuner;
import de.verdox.noise.aparapi.backend.AparapiNoiseBackend;
import de.verdox.noise.aparapi.kernel.scalar.AbstractScalarSimplexNoise3DAparapiKernel;
import de.verdox.noise.aparapi.kernel.scalar.BatchedScalarSimplexNoise3DKernel1D;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel1D;

import java.util.ArrayList;
import java.util.List;

public class BatchedGPUOpenCLBackend
        extends AparapiNoiseBackend<BatchedScalarSimplexNoise3DKernel1D> {

    public static final int TILE = 128;
    private final List<Tile> tiles = new ArrayList<>();

    public BatchedGPUOpenCLBackend(OpenCLDevice preferredDevice, float[] result,
                                   int width, int height, int depth) {
        super(preferredDevice, result, width, height, depth);
    }

    @Override
    protected BatchedScalarSimplexNoise3DKernel1D setup() {
        // Kernel erstellen & auf explizite Transfers stellen
        this.kernel = createKernel();
        kernel.setExplicit(true);
        kernel.bindOutput(result); // write-only → kein put()

        // Tiles vorab planen (1D-Range; local=256, global gepaddet)
        for (int zBase = 0; zBase < depth; zBase += TILE) {
            int td = Math.min(TILE, depth - zBase);
            for (int yBase = 0; yBase < height; yBase += TILE) {
                int th = Math.min(TILE, height - yBase);
                for (int xBase = 0; xBase < width; xBase += TILE) {
                    int tw = Math.min(TILE, width - xBase);

                    OpenCLTuner.Plan plan = OpenCLTuner.plan((OpenCLDevice) preferredDevice, tw, th, td,
                            /*prefer3D=*/false,
                            /*alignBaseIndexTo32=*/true);
                    Range tileRange = plan.toRange();

                    int baseIndex = xBase + yBase * width + zBase * width * height;

                    tiles.add(new Tile(xBase, yBase, zBase, tw, th, td, baseIndex, tileRange));
                }
            }
        }
        return this.kernel;
    }

    @Override
    protected BatchedScalarSimplexNoise3DKernel1D createKernel() {
        return new BatchedScalarSimplexNoise3DKernel1D();
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float frequency) {
        // Tiles rechnen; kein get() im Loop
        for (Tile t : tiles) {
            float bx = x0 + t.bx * frequency;
            float by = y0 + t.by * frequency;
            float bz = z0 + t.bz * frequency;

            kernel.setParameters(bx, by, bz, t.tw, t.th, t.td, frequency, t.baseIndex);
            kernel.globalWidth = width;
            kernel.globalHeight = height;
            kernel.execute(t.range);
        }
        // EIN get() am Ende (oder blockweise, falls gewünscht)
        kernel.get(result);
    }

    @Override
    public void logSetup() { /* optional */ }

    private record Tile(int bx, int by, int bz, int tw, int th, int td,
                        int baseIndex, Range range) {
    }
}

