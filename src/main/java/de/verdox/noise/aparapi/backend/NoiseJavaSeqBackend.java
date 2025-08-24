package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import de.verdox.noise.AparapiBackendUtil;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel1D;
import de.verdox.noise.aparapi.kernel.scalar.VectorizedSimplexNoise3DKernel1D;

public class NoiseJavaSeqBackend extends AparapiNoiseBackend<AbstractSimplexNoise3DAparapiKernel> {
    private final boolean vectorized;

    public NoiseJavaSeqBackend(Device preferred, boolean vectorized, float[] result, int w, int h, int d) {
        super(preferred, result, w, h, d);
        this.vectorized = vectorized;
        this.executionMode = Kernel.EXECUTION_MODE.SEQ;
    }

    protected AbstractSimplexNoise3DAparapiKernel setup() {
        this.use3DRange = false;
        this.local1D = 0;
        this.slabDepth = depth;

        this.kernel = vectorized ? new VectorizedSimplexNoise3DKernel1D() : new ScalarSimplexNoise3DKernel1D();
        return this.kernel;
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float f) {
        final int plane = width * height;
        kernel.bindOutput(result);
        int slice = plane * depth;
        Range range = Range.create(slice, 1);
        kernel.setParameters(x0, y0, z0, width, height, depth, f, 0);
        kernel.setExecutionMode(executionMode);
        kernel.execute(range);
    }

    @Override
    public void logSetup() {
        System.out.println("=== CPU-Seq Backend ===");
        System.out.println("Device: Sequential CPU");
        System.out.println("CPU Cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("JVM Memory: " + AparapiBackendUtil.formatBytes(Runtime.getRuntime()
                                                                                  .freeMemory()) + "/" + AparapiBackendUtil.formatBytes(Runtime
                .getRuntime().totalMemory()));
        System.out.println("Allocated: " + AparapiBackendUtil.formatBytes((long) width * height * depth * Float.BYTES));
        System.out.printf("Mode: 1D | local=%d | slabDepth=%d | dims=(%d,%d,%d)%n",
                local1D, slabDepth, width, height, depth);
        System.out.println("================================");
    }
}