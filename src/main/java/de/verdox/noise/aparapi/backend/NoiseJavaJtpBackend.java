package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import de.verdox.noise.AparapiBackendUtil;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel1D;
import de.verdox.noise.aparapi.kernel.scalar.VectorizedSimplexNoise3DKernel1D;

import java.util.Arrays;

public class NoiseJavaJtpBackend extends AparapiNoiseBackend<AbstractSimplexNoise3DAparapiKernel> {
    private final boolean vectorized;

    public NoiseJavaJtpBackend(Device preferred, boolean vectorized, float[] result, int w, int h, int d) {
        super(preferred, result, w, h, d);
        this.vectorized = vectorized;
        this.executionMode = Kernel.EXECUTION_MODE.JTP;
    }

    protected AbstractSimplexNoise3DAparapiKernel setup() {
        this.use3DRange = false;
        this.local1D = 0;
        this.slabDepth = Math.min(32, depth);

        this.kernel = vectorized ? new VectorizedSimplexNoise3DKernel1D() : new ScalarSimplexNoise3DKernel1D();
        return kernel;
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float f) {
        final int plane = width * height;
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

    @Override
    public void logSetup() {
        final int maxWG = preferredDevice.getMaxWorkGroupSize();
        final int[] maxIt = preferredDevice.getMaxWorkItemSize();

        System.out.println();
        System.out.println("=== CPU-Parallel Backend ===");
        System.out.println("Device: " + preferredDevice.getShortDescription() + " (" + preferredDevice.getType() + ")");

        System.out.println("CPU Cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("JVM Memory: " + AparapiBackendUtil.formatBytes(Runtime.getRuntime()
                                                                                  .freeMemory()) + "/" + AparapiBackendUtil.formatBytes(Runtime
                .getRuntime().totalMemory()));
/*        if(vectorized) {
            System.out.println("SIMD Details:");
            System.out.println("\tFloat Lanes: "+SPEC_F.length());
            System.out.println("\tRegister Size: "+(SPEC_F.length() * SPEC_F.elementSize())+" B");
            System.out.println("\t");
            System.out.println("\tInteger Lanes: "+SPEC_I.length());
            System.out.println("\tRegister Size: "+(SPEC_I.length() * SPEC_I.elementSize())+" B");
        }*/

        System.out.println("Allocated: " + AparapiBackendUtil.formatBytes((long) width * height * depth * Float.BYTES));

        System.out.println("MaxWorkGroupSize: " + maxWG + " | MaxWorkItemSizes: " + Arrays.toString(maxIt));

        System.out.printf("Mode: 1D | local=%d | slabDepth=%d | dims=(%d,%d,%d)%n",
                local1D, slabDepth, width, height, depth);
        System.out.println("================================");
    }
}