package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import de.verdox.noise.NoiseBackend;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;

import java.util.LinkedHashSet;

public abstract class AparapiNoiseBackend<BACKEND extends AbstractSimplexNoise3DAparapiKernel> extends NoiseBackend {
    protected final Device preferredDevice;
    protected BACKEND kernel;

    protected boolean use3DRange;
    protected int localX, localY, localZ;
    protected int local1D;
    protected int slabDepth;
    protected Kernel.EXECUTION_MODE executionMode;

    public AparapiNoiseBackend(Device preferredDevice, float[] result, int width, int height, int depth) {
        super(result, width, height, depth);
        this.preferredDevice = preferredDevice;
    }

    @Override
    public void rebind(float[] result, int width, int height, int depth) {
        super.rebind(result, width, height, depth);
        dispose();
        this.kernel = setup();
    }

    protected abstract BACKEND setup();

    @Override
    public void dispose() {
        this.kernel.dispose();
    }

    protected static int roundUp(int n, int m) {
        return (m == 0) ? n : ((n + m - 1) / m) * m;
    }

    public abstract void generate1D(float x0, float y0, float z0, float frequency);

    @Override
    public void generate(float x0, float y0, float z0, float frequency) {
        generate1D(x0, y0, z0, frequency);
    }

    public static AparapiNoiseBackend<?> construct(Device.TYPE deviceType, boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return construct(null, deviceType, vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> construct(Device device, boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return construct(device, device.getType(), vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> construct(Device device, Device.TYPE deviceType, boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        AparapiNoiseBackend<?> result;

        result = switch (deviceType) {
            case GPU -> new GPUOpenCLBackend((OpenCLDevice) device, noiseField, width, height, depth);
            case CPU -> new CPUOpenCLBackend((OpenCLDevice) device, noiseField, width, height, depth);
            case JTP -> new CPUJavaJtpBackend(device, vectorized, optimizeCache, noiseField, width, height, depth);
            case SEQ -> new CPUJavaSeqBackend(device, vectorized, optimizeCache, noiseField, width, height, depth);
            default ->
                    throw new IllegalArgumentException("No backend found for device type: " + device.getType().name());
        };
        Kernel kernel = result.setup();

        if (device != null) {
            LinkedHashSet<Device> preferred = new LinkedHashSet<>();
            preferred.add(device);
            KernelManager.instance().setPreferredDevices(kernel, preferred);
        }
        return result;
    }
}
