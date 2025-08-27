package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import de.verdox.noise.NoiseBackend;
import de.verdox.noise.aparapi.backend.cpu.CPUJavaJtpBackend;
import de.verdox.noise.aparapi.backend.cpu.CPUJavaSeqBackend;
import de.verdox.noise.aparapi.backend.cpu.CPUOpenCLBackend;
import de.verdox.noise.aparapi.backend.gpu.BatchedGPUOpenCLBackend;
import de.verdox.noise.aparapi.backend.gpu.DirectGPUOpenCLBackend;
import de.verdox.noise.aparapi.backend.gpu.OldGPUOpenCLBackend;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoise3DAparapiKernel;

import java.util.LinkedHashSet;

public abstract class AparapiNoiseBackend<KERNEL extends AbstractSimplexNoise3DAparapiKernel> extends NoiseBackend {
    protected final Device preferredDevice;
    protected KERNEL kernel;

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

    protected abstract KERNEL setup();
    protected abstract KERNEL createKernel();

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

    public static AparapiNoiseBackend<?> constructCPU(Device.TYPE deviceType, boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return constructCPU(null, deviceType, vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> constructCPU(Device device, boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return constructCPU(device, device.getType(), vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> constructCPU(Device device, Device.TYPE deviceType, boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        AparapiNoiseBackend<?> result;

        result = switch (deviceType) {
            case CPU -> new CPUOpenCLBackend((OpenCLDevice) device, noiseField, width, height, depth);
            case JTP -> new CPUJavaJtpBackend(device, vectorized, optimizeCache, noiseField, width, height, depth);
            case SEQ -> new CPUJavaSeqBackend(device, vectorized, optimizeCache, noiseField, width, height, depth);
            default ->
                    throw new IllegalArgumentException("No cpu backend found for device type: " + device.getType().name());
        };
        Kernel kernel = result.setup();

        if (device != null) {
            LinkedHashSet<Device> preferred = new LinkedHashSet<>();
            preferred.add(device);
            KernelManager.instance().setPreferredDevices(kernel, preferred);
        }
        return result;
    }

    public static AparapiNoiseBackend<?> constructGPU(OpenCLDevice openCLDevice, boolean tiled, float[] noiseField, int width, int height, int depth) {
        AparapiNoiseBackend<?> result = null;

        if(tiled) {
            result = new BatchedGPUOpenCLBackend(openCLDevice, noiseField, width, height, depth);
        }
        else {
            result = new DirectGPUOpenCLBackend(openCLDevice, noiseField, width, height, depth);
        }

        Kernel kernel = result.setup();

        if (openCLDevice != null) {
            LinkedHashSet<Device> preferred = new LinkedHashSet<>();
            preferred.add(openCLDevice);
            KernelManager.instance().setPreferredDevices(kernel, preferred);
        }
        return result;
    }
}
