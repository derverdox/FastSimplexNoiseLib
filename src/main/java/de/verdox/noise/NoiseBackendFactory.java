package de.verdox.noise;

import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import de.verdox.noise.aparapi.backend.*;

public class NoiseBackendFactory {
    public static AparapiNoiseBackend<?> gpu(OpenCLDevice device, boolean batching, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.constructGPU(device, batching, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> firstGPU(boolean batching, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.constructGPU((OpenCLDevice) KernelManager.DeprecatedMethods.firstDevice(Device.TYPE.GPU), batching, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuParallel(boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.constructCPU(KernelManager.DeprecatedMethods.firstDevice(Device.TYPE.JTP), vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuScalarParallel(boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return cpuParallel(false, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuVectorizedParallel(boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return cpuParallel(true, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuSeq(boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.constructCPU(null, Device.TYPE.SEQ, vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuScalarSeq(boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return cpuSeq(false, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuVectorizedSeq(boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return cpuSeq(true, optimizeCache, noiseField, width, height, depth);
    }
}
