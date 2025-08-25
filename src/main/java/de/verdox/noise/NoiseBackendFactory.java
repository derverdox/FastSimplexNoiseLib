package de.verdox.noise;

import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import de.verdox.noise.aparapi.backend.*;

public class NoiseBackendFactory {
    public static AparapiNoiseBackend<?> gpu(OpenCLDevice device, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(device, Device.TYPE.GPU, false, false, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> best(boolean vectorizedIfAvailable, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(KernelManager.instance()
                                      .bestDevice(), vectorizedIfAvailable, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> firstGPU(float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(KernelManager.DeprecatedMethods.firstDevice(Device.TYPE.GPU), Device.TYPE.GPU, false, false, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuParallel(boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(KernelManager.DeprecatedMethods.firstDevice(Device.TYPE.JTP), vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuScalarParallel(boolean optimizeCache,float[] noiseField, int width, int height, int depth) {
        return cpuParallel(false, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuVectorizedParallel(boolean optimizeCache,float[] noiseField, int width, int height, int depth) {
        return cpuParallel(true, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuSeq(boolean vectorized, boolean optimizeCache, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(null, Device.TYPE.SEQ, vectorized, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuScalarSeq(boolean optimizeCache,float[] noiseField, int width, int height, int depth) {
        return cpuSeq(false, optimizeCache, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuVectorizedSeq(boolean optimizeCache,float[] noiseField, int width, int height, int depth) {
        return cpuSeq(true, optimizeCache, noiseField, width, height, depth);
    }
}
