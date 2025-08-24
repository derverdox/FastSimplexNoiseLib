package de.verdox.noise;

import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import de.verdox.noise.aparapi.backend.*;

public class NoiseBackendFactory {
    public static AparapiNoiseBackend<?> gpu(OpenCLDevice device, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(device, Device.TYPE.GPU, false, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> best(boolean vectorizedIfAvailable, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(KernelManager.instance()
                                      .bestDevice(), vectorizedIfAvailable, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> firstGPU(float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(KernelManager.DeprecatedMethods.firstDevice(Device.TYPE.GPU), Device.TYPE.GPU, false, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuParallel(boolean vectorized, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(KernelManager.DeprecatedMethods.firstDevice(Device.TYPE.JTP), vectorized, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuScalarParallel(float[] noiseField, int width, int height, int depth) {
        return cpuParallel(false, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuVectorizedParallel(float[] noiseField, int width, int height, int depth) {
        return cpuParallel(true, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuSeq(boolean vectorized, float[] noiseField, int width, int height, int depth) {
        return AparapiNoiseBackend.construct(null, Device.TYPE.SEQ, vectorized, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuScalarSeq(float[] noiseField, int width, int height, int depth) {
        return cpuSeq(false, noiseField, width, height, depth);
    }

    public static AparapiNoiseBackend<?> cpuVectorizedSeq(float[] noiseField, int width, int height, int depth) {
        return cpuSeq(true, noiseField, width, height, depth);
    }
}
