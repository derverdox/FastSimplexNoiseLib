package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.device.OpenCLDevice;
import de.verdox.noise.aparapi.kernel.scalar.AbstractScalarSimplexNoise3DAparapiKernel;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel1D;
import de.verdox.util.FormatUtil;

import java.util.Arrays;

public class CPUOpenCLBackend extends GPUOpenCLBackend {
    private final OpenCLDevice clCpu;

    public CPUOpenCLBackend(OpenCLDevice clCpu, float[] result, int w, int h, int d) {
        super(clCpu, result, w, h, d);
        this.clCpu = clCpu;
        this.executionMode = Kernel.EXECUTION_MODE.CPU;
    }

    @Override
    protected AbstractScalarSimplexNoise3DAparapiKernel setup() {
        this.use3DRange = false;
        this.local1D = Math.min(256, clCpu.getMaxWorkGroupSize());
        this.slabDepth = Math.min(32, depth);

        this.kernel = createKernel();
        return this.kernel;
    }

    @Override
    public void logSetup() {
        final int maxWG = openCLDevice.getMaxWorkGroupSize();
        final int[] maxIt = openCLDevice.getMaxWorkItemSize();
        final int CUs = openCLDevice.getMaxComputeUnits();
        final long lmem = openCLDevice.getLocalMemSize();

        System.out.println();
        System.out.println("=== CPUOpenCLBackend ===");
        System.out.println("Device: " + openCLDevice.getName() + " (" + openCLDevice.getType() + ") | Vendor: " + openCLDevice
                .getOpenCLPlatform().getVersion());

        System.out.println("CPU Cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("JVM Memory: " + FormatUtil.formatBytes2(Runtime.getRuntime()
                                                                           .freeMemory()) + "/" + FormatUtil.formatBytes2(Runtime
                .getRuntime().totalMemory()));

        System.out.println("Allocated: " + FormatUtil.formatBytes2((long) width * height * depth * Float.BYTES) + " / " + FormatUtil.formatBytes2(openCLDevice.getMaxMemAllocSize()));

        System.out.println("OpenCL: " + openCLDevice.getOpenCLPlatform().getVersion());
        System.out.println("MaxWorkGroupSize: " + maxWG + " | MaxWorkItemSizes: " + Arrays.toString(maxIt));
        System.out.println("MaxComputeUnits: " + CUs + " | LocalMemSize: " + FormatUtil.formatBytes2(lmem));

        System.out.printf("Mode: 1D | local=%d | slabDepth=%d | dims=(%d,%d,%d)%n",
                local1D, slabDepth, width, height, depth);
        System.out.println("================================");
    }
}
