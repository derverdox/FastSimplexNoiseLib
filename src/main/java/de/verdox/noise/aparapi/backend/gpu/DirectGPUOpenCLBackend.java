package de.verdox.noise.aparapi.backend.gpu;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import de.verdox.noise.OpenCLTuner;
import de.verdox.noise.aparapi.backend.AparapiNoiseBackend;
import de.verdox.noise.aparapi.kernel.scalar.AbstractScalarSimplexNoise3DAparapiKernel;
import de.verdox.noise.aparapi.kernel.scalar.ScalarSimplexNoise3DKernel1D;

public class DirectGPUOpenCLBackend extends AparapiNoiseBackend<AbstractScalarSimplexNoise3DAparapiKernel> {
    private OpenCLTuner.Plan plan;
    private Range range;

    public DirectGPUOpenCLBackend(OpenCLDevice preferredDevice, float[] result, int width, int height, int depth) {
        super(preferredDevice, result, width, height, depth);
    }

    @Override
    protected AbstractScalarSimplexNoise3DAparapiKernel setup() {
        plan = OpenCLTuner.plan((OpenCLDevice) preferredDevice,width, height, depth, false, true);
        range = plan.toRange();
        this.kernel = createKernel();
        return this.kernel;
    }

    @Override
    protected AbstractScalarSimplexNoise3DAparapiKernel createKernel() {
        return new ScalarSimplexNoise3DKernel1D();
    }

    @Override
    public void generate1D(float x0, float y0, float z0, float frequency) {
        kernel.bindOutput(result);
        kernel.setExplicit(true);
        kernel.setParameters(0, 0, 0, width, height, depth, frequency, 0);
        kernel.execute(range);
    }

    @Override
    public void logSetup() {

    }
}
