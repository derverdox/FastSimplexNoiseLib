package de.verdox.noise.aparapi.backend;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import de.verdox.noise.NoiseBackend;
import de.verdox.noise.NoiseBackendBuilder;
import de.verdox.noise.aparapi.kernel.AbstractSimplexNoiseKernel;

public abstract class AparapiNoiseBackend<KERNEL extends AbstractSimplexNoiseKernel> extends NoiseBackend {
    protected final Device preferredDevice;
    protected final NoiseBackendBuilder.NoiseCalculationMode calculationMode;
    protected KERNEL kernel;

    protected boolean use1DIndexing;
    protected int localX, localY, localZ;
    protected int local1D;
    protected int slabDepth;
    protected Kernel.EXECUTION_MODE executionMode;

    public AparapiNoiseBackend(Device preferredDevice, NoiseBackendBuilder.NoiseCalculationMode calculationMode, float[] result, int width, int height, int depth) {
        super(result, width, height, depth);
        this.preferredDevice = preferredDevice;
        this.calculationMode = calculationMode;
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

    public abstract void generate3DNoise1DIndexed(float x0, float y0, float z0, float frequency);

    public abstract void generate3DNoise3DIndexed(float x0, float y0, float z0, float frequency);

    public abstract void generate2DNoise1DIndexed(float x0, float y0, float frequency);

    public abstract void generate2DNoise2DIndexed(float x0, float y0, float frequency);

    @Override
    public void generate(float x0, float y0, float z0, float frequency) {
        if(use1DIndexing) {
            generate3DNoise1DIndexed(x0, y0, z0, frequency);
        }
        else {
            generate3DNoise3DIndexed(x0, y0, z0, frequency);
        }
    }

    @Override
    public void generate(float x0, float y0, float frequency) {
        if(use1DIndexing) {
            generate2DNoise1DIndexed(x0, y0, frequency);
        }
        else {
            generate2DNoise2DIndexed(x0, y0, frequency);
        }
    }
}
