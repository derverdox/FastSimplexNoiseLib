package de.verdox.noise.aparapi.kernel.scalar;

public class ScalarSimplexNoise3DKernel3D extends AbstractScalarSimplexNoise3DAparapiKernel {
    @Override
    public void run() {
        int x = getGlobalId(0);
        int y = getGlobalId(1);
        int z = getGlobalId(2);
        int li = (z * argHeight + y) * argWidth + x;
        int idx = argBase + li;

        result[idx] = scalarNoise(argX + x * argFrequency,
                argY + y * argFrequency,
                argZ + z * argFrequency);
    }
}
