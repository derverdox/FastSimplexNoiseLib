package de.verdox.noise.aparapi.kernel.scalar;

public class ScalarSimplexNoise3DKernel3D extends AbstractScalarSimplexNoise3DAparapiKernel {
    @Override
    public void run() {
        int x = getGlobalId(0);
        int y = getGlobalId(1);
        int z = getGlobalId(2);
        int li = (z * gridHeight + y) * gridWidth + x;
        int idx = baseIndex + li;

        noiseResult[idx] = scalarNoiseAluOnly(baseX + x * frequency,
                baseY + y * frequency,
                baseZ + z * frequency);
    }
}
