package de.verdox.noise.aparapi.kernel.scalar;

public class ScalarSimplexNoise3DKernel1D extends AbstractScalarSimplexNoise3DAparapiKernel {
    @Override
    public void run() {
        int i = getGlobalId(0);

        int x = i % gridWidth;
        int y = (i / gridWidth) % gridHeight;
        int z = i / (gridWidth * gridHeight);

        noiseResult[baseIndex +  i] = cpuScalarNoiseLookup(
                baseX + x * frequency,
                baseY + y * frequency,
                baseZ + z * frequency
        );
    }
}
