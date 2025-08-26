package de.verdox.noise.aparapi.kernel.scalar;

public class ScalarSimplexNoise3DKernel1D extends AbstractScalarSimplexNoise3DAparapiKernel {
    @Override
    public void run() {
        int i = getGlobalId(0);

        // argWidth*argHeight*argDepth == plane*dz pro Slab
        int sliceElems = gridWidth * gridHeight * gridDepth;
        if (i >= sliceElems) return;

        int idx = baseIndex + i;
        if (idx < 0 || idx >= noiseResult.length) return;

        int x = i % gridWidth;
        int y = (i / gridWidth) % gridHeight;
        int z = i / (gridWidth * gridHeight);

        noiseResult[idx] = cpuScalarNoiseLookup(
                baseX + x * frequency,
                baseY + y * frequency,
                baseZ + z * frequency
        );
    }
}
