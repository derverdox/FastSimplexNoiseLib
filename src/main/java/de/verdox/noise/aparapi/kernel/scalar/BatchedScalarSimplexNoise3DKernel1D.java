package de.verdox.noise.aparapi.kernel.scalar;

public class BatchedScalarSimplexNoise3DKernel1D extends AbstractScalarSimplexNoise3DAparapiKernel {
    public int globalWidth, globalHeight;
    @Override
    public void run() {
        int gid = getGlobalId(); // 1D
        int n = gridWidth * gridHeight * gridDepth;
        if (gid >= n) return; // Guard, falls global gepaddet

        int x = gid % gridWidth;
        int y = (gid / gridWidth) % gridHeight;
        int z = gid / (gridWidth * gridHeight);

        int idx = baseIndex + x + y * globalWidth + z * globalWidth * globalHeight; // x-major
        float xin = baseX + x * frequency, yin = baseY + y * frequency, zin = baseZ + z * frequency;
        noiseResult[idx] = scalarNoiseAluOnly(xin, yin, zin);
    }
}
