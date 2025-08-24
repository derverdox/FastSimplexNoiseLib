package de.verdox.noise.aparapi.kernel.scalar;

import de.verdox.noise.aparapi.kernel.vectorized.AbstractVectorizedSimplexNoise3DAparapiKernel;

public class VectorizedSimplexNoise3DKernel1D extends AbstractVectorizedSimplexNoise3DAparapiKernel {
    @Override
    public void run() {
        int i = getGlobalId(0);

        // argWidth*argHeight*argDepth == plane*dz pro Slab
        int sliceElems = argWidth * argHeight * argDepth;
        if (i >= sliceElems) return;                 // Padding-Guard

        int idx = argBase + i;
        if (idx < 0 || idx >= result.length) return; // Sicherheits-Guard

        int x = i % argWidth;
        int y = (i / argWidth) % argHeight;
        int z = i / (argWidth * argHeight);

        result[idx] = scalarNoise(
                argX + x * argFrequency,
                argY + y * argFrequency,
                argZ + z * argFrequency
        );
    }
}
