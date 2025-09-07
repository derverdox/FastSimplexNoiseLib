package de.verdox.util;

public class LODUtil {

    /**
     * LOD für 3D Volumen (X/Y/Z). chunkX/chunkY/chunkZ sind Volumen-Chunk-Indices auf LOD0.
     */
    public static LOD3DParams computeLOD3D(
            int width, int height, int depth,
            float x0, float y0, float z0,
            float frequency,
            int lod,
            int chunkX, int chunkY, int chunkZ,
            LODMode mode
    ) {
        int scale = 1 << lod;

        if (mode == LODMode.TILE_PYRAMID) {
            int ixL = chunkX >> lod, iyL = chunkY >> lod, izL = chunkZ >> lod;
            float spanX = width * frequency * scale;
            float spanY = height * frequency * scale;
            float spanZ = depth * frequency * scale;
            float baseX = x0 + ixL * spanX;
            float baseY = y0 + iyL * spanY;
            float baseZ = z0 + izL * spanZ;
            return new LOD3DParams(width, height, depth, baseX, baseY, baseZ, frequency * scale);
        } else { // CHUNK_LOCAL
            float baseX = x0 + chunkX * width * frequency;
            float baseY = y0 + chunkY * height * frequency;
            float baseZ = z0 + chunkZ * depth * frequency;
            int wL = Math.max(1, width / scale);
            int hL = Math.max(1, height / scale);
            int dL = Math.max(1, depth / scale);
            return new LOD3DParams(wL, hL, dL, baseX, baseY, baseZ, frequency);
        }
    }

    /** 2D Heightmap (X/Z). y spielt keine Rolle. */
    public static LOD2DParams computeLOD2D(
            int width, int depth,
            float x0, float z0,
            float frequency,
            int lodLevel,
            LODMode mode
    ) {
        final int scale = 1 << Math.max(0, lodLevel);
        switch (mode) {
            case TILE_PYRAMID:
                // Gleiches Raster, größere Schrittweite
                return new LOD2DParams(
                        width, depth,
                        x0, z0,
                        frequency * scale
                );
            case CHUNK_LOCAL:
            default:
                // Kleineres Raster, gleiche Schrittweite
                int wL = atLeast1(width / scale);
                int dL = atLeast1(depth / scale);
                return new LOD2DParams(
                        wL, dL,
                        x0, z0,
                        frequency
                );
        }
    }

    /** 3D Volumen (X/Y/Z). */
    public static LOD3DParams computeLOD3D(
            int width, int height, int depth,
            float x0, float y0, float z0,
            float frequency,
            int lodLevel,
            LODMode mode
    ) {
        final int scale = 1 << Math.max(0, lodLevel);
        switch (mode) {
            case TILE_PYRAMID:
                // Gleiches Raster, größere Schrittweite
                return new LOD3DParams(
                        width, height, depth,
                        x0, y0, z0,
                        frequency * scale
                );
            case CHUNK_LOCAL:
            default:
                // Kleineres Raster, gleiche Schrittweite
                int wL = atLeast1(width  / scale);
                int hL = atLeast1(height / scale);
                int dL = atLeast1(depth  / scale);
                return new LOD3DParams(
                        wL, hL, dL,
                        x0, y0, z0,
                        frequency
                );
        }
    }

    // OPTIONAL: Varianten mit LOD-Kachel-Ausrichtung (wenn du LOD-Tile-Indizes benutzen willst)
    public static LOD2DParams computeLOD2DAligned(
            int width, int depth,
            float x0, float z0,
            float frequency,
            int lodLevel, int tileX, int tileZ, // LOD-Kachel-Indizes
            LODMode mode
    ) {
        final int scale = 1 << Math.max(0, lodLevel);
        if (mode == LODMode.TILE_PYRAMID) {
            float spanX = width * frequency * scale;
            float spanZ = depth * frequency * scale;
            float baseX = x0 + tileX * spanX;
            float baseZ = z0 + tileZ * spanZ;
            return new LOD2DParams(width, depth, baseX, baseZ, frequency * scale);
        } else {
            int wL = atLeast1(width / scale);
            int dL = atLeast1(depth / scale);
            float baseX = x0 + tileX * (width * frequency);
            float baseZ = z0 + tileZ * (depth * frequency);
            return new LOD2DParams(wL, dL, baseX, baseZ, frequency);
        }
    }

    public static LOD3DParams computeLOD3DAligned(
            int width, int height, int depth,
            float x0, float y0, float z0,
            float frequency,
            int lodLevel, int tileX, int tileY, int tileZ,
            LODMode mode
    ) {
        final int scale = 1 << Math.max(0, lodLevel);
        if (mode == LODMode.TILE_PYRAMID) {
            float spanX = width  * frequency * scale;
            float spanY = height * frequency * scale;
            float spanZ = depth  * frequency * scale;
            float baseX = x0 + tileX * spanX;
            float baseY = y0 + tileY * spanY;
            float baseZ = z0 + tileZ * spanZ;
            return new LOD3DParams(width, height, depth, baseX, baseY, baseZ, frequency * scale);
        } else {
            int wL = atLeast1(width  / scale);
            int hL = atLeast1(height / scale);
            int dL = atLeast1(depth  / scale);
            float baseX = x0 + tileX * (width  * frequency);
            float baseY = y0 + tileY * (height * frequency);
            float baseZ = z0 + tileZ * (depth  * frequency);
            return new LOD3DParams(wL, hL, dL, baseX, baseY, baseZ, frequency);
        }
    }

    private static int atLeast1(int v) { return (v <= 0) ? 1 : v; }

    /**
     * LOD für 2D Heightmap (X/Z). chunkX/chunkZ sind Chunk-Indices auf LOD0.
     */
    public static LOD2DParams computeLOD2D(
            int width, int depth,
            float x0, float z0,
            float frequency,
            int lod,
            int chunkX, int chunkZ,
            LODMode mode
    ) {
        int scale = 1 << lod;

        if (mode == LODMode.TILE_PYRAMID) {
            // größere Tiles auf LOD L
            int ixL = chunkX >> lod;
            int izL = chunkZ >> lod;
            float tileSpanX = width * frequency * scale;
            float tileSpanZ = depth * frequency * scale;
            float baseX = x0 + ixL * tileSpanX;
            float baseZ = z0 + izL * tileSpanZ;
            return new LOD2DParams(width, depth, baseX, baseZ, frequency * scale);
        } else { // CHUNK_LOCAL
            // selbes Weltgebiet des Chunks, nur gröberes Raster
            float baseX = x0 + chunkX * width * frequency;
            float baseZ = z0 + chunkZ * depth * frequency;
            int wL = Math.max(1, width / scale);
            int dL = Math.max(1, depth / scale);
            return new LOD2DParams(wL, dL, baseX, baseZ, frequency);
        }
    }

    public enum LODMode {TILE_PYRAMID, CHUNK_LOCAL}

    public record LOD2DParams(int widthLOD, int depthLOD, float baseX, float baseZ, float frequencyLOD) {
    }

    public record LOD3DParams(int widthLOD, int heightLOD, int depthLOD, float baseX, float baseY, float baseZ,
                              float frequencyLOD) {
    }
}
