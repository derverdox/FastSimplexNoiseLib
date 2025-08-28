package de.verdox.noise;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

public abstract class NoiseBackend {
    protected int width;
    protected int height;
    protected int depth;
    protected float[] result;

    public NoiseBackend(float[] result, int width, int height, int depth) {
        if (result.length != width * height * depth) {
            throw new IllegalArgumentException("Result array does not have the correct length");
        }
        this.result = result;
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    public void rebind(float[] result, int width, int height, int depth) {
        if (result.length != width * height * depth) {
            throw new IllegalArgumentException("Result array does not have the correct length");
        }
        this.result = result;
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    public float[] getResult() {
        return result;
    }

    public void postInit() {

    }

    public abstract void dispose();

    public abstract void generate(float x0, float y0, float z0, float frequency);

    public abstract void generate(float x0, float y0, float frequency);

    public abstract void logSetup();

    /**
     * Erstellt aus der obersten Schicht (z = depth-1) eines Z-major 3D-Felds
     * ein 8-bit Graustufenbild (TYPE_BYTE_GRAY) mit Min/Max-Normalisierung.
     */
    public BufferedImage topLayerToGrayscale() {
        int plane = width * height;
        if (result.length < (long) plane * depth) {
            throw new IllegalArgumentException("field too small for given dims");
        }

        int zTop = depth - 1;
        int base = zTop * plane;

        // 1) Min/Max der Schicht bestimmen (NaN ignorieren)
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < plane; i++) {
            float v = result[base + i];
            if (Float.isNaN(v)) continue;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        // Fallback falls alles NaN oder konstant
        if (!Float.isFinite(min) || !Float.isFinite(max) || min == max) {
            min = (min == max && Float.isFinite(min)) ? min - 1e-6f : -1f;
            max = (min == max && Float.isFinite(max)) ? max + 1e-6f : 1f;
        }
        float invSpan = 1.0f / (max - min);

        // 2) Bild füllen
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = img.getRaster();
        byte[] row = new byte[width];

        for (int y = 0; y < height; y++) {
            int rowOff = base + y * width;
            for (int x = 0; x < width; x++) {
                float v = result[rowOff + x];
                if (Float.isNaN(v)) v = min;               // NaN -> niedrigster Wert
                float t = (v - min) * invSpan;             // 0..1
                if (t < 0f) t = 0f;
                else if (t > 1f) t = 1f;
                int gray = Math.round(t * 255f);
                row[x] = (byte) (gray & 0xFF);
            }
            raster.setDataElements(0, y, width, 1, row);
        }
        return img;
    }

    /**
     * Optional: Beliebigen z-Index in ein Graustufenbild wandeln.
     */
    public BufferedImage layerToGrayscale(int zIndex) {
        if (zIndex < 0 || zIndex >= depth) throw new IllegalArgumentException("zIndex out of range");
        int plane = width * height;
        if (result.length < (long) plane * depth) {
            throw new IllegalArgumentException("field too small for given dims");
        }

        int base = zIndex * plane;
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < plane; i++) {
            float v = result[base + i];
            if (Float.isNaN(v)) continue;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        if (!Float.isFinite(min) || !Float.isFinite(max) || min == max) {
            min = (min == max && Float.isFinite(min)) ? min - 1e-6f : -1f;
            max = (min == max && Float.isFinite(max)) ? max + 1e-6f : 1f;
        }
        float invSpan = 1.0f / (max - min);

        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = img.getRaster();
        byte[] row = new byte[width];

        for (int y = 0; y < height; y++) {
            int rowOff = base + y * width;
            for (int x = 0; x < width; x++) {
                float v = result[rowOff + x];
                if (Float.isNaN(v)) v = min;
                float t = (v - min) * invSpan;
                if (t < 0f) t = 0f;
                else if (t > 1f) t = 1f;
                row[x] = (byte) Math.round(t * 255f);
            }
            raster.setDataElements(0, y, width, 1, row);
        }
        return img;
    }
}
