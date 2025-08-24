package de.verdox;

public class Main {
    public static void main(String[] args) {
        float[] result = NoiseEngine.generate(Backend.GPU, true, 0, 0, 0, 16, 16, 16, 1, 1, 1);
    }
}
