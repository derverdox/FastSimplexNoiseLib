package de.verdox;

public enum Backend {
    CPU_SCALAR_SEQ,
    CPU_SCALAR_PARALLEL,
    CPU_VECTORIZED_SEQ,
    CPU_VECTORIZED_PARALLEL,
    GPU
}
