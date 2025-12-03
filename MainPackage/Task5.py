import numpy as np

def DFT_or_IDFT(signal, inverse=False, fs=1, polar=False):
 
    if polar and inverse:
        # Convert polar to complex
        mag = signal[:, 0]
        phase = signal[:, 1]
        x = mag * (np.cos(phase) + 1j * np.sin(phase))
        N = len(x)
    else:
        # Standard DFT/IDFT
        x = signal[:, 1]
        N = len(x)

    X = np.zeros(N, dtype=complex)
    sign = 1 if inverse else -1  # IDFT uses +, DFT uses -

    for k in range(N):
        summation = 0
        for n in range(N):
            summation += x[n] * np.exp(sign * 1j * 2 * np.pi * k * n / N)
        X[k] = summation / N if inverse else summation

    # Frequency axis for DFT
    freq = np.arange(N) * (fs / N)

    return freq, X
