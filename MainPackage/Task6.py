import numpy as np
import os
import Task4

import numpy as np

# -----------------------------
# Select window based on stopband attenuation
# -----------------------------
def select_window(stopband_attenuation):
    if stopband_attenuation <= 21:
        return "rectangular"
    elif stopband_attenuation <= 44:
        return "hanning"
    elif stopband_attenuation <= 53:
        return "hamming"
    else:
        return "blackman"

# -----------------------------
# Calculate filter order
# -----------------------------
def calculate_filter_order(transition_band, fs, window_type):
    delta_f = transition_band / fs
    if window_type == "rectangular":
        N = np.ceil(0.9 / delta_f)
    elif window_type == "hanning":
        N = np.ceil(3.1 / delta_f)
    elif window_type == "hamming":
        N = np.ceil(3.3 / delta_f)
    elif window_type == "blackman":
        N = np.ceil(5.5 / delta_f)

    # Ensure odd
    if N % 2 == 0:
        N += 1

    return int(N)

# -----------------------------
# Window values
# -----------------------------
def window_value(window_type, n, N):
    #M = (N - 1) // 2
    if window_type == "rectangular":
        return 1.0
    elif window_type == "hanning":
        return 0.5 + 0.5 * np.cos(2 * np.pi * n  / N)
    elif window_type == "hamming":
        return 0.54 + 0.46 * np.cos(2 * np.pi * n  / N)
    elif window_type == "blackman":
        return 0.42 + 0.5 * np.cos(2 * np.pi * (n) / (N-1)) + 0.08 * np.cos(4 * np.pi * n  / (N-1))

# -----------------------------
# FIR design
# -----------------------------
def design_fir(filter_type, fs, stopband_attenuation, transition_band, fc=None, f1=None, f2=None):
    # Select window
    window_type = select_window(stopband_attenuation)

    # Filter order
    N = calculate_filter_order(transition_band, fs, window_type)
    M = (N - 1) // 2
    n = np.arange(-M, M + 1)

        # Normalized frequencies with half-transition-band adjustment
    if filter_type == "Low Pass":
        fc = (fc + transition_band / 2) / fs
    elif filter_type == "High Pass":
        # CORRECTED LINE: Subtract for High Pass
        fc = (fc - transition_band / 2) / fs  
    elif filter_type in ["Band Pass", "Band Stop"]:
        f1 = (f1 - transition_band / 2) / fs
        f2 = (f2 + transition_band / 2) / fs

    # Ideal impulse response
    h = []
    for i in n:
        if filter_type == "Low Pass":
            h_i = 2 * fc if i == 0 else np.sin(2 * np.pi * fc * i) / (np.pi * i)
        elif filter_type == "High Pass":
            h_i = 1 - 2 * fc if i == 0 else -np.sin(2 * np.pi * fc * i) / (np.pi * i)
        elif filter_type == "Band Pass":
            if i == 0:
                h_i = 2 * (f2 - f1)
            else:
                h_i = (np.sin(2 * np.pi * f2 * i) - np.sin(2 * np.pi * f1 * i)) / (np.pi * i)
        elif filter_type == "Band Stop":
            if i == 0:
                h_i = 1 - 2 * (f2 - f1)
            else:
                h_i = (np.sin(2 * np.pi * f1 * i) - np.sin(2 * np.pi * f2 * i)) / (np.pi * i)

        # Apply window
        h_i *= window_value(window_type, i, N)
        h.append(h_i)

    return np.array(h), n

# -------------------------------------------------
# -------------------------------------------------
# FIR CONVOLUTION using your custom method
# -------------------------------------------------
def apply_fir(signal, h):
    """
    Apply FIR filter using custom convolution function.
    signal: 2D array [[n0, x0], [n1, x1], ...]
    h: 1D array of filter coefficients
    """
    # Convert h to 2D array with indices
    n_h = np.arange(-(len(h)//2), len(h)//2 + 1)
    h_2d = np.column_stack((n_h, h))

    # Use the custom convolution
    return Task4.convolution(signal, h_2d)




# -------------------------------------------------
# Save coefficients
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def save_filter_coefficients(n, h, filename):
    filepath = os.path.join(BASE_DIR, filename)

    with open(filepath, "w") as f:
        f.write(f"{len(h)}\n")
        for i, val in zip(n, h):
            f.write(f"{i} {val:.8f}\n")

    print("Saved:", filepath)


def save_filtered_signal(signal, filename):
    filepath = os.path.join(BASE_DIR, filename)

    with open(filepath, "w") as f:
        f.write(f"{len(signal)}\n")
        for sample in signal:
            f.write(f"{int(sample[0])} {sample[1]:.8f}\n")

    print("Saved:", filepath)
    
    
    


import Task5

def apply_fir_fast(signal, h, fs):
    x = signal[:, 1]

    # Zero-padding
    L = len(x) + len(h) - 1
    x_padded = np.pad(x, (0, L - len(x)))
    h_padded = np.pad(h, (0, L - len(h)))

    # DFT
    _, X = Task5.DFT_or_IDFT(
        np.column_stack((np.arange(L), x_padded)),
        inverse=False,
        fs=fs
    )

    _, H = Task5.DFT_or_IDFT(
        np.column_stack((np.arange(L), h_padded)),
        inverse=False,
        fs=fs
    )

    # Multiply
    Y = X * H

    # IDFT
    _, y = Task5.DFT_or_IDFT(
        np.column_stack((np.arange(L), Y)),
        inverse=True,
        fs=fs
    )

    y = np.real(y[:len(x)])

    return np.column_stack((signal[:, 0], y))
