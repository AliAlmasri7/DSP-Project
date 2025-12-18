import numpy as np
import os
import Task1
import CompareSignal
#Direct correlation

import numpy as np

def direct_correlation(x, y):

    if x.ndim == 2:
        indices = x[:, 0].ravel()   # first column → 1D
        x_vals = x[:, 1].astype(float)
    else:
        x_vals = np.asarray(x, dtype=float)
        indices = np.arange(len(x_vals))  # auto indices

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    N = len(x)

    # Normalization
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))

    corr = np.zeros(N)

    for k in range(N):
        s = 0.0
        for n in range(N):
            s += x[n] * y[(n + k) % N]
        corr[k] = s / norm

    fileName = r"C:\Users\Maxs_Z\Documents\DSP Tasks\MainPackage\CorrOutput.txt"
    CompareSignal.Compare_Signals(fileName,indices,corr)
    
    return corr

#Time delay using correlation

def compute_time_delay(x, y, fs):
    
    corr = direct_correlation(x, y)

    # find index of maximum absolute correlation
    j_max = np.argmax(np.abs(corr))

    Ts = 1 / fs
    delay = j_max * Ts

    return corr, j_max, delay

# Load signals from folder

def load_signals(folder_path):
    signals = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            signals.append(Task1.readSignal(os.path.join(folder_path, file)))
    return signals

#Classification

def read_1d_signal(file_path):
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return np.array(values)


def normalized_correlation(x, y):
    
    N = len(x)
    y = np.roll(y, 0)

    corr = np.zeros(N)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_std = np.std(x)
    y_std = np.std(y)

    for k in range(N):
        y_shifted = np.roll(y, k)
        corr[k] = np.sum((x - x_mean) * (y_shifted - y_mean)) / (N * x_std * y_std)

    return corr


def classify_eog_signal(test_file, classA_folder, classB_folder):
    
    test_signal = read_1d_signal(test_file)

    def avg_max_corr(folder):
        values = []
        for fname in os.listdir(folder):
            if fname.endswith('.txt'):
                template = read_1d_signal(os.path.join(folder, fname))

                # Match lengths
                N = min(len(test_signal), len(template))
                corr = normalized_correlation(
                    test_signal[:N],
                    template[:N]
                )
                values.append(np.max(np.abs(corr)))

        return np.mean(values) if values else 0

    avgA = avg_max_corr(classA_folder)
    avgB = avg_max_corr(classB_folder)

    if avgA > avgB:
        return "Class A (Down movement)", avgA, avgB
    else:
        return "Class B (Up movement)", avgA, avgB
