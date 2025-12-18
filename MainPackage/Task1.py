import numpy as np
import matplotlib.pyplot as plt


def readSignal(fileName):
    with open(fileName, 'r') as f:
        f.readline()
        f.readline()
        n = int(f.readline().strip())

        samples = np.zeros((n, 2))
        for i in range(n):
            line = f.readline().strip()
            if not line:
                break

            x, y = line.split()

            # Strip trailing 'f' if present (example: 20.9f → 20.9)
            if x.endswith('f') or x.endswith('F'):
                x = x[:-1]
            if y.endswith('f') or y.endswith('F'):
                y = y[:-1]

            samples[i, 0] = float(x)
            samples[i, 1] = float(y)

    return samples


def displaySignal(samples,title = "Signal"):
    plt.figure(figsize=(8,4))
    plt.stem(samples[:,0],samples[:,1])
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.show()

def addSubtractSignals(*signals,operation):
    
    all_indices = sorted(set(np.concatenate([sig[:, 0] for sig in signals])))

    result_values = np.zeros(len(all_indices))
    count = 0

    for sig in signals:
        if count != 0:
            if operation == 1:
               sig = multiplySignal(sig,-1)
        sig_dict = {int(idx): val for idx, val in sig}
        for i, idx in enumerate(all_indices):
            result_values[i] += sig_dict.get(idx, 0.0)  # add value or 0 if missing
        count+=1
    return np.column_stack((all_indices, result_values))

def multiplySignal(signal,constant):
    signal[:,1] = signal[:,1] * constant
    return signal

def shiftSignal(signal,k,sign):
    newSignal = np.zeros(signal.shape)
    
    if sign == '+':
       for i in range(signal.shape[0]):
           newSignal[i][0] = signal[i][0] - k
           newSignal[i][1] = signal[i][1]
    elif sign == '-':
       for i in range(signal.shape[0]):
           newSignal[i][0] = signal[i][0] + k
           newSignal[i][1] = signal[i][1]

    return newSignal

def foldSignal(signal):
    
    folded = signal.copy()
    folded[:,0] = -folded[:,0]

    folded = folded[folded[:,0].argsort()]

    return folded 

