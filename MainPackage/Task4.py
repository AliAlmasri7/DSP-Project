import numpy as np
import Task1


def first_derivative(fileName):
    signal = Task1.readSignal(fileName)
    n = len(signal)
    result = []

    for i in range(1, n):
        y = int(signal[i, 1] - signal[i-1, 1]) 
        result.append([i-1, y]) 
       
    return np.array(result)




def second_derivative(fileName):
    signal = Task1.readSignal(fileName)
    n = len(signal)
    result = []

    # Compute second derivative
    for i in range(1, n-1):
        y = int(signal[i+1, 1] - 2*signal[i, 1] + signal[i-1, 1])
        result.append([i-1, y])  
     
    return np.array(result)



def moving_average(signal, window_size):
  
    n = len(signal)
    if window_size < 1 or window_size > n:
        raise ValueError("Window size must be >=1 and <= signal length")
    
    result = []
    
    for i in range(n - window_size + 1):
        y = np.mean(signal[i:i+window_size, 1])
        result.append([i, y])
        
    
    return np.array(result)


def convolution(signal1, signal2):

    n1 = signal1[:, 0].astype(int)
    x = signal1[:, 1]
    n2 = signal2[:, 0].astype(int)
    h = signal2[:, 1]

    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1
    
    y = np.zeros(len_y)

    for i in range(len_x):
        for j in range(len_h):
            y[i + j] += x[i] * h[j]

    n_start = n1[0] + n2[0]
    n_end = n1[-1] + n2[-1]
    n = np.arange(n_start, n_end + 1)


    return np.column_stack((n, y))

