import Task1
import numpy as np

def quantizeSignal(fileName, flag, levels=0, numOfBits=0):
    signal = Task1.readSignal(fileName)

    if flag == 0:
        levels = np.power(2, numOfBits)

    minNum = np.min(signal[:, 1])
    maxNum = np.max(signal[:, 1])

    delta = (maxNum - minNum) / levels
    intervals = np.zeros((levels, 2))

    amountOfIncrease = minNum
    for i in range(intervals.shape[0]):
        intervals[i][0] = amountOfIncrease
        amountOfIncrease += delta
        intervals[i][1] = amountOfIncrease

    midPoints = (intervals[:, 0] + intervals[:, 1]) / 2
    outputValues = np.zeros((signal.shape[0], 4), dtype=object)

    for i in range(signal.shape[0]):
        sample = signal[i, 1]
        for j in range(levels):
            if intervals[j, 0] <= sample < intervals[j, 1] or (j == levels - 1 and sample == intervals[j, 1]):
                intervalIndex = j
                break

        quantizedValue = midPoints[intervalIndex]
        error = quantizedValue - sample
        bits = int(np.ceil(np.log2(levels)))
        encodedValue = format(intervalIndex, f'0{bits}b')

        outputValues[i] = [intervalIndex + 1, encodedValue, quantizedValue, error]

    # Compute Squared Average Error (MSE)
    mse = np.mean((outputValues[:, 3].astype(float)) ** 2)

    return outputValues, signal, mse
