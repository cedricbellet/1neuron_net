import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data with numpy
dataset = np.loadtxt(
    open('./data/winequality-red.csv', 'rb'),
    delimiter=';',
    skiprows=1
)
input_vectors = np.transpose(dataset[:, 10])
training_values = np.transpose(dataset[:, 11])

# The output of the OneNeuronNet
weight = 0.3516061
bias = 1.96613827918

x = np.arange(5, 20)
y1 = [weight * a + bias for a in x]

# The output of the statistical analysis
intercept = 1.8750
slope = 0.3608

y2 = [slope * a + intercept for a in x]

# Plot the solutions
plt.plot(input_vectors, training_values, 'ro', x, y1, 'b--', x, y2, 'r-')
plt.show()

# Calc the costs
y1 = [weight * a + bias for a in input_vectors]
y2 = [slope * a + intercept for a in input_vectors]
size = training_values.size

C1 = 1./(2 * size) * np.sum((training_values -  y1) ** 2)
C2 = 1./(2 * size) * np.sum((training_values -  y2) ** 2)

print C1, C2
