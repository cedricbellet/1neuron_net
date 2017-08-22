"""Regressions using the ONN."""
# pylint: disable=C0103
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet
import matplotlib.pyplot as plt

# Load the data with numpy
dataset = np.loadtxt(
    open('./data/multilinear.csv', 'rb'),
    delimiter=',',
    skiprows=1
)

# Set the parameters
NUMBER_OF_VARIABLES = 3
ACTIVATION_FUNCTION = m.Identity
COST_FUNCTION = m.SquaredError
EPOCHS = 400
STEP = 0.01

# Extract the input and training vectors
input_vectors = np.transpose(dataset[:, 0:NUMBER_OF_VARIABLES])
training_values = np.transpose(dataset[:, NUMBER_OF_VARIABLES])

# Create a single neuron net
onn = OneNeuronNet(
    number_of_dendrons=NUMBER_OF_VARIABLES,
    activation_function=ACTIVATION_FUNCTION,
    cost_function=COST_FUNCTION
)

# Train the neural net
cost_array = []
bias_array = []
weight_array = [[] for i in range(NUMBER_OF_VARIABLES)]

for i in range(EPOCHS):
    cost = onn.minibatch_train(
        input_vectors,
        training_values,
        step=STEP
    )
    cost_array.append(cost)
    bias_array.append(onn.bias)
    for j in range(NUMBER_OF_VARIABLES):
        weight_array[j].append(onn.weights[j])

# Print the cost, bias, and weights
print onn.bias, onn.weights

# Plot the cost, bias, and weights
base_subplot_number = 200 + 10 + NUMBER_OF_VARIABLES * 100
plt.subplot(base_subplot_number + 1)
plt.yscale('log')
plt.plot(cost_array)
plt.subplot(base_subplot_number + 2)
plt.plot(bias_array)
for i in range(NUMBER_OF_VARIABLES):
    subplot_number = base_subplot_number + 3 + i * 1
    plt.subplot(subplot_number)
    plt.plot(weight_array[i])

plt.show()
