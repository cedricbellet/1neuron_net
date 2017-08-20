# pylint: disable=C0103
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet
import matplotlib.pyplot as plt

# Load the data with numpy
dataset = np.loadtxt(
    open('./data/simple_data.csv', 'rb'),
    delimiter=',',
    skiprows=1
)
input_vectors = np.transpose(dataset[:, 0:1])
training_values = np.transpose(dataset[:, 1])

# Create a single neuron net
onn = OneNeuronNet(
    number_of_dendrons=1,
    activation_function=m.Identity,
    cost_function=m.L2Norm
)
onn.bias = 1
onn.weights = np.array([3])

# Train the neural net
epochs = 1000
cost_array = []
bias_array = []
weight_array = []

for i in range(epochs):
    cost = onn.minibatch_train(
        input_vectors,
        training_values,
        step=0.1
    )
    cost_array.append(cost)
    bias_array.append(onn.bias)
    weight_array.append(onn.weights[0])

# Plot the cost over the iterations
print onn.bias, onn.weights[0], cost_array
plt.subplot(311)
plt.yscale('log')
plt.plot(cost_array)
plt.subplot(312)
plt.plot(bias_array)
plt.subplot(313)
plt.plot(weight_array)
plt.show()
