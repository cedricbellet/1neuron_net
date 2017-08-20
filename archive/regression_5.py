# pylint: disable=C0103
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet
import matplotlib.pyplot as plt

# Load the data with numpy
dataset = np.loadtxt(
    open('./data/simple_logistic_data.csv', 'rb'),
    delimiter=',',
    skiprows=1
)
input_vectors = np.transpose(dataset[:, 0:2])
training_values = np.transpose(dataset[:, 2])

# Create a single neuron net
onn = OneNeuronNet(
    number_of_dendrons=2,
    activation_function=m.SoftRectifiedLinear,
    cost_function=m.CrossEntropy
)

# Train the neural net
epochs = 50000
cost_array = []
bias_array = []
weight0_array = []
weight1_array = []

for i in range(epochs):
    cost = onn.minibatch_train(
        input_vectors,
        training_values,
        step=0.02
    )
    cost_array.append(cost)
    bias_array.append(onn.bias)
    weight0_array.append(onn.weights[0])
    weight1_array.append(onn.weights[1])

# Plot the cost over the iterations
print onn.bias, onn.weights
plt.subplot(411)
plt.yscale('log')
plt.plot(cost_array)
plt.subplot(412)
plt.plot(bias_array)
plt.subplot(413)
plt.plot(weight0_array)
plt.subplot(414)
plt.plot(weight1_array)
plt.show()
