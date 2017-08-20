'''A regression on the wine dataset using the onn network'''
# pylint: disable=C0103
import math
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet
import matplotlib.pyplot as plt

# Load the data
dataset = np.loadtxt(
    open('./data/simple_multi_data.csv', 'rb'),
    delimiter=',',
    skiprows=1
)
input_vectors = np.transpose(dataset[:, 0:3])
training_values = np.transpose(dataset[:, 3])

# Create a single neuron net
onn = OneNeuronNet(
    number_of_dendrons=3,
    activation_function=m.Identity,
    cost_function=m.L2Norm
)

# Train the neural net
epochs = 60
minibatch_size = 20
batches_per_epoch = int(math.floor(training_values.shape[0] / minibatch_size))+1
cost = []
weight0_array = []
weight1_array = []
weight2_array = []
bias_array = []

for i in range(epochs):
    np.random.shuffle(dataset)
    for j in range(batches_per_epoch):
        start = 10 * j
        finish = 10 * j + minibatch_size - 1
        minibatch_input_vectors = input_vectors[:, start:finish]
        minibatch_training_values = training_values[start:finish]
        minibatch_cost = onn.minibatch_train(
            minibatch_input_vectors,
            minibatch_training_values,
            step=0.01
        )
        cost.append(minibatch_cost)
        weight0_array.append(onn.weights[0])
        weight1_array.append(onn.weights[1])
        weight2_array.append(onn.weights[2])
        bias_array.append(onn.bias)

# print onn params
print onn.weights, onn.bias

# Plot the cost over the iterations
plt.subplot(511)
plt.yscale('log')
plt.plot(cost)
plt.subplot(512)
plt.plot(bias_array)
plt.subplot(513)
plt.plot(weight0_array)
plt.subplot(514)
plt.plot(weight1_array)
plt.subplot(515)
plt.plot(weight2_array)
plt.show()

# onn.weights = np.array([
#     0.041761, -0.2806674, 0.34856949, -0.00607055, 0.43358165,  0.00680184,
#     -0.00331988, 0.00937817, 0.15510989, 0.61165884, 0.34848919
# ])
# onn.bias = 0.805545546804

# [-2.16687367  3.88784345 -4.81950255] 10.4036784308
