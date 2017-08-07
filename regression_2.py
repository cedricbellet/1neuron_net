'''A regression on the wine dataset using the onn network'''
# pylint: disable=C0103
import math
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet
import matplotlib.pyplot as plt

# Load the data
dataset = np.loadtxt(
    open('./data/winequality-red.csv', 'rb'),
    delimiter=';',
    skiprows=1
)
input_vectors = np.transpose(dataset[:, 10:11])
training_values = np.transpose(dataset[:, 11])

# Create a single neuron net
onn = OneNeuronNet(
    number_of_dendrons=1,
    activation_function=m.Identity,
    cost_function=m.L2Norm
)
onn.bias = 2
onn.weights = np.array([0.3608])

# Train the neural net
epochs = 3000
minibatch_size = 1000
batches_per_epoch = int(math.floor(training_values.shape[0] / minibatch_size))+1
cost = []

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
            step=0.005
        )
        cost.append(minibatch_cost)

# Plot the cost over the iterations
plt.plot(cost)
plt.yscale('log')
plt.show()

print cost
print onn.weights, onn.bias

# [-0.1388386   0.19566395  0.62354179  0.33076839  0.31732113  0.02536196
#  -0.01404212  0.82769649 -0.06891764  0.52937966  0.45346027]

# [ 0.00811374  0.39881568  0.71427037  0.01110518  0.09796632  0.1905478
#  -0.10051772  0.85468529  0.30859389  0.52782882  0.45796365]
