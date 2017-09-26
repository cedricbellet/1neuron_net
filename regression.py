"""Regressions using the ONN."""
# pylint: disable=C0103
from __future__ import division
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet
from data_increase_order import increase_dataset_order
import data_visualizer as viz
import matplotlib.pyplot as plt

# Load the data with numpy
dataset = np.loadtxt(
    open('/Users/cedricbellet/lab/one_neuron_net/data/outside_circle.csv', 'rb'),
    delimiter=',',
    skiprows=1
)

# Set the parameters
NUMBER_OF_VARIABLES = 2
REGRESSION_TYPE = 'logistic'
ACTIVATION_FUNCTION = m.Sigmoid
COST_FUNCTION = m.CrossEntropy
EPOCHS = 6000
STEP = 12
DATASET_ORDER = 5
NUMBER_OF_VARIABLES_ADJ = NUMBER_OF_VARIABLES * DATASET_ORDER

# Augment the dataset's order
input_dataset = increase_dataset_order(
    dataset[:, 0:NUMBER_OF_VARIABLES],
    DATASET_ORDER)

# Extract the input and training vectors
input_vectors = np.transpose(input_dataset)
training_values = np.transpose(dataset[:, NUMBER_OF_VARIABLES])

# Create a single neuron net
onn = OneNeuronNet(
    number_of_dendrons=NUMBER_OF_VARIABLES_ADJ,
    activation_function=ACTIVATION_FUNCTION,
    cost_function=COST_FUNCTION
)

# Train the neural net
cost_array = []
bias_array = []
weight_array = [[] for i in range(NUMBER_OF_VARIABLES_ADJ)]

for i in range(EPOCHS):
    cost = onn.minibatch_train(
        input_vectors,
        training_values,
        step=STEP
    )
    cost_array.append(cost)
    bias_array.append(onn.bias)
    for j in range(NUMBER_OF_VARIABLES_ADJ):
        weight_array[j].append(onn.weights[j])

# Print the cost, bias, and weights
print onn.bias, onn.weights

# Plot the cost, bias, and weights
# base_subplot_number = 200 + 10 + NUMBER_OF_VARIABLES_ADJ * 100
# plt.subplot(base_subplot_number + 1)
# plt.yscale('log')
# plt.plot(cost_array)
# plt.subplot(base_subplot_number + 2)
# plt.plot(bias_array)
# for i in range(NUMBER_OF_VARIABLES_ADJ):
#     subplot_number = base_subplot_number + 3 + i * 1
#     plt.subplot(subplot_number)
#     plt.plot(weight_array[i])
# plt.show()

# Visualize the dataset
if REGRESSION_TYPE == 'linear' and NUMBER_OF_VARIABLES == 1:
    viz.visualize_linear_regression(dataset, onn.bias, onn.weights,
                                    DATASET_ORDER)
elif REGRESSION_TYPE == 'logistic' and NUMBER_OF_VARIABLES == 2:
    viz.visualize_logistic_regression(dataset, onn.bias, onn.weights,
                                      DATASET_ORDER)
