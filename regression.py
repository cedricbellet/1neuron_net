"""Regressions using the ONN."""
# pylint: disable=C0103
from __future__ import division
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet
from data_increase_order import increase_dataset_order
import data_visualizer as viz
# import matplotlib.pyplot as plt


# Main function
def main(
        dataset_name, number_of_variables, regression_type, activation_function,
        cost_function, epochs, step, dataset_order=1):
    """Main function drawing the regressions's output.

    Args:
        dataset_name: the name of the dataset to be used
        number_of_variables (int): number of predicting variables in the dataset
        regression_type: one of 'linear' or 'logistic', which dictates the
            visualization function used at the end of the training
        activation_function: a function from the math_functions module
        cost_function: a function from the math_functions module
        epochs (int): the number of iterations over the full dataset
        step (float): the backpropagation step; 1e-3 to 100 based on the dataset
        dataset_order (opt int): the order of the dataset if it needs to be increased
    """
    # Load the data with numpy
    dataset = np.loadtxt(
        open('/Users/cedricbellet/lab/one_neuron_net/data/'+dataset_name+'.csv', 'rb'),
        delimiter=',',
        skiprows=1
    )

    # Set the parameters
    NUMBER_OF_VARIABLES = number_of_variables
    REGRESSION_TYPE = regression_type
    ACTIVATION_FUNCTION = activation_function
    COST_FUNCTION = cost_function
    EPOCHS = epochs
    STEP = step
    DATASET_ORDER = dataset_order
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
    # print onn.bias, onn.weights

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

if __name__ == '__main__':
    # Logistic regressions
    main('outside_circle', 2, 'logistic', m.Sigmoid, m.CrossEntropy, 500, 10, 5) # Good
    main('logistic_line', 2, 'logistic', m.Sigmoid, m.CrossEntropy, 500, 10, 5) # Good

    # Sinusoid
    # main('sinusoid', 1, 'linear', m.Identity, m.SquaredError, 500, 0.1, 5)
    # main('sinusoid', 1, 'linear', m.Identity, m.SquaredError, 3000, 1.5, 8)
    # main('sinusoid', 1, 'linear', m.Identity, m.SquaredError, 3000, 1.5, 12) # not better
    # main('sinusoid', 1, 'linear', m.Identity, m.SquaredError, 3000, 2, 8) # step too large
    main('sinusoid', 1, 'linear', m.Identity, m.SquaredError, 6000, 1.5, 8) # best

    # Multilinear
    # main('polynomial', 1, 'linear', m.Identity, m.SquaredError, 2000, 1) # linear regression, 1 param
    main('polynomial', 1, 'linear', m.Identity, m.SquaredError, 2000, 1, 5) # linear regression, 5 param
