"""A class for one-neuron nets."""
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np


class OneNeuronNet(object):
    """A neural network with a single neuron (but many incoming dendrons), for
    experiments with different activation functions, cost functions and training
    objectives."""

    def __init__(self, number_of_dendrons, activation_function, cost_function):
        """Single neuron net initialization.

        Args:
            number_of_dendrons (int): number of dendrons, excl. the bias dendron
            activation_function: a function from the math_functions module
            cost_function: a function from the math_functions module
            weights (optional): a numpy array of shape (number_of_dendrons) as
            type float
            bias (float): the initial value of the bias
        """
        self.number_of_dendrons = number_of_dendrons
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.weights = np.random.rand(number_of_dendrons)
        self.bias = np.random.sample()


    def forward_pass(self, input_vector):
        """Performs a single forward pass in the neural net.

        Params:
            input_vector: a single numpy array of shape (number_of_dendrons)
        """
        weighted_input = np.dot(self.weights, input_vector) + self.bias
        activation = self.activation_function.calc(weighted_input)
        return weighted_input, activation


    def backpropagation(self, input_vector, training_value):
        """Calculates the error and cost gradients for a single input_vector

        Params:
            input_vector: a single numpy array of shape (number_of_dendrons)
            training_value: the expected output for the input_vector, a real
            number
        """
        weighted_input, activation = self.forward_pass(input_vector)
        cost = self.cost_function.calc(activation, training_value)
        delta = self.cost_function.diff(activation, training_value) * \
                 self.activation_function.diff(weighted_input)
        bias_adjustment = delta
        weight_adjustments = delta * input_vector

        return cost, bias_adjustment, weight_adjustments


    def minibatch_train(self, input_vectors, training_values, step=0.001):
        """Changes the neuron's parameter based on a batch of training examples.

        Params:
            input_vectors: a numpy array of shape (number_of_dendrons, m) where
            m is the size of the minibatch
            training_values: a numpy array of shape (m)
        """
        self.weights = self.weights.astype(float)
        self.bias = float(self.bias)

        minibatch_cost, minibatch_bias_adj = (0., 0.)
        minibatch_weight_adjs = np.zeros(self.number_of_dendrons)
        size = training_values.size

        for i in range(size):
            cost, bias_adj, weight_ajds = self.backpropagation(
                input_vectors[:, i], training_values[i]
            )
            minibatch_cost += 1./size * cost
            minibatch_bias_adj += 1./size * bias_adj
            minibatch_weight_adjs += 1./size * weight_ajds

        self.bias -= step *  minibatch_bias_adj
        self.weights -= step * minibatch_weight_adjs

        return minibatch_cost
