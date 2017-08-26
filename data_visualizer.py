"""Generates simple datasets on which to evaluate the single neuron."""
# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def visualize_linear_regression(dataset, bias, weights, order, figure=2):
    """Plots a 2-dimensional dataset and a fit function"""
    plt.figure(figure)
    # Plot the dataset
    plt.scatter(dataset[:, 0], dataset[:, 1])
    # Calculate the fit
    x = np.linspace(-1.0, 1.0, 100)
    y = bias
    for i in range(order):
        y += weights[i] * x**(i+1)
    plt.plot(x, y)
    plt.show()

def visualize_logistic_regression(dataset, bias, weights, order, figure=2):
    """Plots a 2-dimensional dataset and a contour"""
    plt.figure(figure)
    # Plot the dataset
    plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2], cmap='gray')
    # Calculate the contour figure
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)

    Z = weights[0] * X + weights[1] * Y + bias
    if order > 1:
        for i in range(1, order):
            Z += weights[2*i] * X**(i+1) + weights[i*2 + 1] * Y**(i+1)
    # Plot the countour
    plt.contour(X, Y, Z, [0])
    plt.show()
