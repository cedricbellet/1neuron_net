"""Increase the order of a dataset."""
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np

def increase_dataset_order(dataset, order=2):
    """Increase the order of a dataset

    For example, if a dataset contains the variables x and y, a dataset of
    higher order will be a dataset containing x, y, x^2, y^2

    Params:
        order: an integer describing the highest exponent
    """
    dataset_width = dataset.shape[1]

    for k in range(1, order):
        for i in range(dataset_width):
            new_col = dataset[:, i] ** (k + 1)
            dataset = np.c_[dataset, new_col]

    return dataset
