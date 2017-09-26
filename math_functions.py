"""Classes for activation and cost functions."""
# -*- coding: utf-8 -*-
from __future__ import division
import math

"""Activation functions

Activation functions only take one argument, the weighted input,
and return a real number
"""

class Sigmoid(object):
    """Implements the sigmoid activation function"""
    def __init__(self):
        pass

    @classmethod
    def calc(cls, val):
        return 1 / (1 + math.exp(-val))

    @classmethod
    def diff(cls, val):
        return math.exp(-val) / (1 + math.exp(-val)) ** 2


class Identity(object):
    """The Identity function"""
    def __init__(self):
        pass

    @classmethod
    def calc(cls, val):
        return val

    @classmethod
    def diff(cls, _val):
        return 1


"""Cost functions

Cost functions take two arguments, the activation, and the expected activation,
and return the cost associated with the activation.

The diff method is implicitely and differential on the activation argument.
"""

class SquaredError(object):
    """Implements the L2 norm or Euclidian distance"""
    def __init__(self):
        pass

    @classmethod
    def calc(cls, val1, val2):
        return ((val1 - val2) ** 2) / 2

    @classmethod
    def diff(cls, val1, val2):
        return val1 - val2

class CrossEntropy(object):
    """Implements the cross entropy loss function

    val2 is assumed to be 0 or 1 always.
    """
    def __init__(self):
        pass

    @classmethod
    def calc(cls, val1, val2):
        if val1 == val2:
            return 0
        if val1 >= 1 or val1 <= 0:
            return 1e3
        elif val2 == 0:
            return -math.log(1 - val1) - val1
        elif val2 == 1:
            return -math.log(val1) + val1 - 1
        else:
            return -(1 - val2) * (math.log(1 - val1) + val1) - val2 * \
            (math.log(val1) - val1 + 1)

    @classmethod
    def diff(cls, val1, val2):
        if val1 == val2:
            return 0
        if val2 == 0 and val1 >= 1:
            return -1e3
        elif val2 == 1 and val1 <= 0:
            return 1e3
        elif val2 == 0:
            return 1/(1 - val1) - 1
        elif val2 == 1:
            return -1/val1 + 1
        else:
            return (1 - val2) * (1/(1 - val1) - 1) + val2 * (-1/val1 + 1)
