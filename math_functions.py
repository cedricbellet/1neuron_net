"""Classes for activation and cost functions."""
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

class L2Norm(object):
    """Implements the L2 norm or Euclidian distance"""
    def __init__(self):
        pass

    @classmethod
    def calc(cls, val1, val2):
        return ((val1 - val2) ** 2) / 2

    @classmethod
    def diff(cls, val1, val2):
        return val1 - val2
