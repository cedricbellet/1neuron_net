"""Tests for the one_neuron_net module"""
import unittest
import numpy as np
import math_functions as m
from one_neuron_net import OneNeuronNet

class TestOneNeuronNet(unittest.TestCase):
    """Tests for the one_neuron_net module"""

    def setUp(self):
        pass

    def test_initialization(self):
        number_of_dendrons = 10
        onn = OneNeuronNet(number_of_dendrons, m.Sigmoid, m.SquaredError)
        self.assertEqual(onn.number_of_dendrons, number_of_dendrons)
        self.assertEqual(onn.activation_function, m.Sigmoid)
        self.assertEqual(onn.weights.shape, (number_of_dendrons,))
        self.assertTrue(0 <= onn.bias < 1)

    def test_forward_pass(self):
        """Test for the forward pass, under different settings"""
        number_of_dendrons = 2
        onn = OneNeuronNet(number_of_dendrons, m.Sigmoid, m.SquaredError)
        onn.bias = 0
        onn.weights = np.array([1, 0])

        input_vector = np.array([0.05, 2])
        self.assertAlmostEqual(onn.forward_pass(input_vector)[0], 0.05)
        self.assertAlmostEqual(
            onn.forward_pass(input_vector)[1], 0.51249739
        )

        onn.bias = -5
        onn.weights = np.array([1, 2])
        self.assertAlmostEqual(onn.forward_pass(input_vector)[0], -0.95)
        self.assertAlmostEqual(
            onn.forward_pass(input_vector)[1], 0.27888482
        )

    def test_backpropagation(self):
        """Tests for the backpropagation method."""
        number_of_dendrons = 2
        input_vector = np.array([2, 3])

        onn = OneNeuronNet(number_of_dendrons, m.Identity, m.SquaredError)
        onn.bias = 0.5
        onn.weights = np.array([1, 2])
        cost, bias_adj, weight_adjs = onn.backpropagation(input_vector, 9)
        self.assertAlmostEqual(cost, 0.125)
        self.assertAlmostEqual(bias_adj, -0.5)
        self.assertAlmostEqual(weight_adjs[0], -1)
        self.assertAlmostEqual(weight_adjs[1], -1.5)

        onn = OneNeuronNet(number_of_dendrons, m.Sigmoid, m.SquaredError)
        onn.bias = 1.
        onn.weights = np.array([1, 2])
        cost, bias_adj, weight_ajds = onn.backpropagation(input_vector, 0)
        self.assertAlmostEqual(bias_adj, 0.0001233641)
        self.assertAlmostEqual(weight_ajds[0], 0.0002467282)
        self.assertAlmostEqual(weight_ajds[1], 0.0002467282 * 3 / 2)

    def test_minibatch_train(self):
        """Tests for a minibatch training."""
        onn = OneNeuronNet(3, m.Identity, m.SquaredError)
        onn.bias = 1
        onn.weights = np.array([1, 2, 5])
        input_vectors = np.array([
            [2, 5, 8, 12, 3, 8, 12],
            [1, 7, 2, 7, 1, 67, 2],
            [4, 12, 24, 2, 2, 6, 10],
        ])
        training_values = np.array([5, 6, 2, 6, 10, 11, 7])
        minibatch_cost = onn.minibatch_train(input_vectors, training_values)
        self.assertAlmostEqual(minibatch_cost, 3848.4285714285716)
        minibatch_cost = onn.minibatch_train(input_vectors, training_values)
        self.assertAlmostEqual(minibatch_cost, 1184.0608488352768)
        minibatch_cost = onn.minibatch_train(input_vectors, training_values)
        self.assertAlmostEqual(minibatch_cost, 725.63867231797417)


if __name__ == '__main__':
    unittest.main()
