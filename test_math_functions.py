"""Tests for the math_functions module"""
import unittest
import math_functions as m

class TestMathFunctions(unittest.TestCase):
    """Tests for the math_functions module"""

    def setUp(self):
        pass

    def test_sigmoid(self):
        self.assertEqual(m.Sigmoid.calc(0), 0.5)
        self.assertEqual(m.Sigmoid.diff(0), 0.25)
        self.assertAlmostEqual(m.Sigmoid.calc(5), 0.993307149)
        self.assertAlmostEqual(m.Sigmoid.calc(-5), 1 - m.Sigmoid.calc(5))
        self.assertAlmostEqual(m.Sigmoid.diff(5), 0.00664805)
        self.assertAlmostEqual(m.Sigmoid.diff(-5), 0.00664805)
        self.assertAlmostEqual(m.Sigmoid.calc(1), 0.73105857)
        self.assertAlmostEqual(m.Sigmoid.calc(9), 0.99987660)
        self.assertAlmostEqual(m.Sigmoid.diff(9), 0.00012337)

    def test_identity(self):
        self.assertEqual(m.Identity.calc(4.21), 4.21)
        self.assertEqual(m.Identity.diff(-100), 1)

    def test_l2_norm(self):
        self.assertEqual(m.L2Norm.calc(0, 1), 0)
        self.assertEqual(m.L2Norm.calc(3, 1), 2)
        self.assertAlmostEqual(m.L2Norm.calc(1, 4.4), 5.78)
        self.assertAlmostEqual(m.L2Norm.calc(-1, -4.4), 5.78)
        self.assertEqual(m.L2Norm.diff(9.542, 3), 6.542)

if __name__ == '__main__':
    unittest.main()
