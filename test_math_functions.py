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

    def test_squared_error(self):
        self.assertEqual(m.SquaredError.calc(0, 1), 0.5)
        self.assertEqual(m.SquaredError.calc(3, 1), 2)
        self.assertAlmostEqual(m.SquaredError.calc(1, 4.4), 5.78)
        self.assertAlmostEqual(m.SquaredError.calc(-1, -4.4), 5.78)
        self.assertEqual(m.SquaredError.diff(9.542, 3), 6.542)

    def test_cross_entropy(self):
        """Test the cross entropy across edge cases"""
        self.assertEqual(m.CrossEntropy.calc(2, 1), 1e3)
        self.assertEqual(m.CrossEntropy.calc(-2, 0), 1e3)
        self.assertAlmostEqual(m.CrossEntropy.calc(0.5, 1), 0.19314718)
        self.assertAlmostEqual(m.CrossEntropy.calc(0.5, 0), 0.19314718)
        self.assertAlmostEqual(m.CrossEntropy.calc(1, 1), 0)
        self.assertAlmostEqual(m.CrossEntropy.calc(0, 1), 1e3)
        self.assertAlmostEqual(m.CrossEntropy.calc(0.2, 0), 0.023143551)
        self.assertRaises(m.CrossEntropy.calc(0.2, 0.2))
        self.assertAlmostEqual(m.CrossEntropy.diff(0.2, 0), 0.25)
        self.assertAlmostEqual(m.CrossEntropy.diff(0.2, 1), -4)
        self.assertAlmostEqual(m.CrossEntropy.diff(0.9, 1), -0.1111111)
        self.assertAlmostEqual(m.CrossEntropy.diff(0.999, 1), -0.0010010010)


if __name__ == '__main__':
    unittest.main()
