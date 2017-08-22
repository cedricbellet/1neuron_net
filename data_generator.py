"""Generates simple datasets on which to evaluate the single neuron."""
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import random
import math
import csv

class DatasetModel(object):
    """A model able to generate values"""

    def __init__(self, name, model_function, number_of_variables,
                 low=-10, high=10):
        """Single neuron net initialization.

        Args:
            model_function: a function that takes number_of_variables variables
            and returns a real output
            number_of_variables: the number of vars needed to use the model
            function
            low, high: optional range for the variables fed to the model
            function
        """
        self.model_function = model_function
        self.number_of_variables = number_of_variables
        self.low = low
        self.high = high
        self.name = name

    def generate_random_variables(self):
        if self.number_of_variables == 1:
            return random.uniform(self.low, self.high)
        else:
            variable_tuple = tuple(random.uniform(self.low, self.high)
                                   for i in range(self.number_of_variables))
            return variable_tuple

    def generate_one_value(self, noise=0):
        variable_tuple = self.generate_random_variables()
        if random.uniform(0, 1) < noise:
            output = self.model_function(variable_tuple) * random.uniform(-0.5,
                                                                          0.5)
        else:
            output = self.model_function(variable_tuple)
        return variable_tuple, output

    def generate_many_values(self, number_of_records=10, noise=0):
        records = []
        for _ in range(number_of_records):
            record = self.generate_one_value(noise=noise)
            if self.number_of_variables == 1:
                records.append([record[0], record[1]])
            else:
                records.append(list(record[0]) + [record[1]])
        return records

    def generate_csv(self, number_of_records=100, noise=0.05):
        records = self.generate_many_values(
            number_of_records=number_of_records,
            noise=noise)
        with open('./data/'+self.name+'.csv', 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(records)


LINEAR_MODEL = DatasetModel(
    name='linear',
    model_function=lambda (x): 5*x + 4,
    number_of_variables=1)

MULTLINEAR_MODEL = DatasetModel(
    name='multilinear',
    model_function=lambda (x, y, z): 5*x - 4*y + z + 4,
    number_of_variables=3)

POLYNOMIAL_MODEL_1 = DatasetModel(
    name='polynomial',
    model_function=lambda (x): x**3 - 0.732 * x**2 + 4*x + 10,
    number_of_variables=1
)

POLYNOMIAL_MODEL_2 = DatasetModel(
    name='polynomial_multivariate',
    model_function=lambda (x, y): y**2 - 0.732 * x**2 + 4*x + 2*x*y - 4.2,
    number_of_variables=2
)

SINUSOID_MODEL = DatasetModel(
    name='sinusoid',
    model_function=lambda (x): math.sin(x / 5),
    number_of_variables=1
)

OUTSIDE_CIRCLE_MODEL = DatasetModel(
    name='outside_circle',
    model_function=lambda (x, y): 1 if x**2 + y**2 > 5 else 0,
    number_of_variables=2
)

SUM_GREATER_THAN_X_MODEL = DatasetModel(
    name='outside_circle',
    model_function=lambda (x, y): 1 if 3*x - y > 0 else 0,
    number_of_variables=2
)

LINEAR_MODEL.generate_csv()
MULTLINEAR_MODEL.generate_csv()
POLYNOMIAL_MODEL_1.generate_csv()
POLYNOMIAL_MODEL_2.generate_csv()
SINUSOID_MODEL.generate_csv()
OUTSIDE_CIRCLE_MODEL.generate_csv()
SUM_GREATER_THAN_X_MODEL.generate_csv()
