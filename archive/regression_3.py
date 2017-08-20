# pylint: disable=C0103
import pandas as pd
import numpy as np

# Load the data with pandas
df = pd.read_csv('./data/winequality-red.csv', sep=';')
print df[['alcohol', 'quality']]


# Load the data with numpy
dataset = np.loadtxt(
    open('./data/winequality-red.csv', 'rb'),
    delimiter=';',
    skiprows=1
)
input_vectors = np.transpose(dataset[:, 10:11])
training_values = np.transpose(dataset[:, 11])

print input_vectors, training_values
