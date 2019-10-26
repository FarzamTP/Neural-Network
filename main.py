import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
from NeuralNetwork import NeuralNetwork as nn

print()
print("Retrieving data from file...")
# Gets data from data.mat file.
data = io.loadmat('dataset/data.mat')
print("Data loaded successfully!")

print("Loading pre-trained weights...")
# Gets pre-trained weights from weights.mat
weights = io.loadmat('weights/weights.mat')
print("Weights loaded successfully!")
print()

X = data['X']
y = data['y']

print("X shape:", X.shape)
print("y shape:", y.shape)

theta1 = weights['Theta1']
theta2 = weights['Theta2']

print("Theta1 shape:", weights['Theta1'].shape)
print("Theta2 shape:", weights['Theta2'].shape)
print()

model = nn(X=X, y=y, theta1=theta1, theta2=theta2, Lambda=0, add_bias=True, verbose=True)

cost = model.compute_cost()

print("Model biased: {} Lambda {}: cost: {}".format(model.add_bias, model.Lambda, cost))

model2 = nn(X=X, y=y, theta1=theta1[:, 1:], theta2=theta2[:, 1:], Lambda=1, add_bias=False, verbose=True)

cost = model2.compute_cost()

print("Model biased: {} Lambda {}: cost: {}".format(model2.add_bias, model2.Lambda, cost))
