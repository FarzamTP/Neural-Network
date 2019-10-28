import warnings
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import os


def one_hot(y):
    y_new = []
    for i in y:
        if i == [10] or i == [0]:
            y_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        elif i == [9]:
            y_new.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif i == [8]:
            y_new.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif i == [7]:
            y_new.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif i == [6]:
            y_new.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif i == [5]:
            y_new.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif i == [4]:
            y_new.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif i == [3]:
            y_new.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif i == [2]:
            y_new.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif i == [1]:
            y_new.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return np.asarray(y_new)


print()
print("[STEP1] Retrieving data from file...")
# Gets data from data.mat file.
data = io.loadmat('dataset/data.mat')
print("Data loaded successfully!")
print()

print("[STEP2] Loading pre-trained weights...")
# Gets pre-trained weights from weights.mat
weights = io.loadmat('weights/weights.mat')
print("Weights loaded successfully!")
print()

org_X = data['X']
org_y = data['y']

X = org_X.T
y = one_hot(org_y).T

print("X shape:", X.shape)
print("y shape:", y.shape)

theta1 = weights['Theta1']
theta2 = weights['Theta2']

print("Theta1 shape:", weights['Theta1'].shape)
print("Theta2 shape:", weights['Theta2'].shape)
print()

# declaring a list in size of selected data set containing 0 to size of data set.
indices = list(range(len(X)))

# shuffling indices array.
np.random.shuffle(indices)

# number of data samples we want to train
threshold = 4500


# splitting train data arrays according to indices array.
# x_train = X[indices[:threshold]]
# y_train = y[indices[:threshold]]

# splitting test data arrays according to indices array.
# x_test = X[indices[threshold:]]
# y_test = y[indices[threshold:]]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def layer_sizes(X, y):
    n_x = X.shape[0]
    n_h1 = 25
    n_y = y.shape[0]
    return n_x, n_h1, n_y


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # print("W1 shape:", W1.shape)
    # print("b1 shape:", b1.shape)
    # print("W2 shape:", W2.shape)
    # print("b2 shape:", b2.shape)

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def compute_cost(A2, y, parameters):
    m = y.shape[1]
    logprobs = np.multiply(y, np.log(A2)) + np.multiply((1 - y), np.log(1 - A2))
    cost = np.sum(logprobs) * -1 / m
    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))
    return cost


def back_propagation(parameters, cache, X, y):
    m = X.shape[0]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - y
    dW2 = 1 / m * np.matmul(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.matmul(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.matmul(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=0.01):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, y, n_h, num_iterations=10000, print_cost=False):
    n_x = layer_sizes(X, y)[0]
    n_y = layer_sizes(X, y)[2]

    cost_history = []

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, y, parameters)
        grads = back_propagation(parameters, cache, X, y)
        parameters = update_parameters(parameters, grads)

        cost_history.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters, cost_history


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.argmax(A2, axis=0)
    return np.reshape(predictions, (predictions.shape[0], 1))


def plot_cost(cost_history):
    plt.plot(cost_history)
    plt.show()


parameters = {
    "W1": None,
    "b1": None,
    "W2": None,
    "b2": None
}

for param in parameters:
    if os.path.exists('./model/%s.npy' % param):
        parameters[param] = np.load('./model/%s.npy' % param)
        print("[Note]: %s loaded successfully..." % param)
    else:
        warnings.warn("[Error]: Couldn't import %s!" % param)
        warnings.warn("Training model...")
        break

if os.path.exists('./model/costs.npy'):
    cost_history = np.load('./model/costs.npy')
    print("[Note]: Costs loaded successfully...")
else:
    cost_history = None

if parameters["W1"] is None or parameters["b1"] is None or parameters["W2"] is None or parameters["b2"] is None or cost_history is None:
    parameters, cost_history = nn_model(X, y, n_h=25, num_iterations=10000, print_cost=True)

    print("Saving weights...")
    np.save('./model/W1', parameters["W1"])
    np.save('./model/W2', parameters["W2"])
    np.save('./model/b1', parameters["b1"])
    np.save('./model/b2', parameters["b2"])
    print("Weights saved successfully...")
    np.save('./model/costs', cost_history)
    print("Cost history saved successfully.")

print()

pred = predict(parameters, X)

accuracy = sum((pred + 1) == org_y) / org_y.shape[0]

plot_cost(cost_history)

print("Accuracy over train set: %.3f" % accuracy)

