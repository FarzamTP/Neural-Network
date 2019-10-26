import numpy as np


class NeuralNetwork:
    def __init__(self, X, y, theta1, theta2, Lambda=0, add_bias=True, verbose=False):
        self.X = X
        self.y = y

        self.theta1 = theta1
        self.theta2 = theta2

        self.Lambda = Lambda

        self.verbose = verbose
        self.add_bias = add_bias

        # Add bias if True
        if self.add_bias:
            self.X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def __Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __one_hot(self, y):
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

    def __argmax_extractor(self, h):
        predict = []
        # extracts argmax of h and appends to predict
        for i in h:
            if np.argmax(i) == 0:
                predict.append([10])
            else:
                predict.append([np.argmax(i) + 1])
        return np.asarray(self.__one_hot(predict))

    def __model(self, X):
        # First layer node
        A1 = X
        # First layer values * layer1 theta
        Z2 = np.matmul(A1, self.theta1.T)
        # Second layer
        A2 = self.__Sigmoid(Z2)
        if self.add_bias:
            # Adds ones to A2
            A2 = np.concatenate((np.ones((A2.shape[0], 1)), A2), axis=1)
        Z3 = np.matmul(A2, self.theta2.T)
        # Last layer
        h = self.__Sigmoid(Z3)
        return h

    def compute_cost(self):
        h = self.__model(self.X)
        # Reshapes y from (5000, 1) to (5000, 10)
        y = self.__one_hot(self.y)
        m = y.shape[0]
        if self.add_bias:
            Jreg = (self.Lambda / (2 * m)) * (sum(sum(self.theta1[:, 1:] ** 2)) + sum(sum(self.theta2[:, 1:] ** 2)))
        else:
            Jreg = (self.Lambda / (2 * m)) * (sum(sum(self.theta1 ** 2)) + sum(sum(self.theta2 ** 2)))
        return - 1 / m * sum(sum(np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h)))) + Jreg

    def predict(self, X):
        h = self.__model(X)
        prediction = self.__argmax_extractor(h)
        return prediction

    def compute_accuracy(self, X, y):
        true_guesses = 0
        prediction = self.predict(X)
        m = y.shape[0]
        for i in range(len(y)):
            if sum(np.multiply(prediction[i], y[i])) != 0:
                true_guesses += 1
        return true_guesses / m