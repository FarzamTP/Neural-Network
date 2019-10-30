import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

# FIXME: IT IS BROKEN!
class NeuralNetwork:
    def __init__(self, theta1, theta2, lr=0.01, num_iter=1000, Lambda=0, add_bias=True, verbose=False):
        self.theta1 = theta1
        self.theta2 = theta2

        self.lr = lr
        self.num_iter = num_iter
        self.Lambda = Lambda

        self.verbose = verbose
        self.add_bias = add_bias

    def __Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def SigmoidGradient(self, x):
        return self.__Sigmoid(x) * (1 - self.__Sigmoid(x))

    def __numericalGradient(self, theta1, theta2, eps):
        return (self.compute_cost(theta1 + eps, theta2 + eps) - self.compute_cost(theta1 - eps, theta2 - eps)) / 2 * eps

    def one_hot(self, y):
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
        return np.asarray(self.one_hot(predict))

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
        return h, Z2, A2, A1

    def compute_cost(self, X, y, theta1, theta2, Lambda):
        h, _, _, _ = self.__model(X)
        m = y.shape[0]
        if self.add_bias:
            Jreg = (Lambda / (2 * m)) * (sum(sum(theta1[:, 1:] ** 2)) + sum(sum(theta2[:, 1:] ** 2)))
        else:
            Jreg = (Lambda / (2 * m)) * (sum(sum(theta1 ** 2)) + sum(sum(theta2 ** 2)))
        return - 1 / m * sum(sum(np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h)))) + Jreg

    def train(self, X, y):
        print("X shape:", X.shape)
        # Reshapes y from (5000, 1) to (5000, 10)
        y = self.one_hot(y)
        print("y shape:", y.shape)
        cost_history = []
        m = y.shape[0]
        result = opt.minimize(fun=self.compute_cost(X, y, theta1=self.theta1, theta2=self.theta2, Lambda=self.Lambda),
                              x0=self.theta1, method='TNC')
        print(result)
        """
        for i in range(self.num_iter):
            h, Z2, A2, A1 = self.__model(X)
            # last layer error
            delta3 = h - y
            delta2 = np.multiply(np.matmul(delta3, self.theta2[:, 1:]), self.SigmoidGradient(Z2))
            Delta2 = np.dot(delta3.T, A2)
            Delta1 = np.dot(delta2.T, A1)

            cost = self.compute_cost(X=X, y=y, theta1=self.theta1, theta2=self.theta2, Lambda=self.Lambda)
            # print("Delta1:", Delta1)
            # print("Delta2:", Delta2)
            # print("NumericalGradient:", self.__numericalGradient(self.theta1, self.theta2, eps=0.02))
            cost_history.append(cost)

            print("Training batch %s:\nCost: %s" % (i + 1, cost))

            theta1_grad = 1 / m * Delta1 + self.Lambda / m * (self.theta1 ** 2)
            theta2_grad = 1 / m * Delta2 + self.Lambda / m * (self.theta2 ** 2)

            self.theta1 -= self.lr * theta1_grad
            self.theta2 -= self.lr * theta2_grad
        """
        return cost_history

    def predict(self, X):
        h, _, _, _ = self.__model(X)
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

    def plot_cost(self, cost_history):
        plt.plot(cost_history)
        plt.show()
