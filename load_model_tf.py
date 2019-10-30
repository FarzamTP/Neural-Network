import tensorflow as tf
from sklearn import preprocessing
import cv2
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np

data = io.loadmat('dataset/data.mat')

org_X = data['X']
org_y = data['y']

print("org_X shape", org_X.shape)
print("org_y shape", org_y.shape)

sample_x = org_X[0]
print(sample_x)

img_name = '0.jpg'
img_read = cv2.imread('images/%s' % img_name, 0)
print(img_read.shape)
plt.imshow(img_read)
plt.show()

img = np.resize(img_read, (1, 400))
norm = preprocessing.normalize(img)
print(norm)

"""
parameters = {
    'W1': tf.get_variable(dtype=tf.float64, name='W1', shape=(25, 400), initializer=tf.random_normal_initializer()),
    'b1': tf.get_variable(dtype=tf.float64, name='b1', shape=(1, 25), initializer=tf.zeros_initializer()),
    'W2': tf.get_variable(dtype=tf.float64, name='W2', shape=(10, 25), initializer=tf.random_normal_initializer()),
    'b2': tf.get_variable(dtype=tf.float64, name='b2', shape=(1, 10), initializer=tf.zeros_initializer())
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.matmul(x, tf.transpose(parameters['W1'])) + parameters['b1']
    rel1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.matmul(rel1, tf.transpose(parameters['W2'])) + parameters['b2']
    out_layer = tf.sigmoid(layer_2)
    return out_layer


saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    for i in range(10):
        img_name = str(i) + '.jpg'
        img = np.resize(cv2.imread('images/%s' % img_name), (1, 400)) / 256
        # img_1D = np.reshape(img, (400, 1))
        # Construct model
        logits = multilayer_perceptron(img)
        print(np.argmax(logits.eval()[0]) + 1)
"""