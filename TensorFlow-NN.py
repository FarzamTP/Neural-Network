import tensorflow as tf
import scipy.io as io
import numpy as np
import os

# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 512
display_step = 10

# Network Parameters
n_hidden = 25  # 1st layer number of neurons
n_input = 400  # MNIST data input_features (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

X = tf.placeholder(dtype=tf.float64, shape=[None, 400], name='X')
Y = tf.placeholder(dtype=tf.float64, shape=[None, 10], name='y')

parameters = {
    'W1': tf.get_variable(dtype=tf.float64, name='W1', shape=(25, 400), initializer=tf.random_normal_initializer()),
    'b1': tf.get_variable(dtype=tf.float64, name='b1', shape=(1, 25), initializer=tf.zeros_initializer()),
    'W2': tf.get_variable(dtype=tf.float64, name='W2', shape=(10, 25), initializer=tf.random_normal_initializer()),
    'b2': tf.get_variable(dtype=tf.float64, name='b2', shape=(1, 10), initializer=tf.zeros_initializer())
}


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
    return y_new


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.matmul(x, tf.transpose(parameters['W1'])) + parameters['b1']
    rel1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    out_layer = tf.matmul(rel1, tf.transpose(parameters['W2'])) + parameters['b2']
    return out_layer


def predict(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.matmul(x, tf.transpose(parameters['W1'])) + parameters['b1']
    rel1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer2 = tf.matmul(rel1, tf.transpose(parameters['W2'])) + parameters['b2']
    out_layer = tf.sigmoid(layer2)
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
with tf.name_scope('Loss'):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss'))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

tf.summary.scalar('Loss', loss_op)

saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    data = io.loadmat('dataset/data.mat')

    org_X = data['X']
    org_y = data['y']

    print("org_X shape", org_X.shape)
    print("org_y shape", org_y.shape)

    if os.listdir('./model').__len__() != 0:
        print("[NOTE] Loading model...")
        saver.restore(sess, './model/TensorFlow/model.ckpt')
        print("[NOTE] Model restored successfully!")
    else:
        # Initializing the session
        sess.run(init)
        # Training cycle
        file_writer = tf.summary.FileWriter('./logs', sess.graph)
        file_writer.add_graph(tf.get_default_graph())

        total_batch = int(org_y.shape[0] / batch_size)

        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = next_batch(num=batch_size, data=org_X, labels=one_hot(org_y))
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], {X: batch_x, Y: batch_y})

                # Compute average loss
                avg_cost += c / batch_size
            # Display logs per epoch s1tep
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        saver.save(sess, './model/TensorFlow/model.ckpt')
        print("Optimization Finished!")

    print("Predicting sample from train set!")
    random_index = np.random.randint(0, 5000)
    sample_x = org_X[random_index]
    logits = np.argmax(predict(np.resize(sample_x, (1, 400))).eval()[0]) + 1
    print("Predicted X:", logits)
    print("True X:", org_y[random_index][0])
    print("Prediction finished!")

    p = []
    logits = predict(org_X)
    for i in logits.eval():
        p.append(np.argmax(i) + 1)
    p = np.asarray(p)
    c = 0
    for i in range(org_y.shape[0]):
        if p[i] == org_y[i][0]:
            c += 1
    print("Accuracy: %.6f" % (c / org_y.shape[0]))