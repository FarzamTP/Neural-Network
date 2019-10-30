# Neural Network
This project contains two phases:
* Pure self implemented gradient descent based neural network
* non-framework TensorFlow implementation

# Self Implemented Gradient Descent
 At phase one, I tried to implement the whole forward-propagation and
 back-propagation on my own and make progress trough the **MNIST** 
 data set.
 For this phase I implemented `main.py` and fed algorithm with given data set
 according to [Deeplearning.ai course on coursera](https://www.coursera.org/learn/neural-networks-deep-learning) presented by Prof.Andrew Ng
 on third week's exercise.

# Code features
The implemented `main.py` code has some features that may help you such:
* Data restored from `data.mat` file stores in [dataset](https://github.com/FarzamTP/Neural-Network-TensorFLow/tree/master/dataset) directory.
* Model will be saved after training in [model](https://github.com/FarzamTP/Neural-Network-TensorFLow/tree/master/model/Gradient%20Descent) directory as parametrized numpy arrays(`*.npy`).
* If a previously trained model exists in denoted directory, It will be loaded automatically
* Accuracy of given data will be computed and displayed at the end of training, or loading model.
* All methods needed for neural network algorithm is implemented inside `main.py`.

# Result (Phase one)
With given knowledge about implemented code, with default hyper parameters(like: number fo iterations, learning rate, ..) 
algorithm will perform as below:

# Accuracy and Cost
The algorithm performs well and will has accuracy of __**0.988%**__ over training-set.
![Cost figure]()