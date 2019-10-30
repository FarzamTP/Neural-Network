# Neural Network
This project contains two phases:
* Pure self implemented gradient descent based neural network
* non-framework TensorFlow implementation

# Self Implemented Gradient Descent (Phase One)
 At phase one, I tried to implement the whole forward-propagation and
 back-propagation on my own and make progress trough the **MNIST** 
 data set.
 For this phase I implemented `main.py` and fed algorithm with given data set
 according to [Deeplearning.ai course on coursera](https://www.coursera.org/learn/neural-networks-deep-learning) presented by Prof.Andrew Ng
 on third week's exercise.

# Code Features
The implemented `main.py` code has some features that may help you such:
* Data restored from `data.mat` file stores in [dataset](https://github.com/FarzamTP/Neural-Network-TensorFLow/tree/master/dataset) directory.
* Model will be saved after training in [model](https://github.com/FarzamTP/Neural-Network-TensorFLow/tree/master/model/Gradient%20Descent) directory as parametrized numpy arrays(`*.npy`).
* If a previously trained model exists in denoted directory, It will be loaded automatically
* Accuracy of given data will be computed and displayed at the end of training, or loading model.
* All methods needed for neural network algorithm is implemented inside `main.py`.

# Accuracy and Cost
The algorithm performs well and will has accuracy of __**0.988%**__ over training-set:
![Cost figure](https://github.com/FarzamTP/Neural-Network-TensorFLow/blob/master/figures/cost.png)

# Result (Phase one)
According to that this is a only one-hidden layer model with one input layer (also known as feature layer)
and a output layer, It is pretty simple neural network and for such structure its 
performance is dominant.

# TensorFlow Based Neural Network (Phase Two)
In second part I noticed that although I designed a simple model, It runs on CPU, So
It is a lot slower than processing on GPU.
So I started to impalement my model on file `TensorFlow-NN.py` with TensorFlow framework which is able to 
be ran on GPU.

# Code Features
`TensorFlow-NN.py` has some features as well as `main.py` such as:
* The cost function is optimized with Adam optimizer
* `TensorFlow.train.Saver` is user in the code which can be user for storing or loading model.
* Some very useful functions like `one_hot()`, `next_batch()` are implemented, which will make code simpler and more readable.
* Model is saved in [model/TensorFlow](https://github.com/FarzamTP/Neural-Network-TensorFLow/tree/master/model/TensorFlow) directory at end of every training.
* A sample from training-set is selected and predicted by model at end of training.
* TensorBoard is a lot useful here with TensorFlow to demonstrate graphs.

# Accuracy and Graph
The algorithm with default parameters will perform like **0.962%** over training set.
And this is our model's graph which can is downloaded from TensorBoard.
![Graph Figure](https://github.com/FarzamTP/Neural-Network-TensorFLow/blob/master/figures/graph.png)

# Conclusion
According to denoted details, in such scale data (about 5000 records of 400 feature images) 
Adam and Gradient Decent algorithm works all good, however Adam optimizer 
that is implemented in TensorFlow is a lot faster than Gradient Descent.

# Note
Here I uploaded pre-trained model for each algorithm for you in [model](https://github.com/FarzamTP/Neural-Network-TensorFLow/tree/master/model) director, Enjoy :)