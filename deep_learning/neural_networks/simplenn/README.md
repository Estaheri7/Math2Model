# Simple Neural Network

This repository contains a simple implementation of a neural network from scratch using Python and NumPy. The implementation includes various layers and loss functions, and demonstrates how to train a multi-layer perceptron (MLP) using these components.

## Layers

### Linear Layer
The `Linear` layer performs a linear transformation on the input data. It includes methods for forward and backward propagation, as well as weight updates using the Adam optimization algorithm.

### Sigmoid Layer
The `Sigmoid` layer applies the sigmoid activation function to the input data. It includes methods for forward and backward propagation.

### ReLU Layer
The `ReLU` layer applies the Rectified Linear Unit (ReLU) activation function to the input data. It includes methods for forward and backward propagation.

### Softmax Layer
The `Softmax` layer applies the softmax activation function to the input data, which is commonly used for classification tasks. It includes methods for forward and backward propagation.

## Loss Functions

### Mean Squared Error (MSE)
The `MSE` class calculates the mean squared error between the predicted and true values. It includes methods for forward and backward propagation.

### Cross Entropy
The `CrossEntropy` class calculates the cross-entropy loss between the predicted and true values. It includes methods for forward and backward propagation.

## Multi Layer Perceptron (MLP)
The `MLP` class represents a multi-layer perceptron, which is a type of neural network. It includes methods for forward and backward propagation, weight updates, training, and prediction.

### Training Algorithm
The training algorithm used in this implementation is based on mini-batch gradient descent with the Adam optimization algorithm. The steps involved are:
1. Shuffle the training data.
2. For each epoch:
    - Select a random mini-batch of data.
    - Perform forward propagation to compute the predictions.
    - Compute the loss using the specified loss function.
    - Perform backward propagation to compute the gradients.
    - Update the weights using the Adam optimization algorithm.

## Usage
To use this implementation, you can create an instance of the `MLP` class with the desired layers and loss function, and then call the `train` method with your training data.
