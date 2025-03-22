"""Feed forward neural network implemented from scratch. 
Follows the tutorial from http://neuralnetworksanddeeplearning.com"""

import random
import numpy as np

class NeuralNetwork():
    
    
    def __init__(self, shape):
        
        """Shape of the neural network, defined as an array of layer sizes. 
        For example, a neural network with five input variables, a hidden layer with three neurons, and two output nodes would be: [5,3,2]"""
        self.shape = shape
        self.layer_count = len(self.shape)

        """Initialises random biases using normal distribution. 
        The bias for neuron j in layer m is located at self.biases[m][j]"""
        self.biases = [np.random.randn(y, 1) for y in shape[1:]]

        """Initialises random weights using normal distribution.
        The weight between neuron j in layer m and neuron k in layer m-1 is located at self.weights[m][j][k]"""
        self.weights = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]
        
        return

    def feedforward(self, input, activation_function):
        """Feeds an input through the neural network to obtain a predicted output. 
        Input shape must be (self.shape[0],1).
        Activation function can be relu or sigmoid.
        Returns a shape (self.shape[-1],1) output vector"""
        # Validate activation function
        if activation_function not in ["relu","sigmoid"]:
            raise ValueError("Activation function must be 'relu' or 'sigmoid'")
        
        # Feed forward input through each layer of the neural network
        activation = input
        for bias, weight in zip(self.biases, self.weights):
            if activation_function == "relu":
                activation = relu(np.dot(weight, activation) + bias)
            elif activation_function == "sigmoid":
                activation = sigmoid(np.dot(weight, activation) + bias)
        
        # Return the output activation
        return activation

    def sgd():
        """TO-DO: Trains the neural network using the stochastic gradient descent algorithm"""
        raise NotImplementedError()
    
    def backprop():
        """TO-DO: Uses the backpropagation algorithm to calculate """
        raise NotImplementedError()
    
###ACTIVATION FUNCTIONS###
def relu(input):
    """Relu function"""
    return max(0, input)

def sigmoid(input):
    """Sigmoid function"""
    return 1.0/(1.0+np.exp(-input))
