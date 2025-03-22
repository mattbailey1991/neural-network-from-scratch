"""Feed forward neural network implemented from scratch. 
Follows the tutorial from http://neuralnetworksanddeeplearning.com"""

import random
import numpy as np

class NeuralNetwork():
    
    
    def __init__(self, shape):
        
        """Shape of the neural network, defined as an array of layer sizes. 
        For example, a neural network with five x variables, a hidden layer with three neurons, and two output nodes would be: [5,3,2]"""
        self.shape = shape
        self.layer_count = len(self.shape)

        """Initialises random biases using normal distribution. 
        The bias for neuron j in layer m is located at self.biases[m][j]"""
        self.biases = [np.random.randn(y, 1) for y in shape[1:]]

        """Initialises random weights using normal distribution.
        The weight between neuron j in layer m and neuron k in layer m-1 is located at self.weights[m][j][k]"""
        self.weights = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]
        
        return

    def feedforward():
        """TO-DO: Feeds an input through the neural network to obtain a predicted output. Returns..."""
    
    def sgd():
        """TO-DO: Trains the neural network using the stochastic gradient descent algorithm"""
        raise NotImplementedError()
    
    def backprop():
        """TO-DO: Uses the backpropagation algorithm to calculate """
        raise NotImplementedError()
    