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

    def sgd(self, training_data, epochs, batch_size, eta):
        """Trains the neural network using the stochastic gradient descent algorithm
        Training data is an array of input and output vectors
        Epochs is the number of epochs to train the model
        Batch size is the mini batch size for each update
        Eta is the learning rate"""
        for i in range(epochs):
            # Shuffle the training data to give a random sample
            random.shuffle(training_data)
            
            # Split the data into batches
            batches = []
            for j in range(0, len(training_data), batch_size):
                batches.append(training_data[j:j+batch_size])
            
            # Train the model on each batch
            for batch in batches:
                self.train(batch, eta)
        
        return

    def train(self, batch, eta):
        """Updates the network weights and biases on a batch of training examples according to the rules:
        b <- b - eta (dc/db)
        w <- w - eta(dc/dw)"""
        # Run backprop on each training example to calculate dc/dw and dc/db. Keep a running sum.
        sum_dcdb = [np.zeros(b.shape) for b in self.biases]
        sum_dcdw = [np.zeros(w.shape) for w in self.weights]

        for input, output in batch:
            dcdb, dcdw = self.backprop(input, output)
            sum_dcdb = [i + j for i, j in zip(sum_dcdb, dcdb)]          
            sum_dcdw = [i + j for i, j in zip(sum_dcdw, dcdw)]          

        # Update biases according to b <- b - eta (dc/db)
        self.biases = [old_bias - (eta/len(batch)) * db for old_bias, db in zip(self.biases, sum_dcdb)]

        # Update weights according to b <- b - eta (dc/db)
        self.biases = [old_weight - (eta/len(batch)) * dw for old_weight, dw in zip(self.weights, sum_dcdw)]
        
        return

    def backprop():
        """TO-DO: Uses the backpropagation algorithm to calculate """
        raise NotImplementedError()
    

###ACTIVATION FUNCTIONS###
def relu(x):
    """Relu function for x"""
    return max(0, x)


def sigmoid(x):
    """Sigmoid function for x"""
    return 1.0/(1.0+np.exp(-x))


def activation_deriv(x, activation_function):
    """Returns the derivative of the activation function for x"""
    # Validate activation function
    if activation_function not in ["relu","sigmoid"]:
        raise ValueError("Activation function must be 'relu' or 'sigmoid'")

    # Calculate derivative
    if activation_function == "relu":
        if x > 0:
            return 1
        else:
            return 0
        
    if activation_function == "sigmoid":
        return sigmoid(x)*(1-sigmoid(x))