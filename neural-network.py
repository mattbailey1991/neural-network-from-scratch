import random
import numpy as np

class NeuralNetwork():
    
    def __init__(self, shape, activation_function):
        
        """Shape of the neural network, defined as an array of layer sizes. 
        For example, a neural network with five input variables, a hidden layer with three neurons, and two output nodes would be: [5,3,2]"""
        self.shape = shape
        
        """Assigns and validates activation function"""
        if activation_function not in ["relu","sigmoid"]:
            raise ValueError("Activation function must be 'relu' or 'sigmoid'")
        self.activation_function = activation_function

        """Initialises random biases using normal distribution. 
        The bias for neuron j in layer m is located at self.biases[m][j]"""
        self.biases = [np.random.randn(y, 1) for y in shape[1:]]

        """Initialises random weights using normal distribution.
        The weight between neuron j in layer m and neuron k in layer m-1 is located at self.weights[m][j][k]"""
        self.weights = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]
        
        return


    def feedforward(self, input, a_z_matrix = False):
        """Feeds an input through the neural network to obtain a predicted output. 
        Input shape must be (self.shape[0],1).
        Activation function can be relu or sigmoid.
        Activation of each layer m is given by a(m) = f(wa(m-1) + b)
        Returns a shape (self.shape[-1],1) output vector
        Alternatively, if a_z_matrix = True, returns a tuple with (output vector, z_matrix, activation_matrix)"""
        
        # Initialise activation and z/a storage
        activation = input
        if a_z_matrix:
            activation_matrix = [activation]
            z_matrix = []
        
        # Feed forward through each layer of the neural network, saving z and a values
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            if self.activation_function == "relu":
                activation = relu(z)
            elif self.activation_function == "sigmoid":
                activation = sigmoid(z)
            if a_z_matrix:
                z_matrix.append(z)
                activation_matrix.append(activation)
        
        # Return a tuple with the output activation, z_matrix, and activation matrix
        if a_z_matrix:
            return activation, z_matrix, activation_matrix
        
        else:
            return activation


    def train_model(self, training_data, epochs, batch_size, eta):
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
                self.train_batch(batch, eta)
        
        return

    def train_batch(self, batch, eta):
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

    def backprop(self, input, output):
        """Uses the backpropagation algorithm to calculate dc/db and dc/dw. 
        Uses the four equation of backpropagation: 
        dc/db = delta
        dc/dw = a(m-1)delta  
        (output layer) delta = dc/da * df/dz and f is the activation function
        (other layers) delta = w(m+1)delta(m+1) * df/dz
        """
        dcdb = np.zeros(b.shape for b in self.biases)
        dcdw = np.zeros(w.shape for w in self.weights)
        
        # Feed forward input to obtain zs and activations
        activation, z_matrix, activation_matrix = self.feedforward(input, a_z_matrix=True)

        # Calculate dcdb and dcdw for output layer
        delta = cost_deriv(activation_matrix[-1], output) * activation_deriv(z_matrix[-1], self.activation_function)
        dcdb[-1] = delta
        dcdw [-1] = np.dot(delta, activation_matrix[-2].transpose())

        # Backward propogate the erros through the neural network to obtain dcdb and dcdw for all other layers

        for m in range(2, len(self.shape)):
            z = z_matrix[-m]
            a_deriv = activation_deriv(z)
            delta = np.dot(self.weights[-m+1].transpose(), delta) * a_deriv
            dcdb[-m] = delta
            dcdw [-m] = np.dot(delta, activation_matrix[-m-1].transpose())
        
        return dcdb, dcdw
    
    def evaluate(self, test_data):
        """Returns a tuple with (count of correct predictions, % of correct predictions)"""
        predictions = [] 
        outputs = []
        for input, output in test_data:
            predictions.append(self.feedforward(input))
            outputs.append(output)
        correct_count = sum(int(x == y) for (x, y) in zip(predictions, outputs))
        percent_correct = correct_count / len(test_data)
        return correct_count, percent_correct

###ACTIVATION FUNCTIONS###
def relu(z):
    """Relu function for x"""
    def simple_relu(z):
        return max(0,z)
    vec_relu = np.vectorize(simple_relu, otypes=[float])
    return vec_relu(z)


def sigmoid(z):
    """Sigmoid function for x"""
    return 1.0/(1.0+np.exp(-z))


###DERIVATIVES####
def activation_deriv(z, activation_function):
    """Returns the derivative of the activation function for x"""
    # Validate activation function
    if activation_function not in ["relu","sigmoid"]:
        raise ValueError("Activation function must be 'relu' or 'sigmoid'")

    # Calculate derivative
    if activation_function == "relu":
        def simple_relu_deriv(z):                
            if z > 0:
                return 1
            else:
                return 0
        vec_relu_deriv = np.vectorize(simple_relu_deriv, otypes=[float])
        return vec_relu_deriv(z)

    if activation_function == "sigmoid":
        return sigmoid(z)*(1-sigmoid(z))
    

def cost_deriv(activation, output):
    """Partial derivative of quadratic cost function with respect to a particular output activation"""
    return (activation - output)