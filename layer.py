import numpy as np

# inherit from base class Layer


class Layer():
    # input_size = number of input neurons
    # output_size = number of output neurons
    id_count = 0

    def __init__(self, first, input_size, output_size, activation, activation_prime):
        self.input = None
        self.output = None
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.activation = activation
        self.activation_prime = activation_prime
        self.delta_w_old = np.zeros(shape = (input_size, output_size)) #previous weights used for the momentum
        if first: # TODO: do this better
            Layer.id_count = 0
        self.id = Layer.id_count
        Layer.id_count += 1       

    def set_weights(self, w, b):
        self.wights = w
        self.bias = b

    def update(self, delta_weights, delta_bias, learning_rate, batch_size, alpha, lambd):
        delta_weights /= batch_size
        delta_bias /= batch_size

        dw = - learning_rate * delta_weights + alpha * self.delta_w_old  # momentum
        self.weights += (1 - 2 * lambd) * dw # weight decay with penality term
        self.delta_w_old  = dw
        #self.weights += self.delta_w_old - lambd * self.weights #per me
        
        self.bias -= learning_rate * delta_bias
        
    def update2(self, delta_weights, delta_bias, learning_rate, batch_size, alpha, lambd):
        delta_weights /= batch_size
        #delta_bias /= batch_size
        
        #self.bias -= learning_rate * delta_bias
        
        dw = - learning_rate * delta_weights + alpha * self.delta_w_old
        self.weights +=  dw
        self.delta_w_old  = dw

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.net = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(self.net)
        return self.output

     # computes dE/dW, dE/dB for a given error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, error):
        delta = self.activation_prime(self.net) * error
        sum_w_delta = np.dot(delta, self.weights.T)
        weights_error = np.dot(self.input.T, delta)  # dE/dW
        # dBias = delta

        return sum_w_delta, weights_error, delta
