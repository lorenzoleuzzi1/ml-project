import numpy as np

class Layer():

    def __init__(self, input_size, output_size, activation, activation_prime):
        self.input = None
        self.output = None
        # random init ok. if we want different case for different activation (img w init)
        self.weights = np.random.uniform(-0.7, 0.7,(input_size, output_size))
        self.bias = np.zeros(shape = (1, output_size))
        self.activation = activation
        self.activation_prime = activation_prime
        self.deltas_weights = np.zeros(shape = (input_size, output_size))
        self.deltas_bias = np.zeros(shape = (1, output_size))
        self.deltas_weights_prev = np.zeros(shape = (input_size, output_size)) #previous weights used for the momentum   

    def set_weights(self, w, b):
        self.wights = w
        self.bias = b

    # def update(self, delta_weightsX, delta_biasX, learning_rate, batch_size, alpha, lambd, nesterov):
    def update(self, learning_rate, batch_size, alpha, lambd, nesterov):

        self.deltas_weights /= batch_size
        self.deltas_bias /= batch_size
 
        dw =  alpha * self.deltas_weights_prev - learning_rate * self.deltas_weights # classic momentum
        if nesterov:
            self.weights = self.weights + alpha * dw - learning_rate * self.deltas_weights #nesterov and update
        else:
            self.weights = self.weights + dw #nesterov

        self.weights -= lambd * self.weights #weight decay Tickonov
        self.deltas_weights_prev  = dw

        self.bias -= learning_rate * self.deltas_bias

        self.deltas_weights.fill(0)
        self.deltas_bias.fill(0)


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
        #accumalte deltas
        self.deltas_weights += weights_error
        self.deltas_bias += delta

        return sum_w_delta
