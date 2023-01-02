import numpy as np
import copy
from utils import ACTIVATIONS, ACTIVATIONS_DERIVATIVES

class Layer():

    def __init__(self, fan_in, fan_out, activation, weights_dist, weights_bound):
        self.input = None
        self.output = None
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = activation
        self.activation_fun = ACTIVATIONS[activation]
        self.activation_prime = ACTIVATIONS_DERIVATIVES[activation]
        self.weights_init(weights_dist, weights_bound)
        self.deltas_weights = np.zeros(shape = (fan_in, fan_out))
        #self.deltas_bias = np.zeros(shape = (1, fan_out))
        self.deltas_bias = np.zeros(shape = (fan_out))
        self.velocity_w = np.zeros(shape = (fan_in, fan_out))
        #self.velocity_b = np.zeros(shape = (1, fan_out))
        self.velocity_b = np.zeros(shape = (fan_out))

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.init_weights = copy.deepcopy(weights)
        self.init_bias = copy.deepcopy(bias)

    def weights_init(self, distribution, bound):
        if distribution:
            if distribution == 'uniform':
                self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            else:
                self.weights = np.random.normal(scale=bound, size=(self.fan_in, self.fan_out))
            #self.bias = np.zeros((1, self.fan_out)) # TODO: same distribution for the bias?
            self.bias = np.zeros((self.fan_out))
        elif self.activation in ['relu', 'leaky_relu', 'softplus']:
            # He inizialization [https://arxiv.org/abs/1502.01852]
            self.weights = np.random.normal(scale=np.sqrt(2 / self.fan_in), size=(self.fan_in, self.fan_out))
            #self.bias = np.zeros((1, self.fan_out))
            self.bias = np.zeros((self.fan_out))
        else: # softmax? nel paper la utilizzano
            # Xavier initialization [https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
            factor = 6.0
            bound = np.sqrt(factor / (self.fan_in + self.fan_out))
            if self.activation in ['logistic', 'softmax']: 
                bound *= 4 # https://arxiv.org/pdf/1206.5533.pdf pag 15
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            #self.bias = np.zeros((1, self.fan_out))
            self.bias = np.zeros((self.fan_out))
        self.init_weights = copy.deepcopy(self.weights)
        self.init_bias = copy.deepcopy(self.bias)

    def update(self, learning_rate, batch_size, alpha, lambd, nesterov):
        self.deltas_weights /= batch_size
        self.deltas_bias /= batch_size
 
        # weights and bias update
        velocity_w =  alpha * self.velocity_w - learning_rate * self.deltas_weights
        velocity_b =  alpha * self.velocity_b - learning_rate * self.deltas_bias
        self.weights -= 2 * lambd * self.weights
        if nesterov:
            self.weights += alpha * velocity_w - learning_rate * self.deltas_weights
            self.bias += alpha * velocity_b - learning_rate * self.deltas_bias
        else:
            self.weights += velocity_w
            self.bias += velocity_b

        self.velocity_w  = velocity_w
        self.velocity_b = velocity_b
        self.deltas_weights.fill(0)
        self.deltas_bias.fill(0)

        # self.bias -= learning_rate * self.deltas_bias

    def forward_propagation(self, input_data):
        self.input = input_data
        self.net = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation_fun(self.net)
        return self.output

    def backward_propagation(self, delta_j): 
        delta = np.dot(delta_j, self.activation_prime(self.net))
        delta_i = np.dot(delta, np.transpose(self.weights)) 
        #weights_error = np.dot(np.transpose(self.input), delta) 
        #delta_w = np.outer(self.input, delta) 
        #self.deltas_weights += delta_w
        self.deltas_weights += np.outer(self.input, delta)
        self.deltas_bias += delta 

        return delta_i