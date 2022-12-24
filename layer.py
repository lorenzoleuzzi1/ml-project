import numpy as np
import copy
from utils import ACTIVATIONS, ACTIVATIONS_DERIVATIVES

class Layer():

    def __init__(self, fan_in, fan_out, activation):
        self.input = None
        self.output = None
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = activation
        self.activation_fun = ACTIVATIONS[activation]
        self.activation_prime = ACTIVATIONS_DERIVATIVES[activation]
        self.weights_init()
        self.deltas_weights = np.zeros(shape = (fan_in, fan_out))
        self.deltas_bias = np.zeros(shape = (1, fan_out))
        self.velocity_w = np.zeros(shape = (fan_in, fan_out))
        self.velocity_b = np.zeros(shape = (1, fan_out))

    def set_weights(self, w, b):
        self.weights = w
        self.bias = b
        self.init_weights = copy.deepcopy(w)
        self.init_bias = copy.deepcopy(b)

    def weights_init(self):
        '''
        TODO: softplus? identity?
        '''
        if self.activation == 'relu' or self.activation == 'leaky_relu':
            # He inizialization [https://arxiv.org/abs/1502.01852]
            self.weights = np.random.normal(scale=np.sqrt(2 / self.fan_in), size=(self.fan_in, self.fan_out))
            self.bias = np.zeros((1, self.fan_out))
        else: # softmax? nel paper la utilizzano
            # Xavier initialization [https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
            factor = 6.0
            if self.activation == "logistic":
                factor = 2.0 # TODO: la logistic in 0 è 1/2, Xavier assume di avere tanh che vale 0 in 0
            bound = np.sqrt(factor / (self.fan_in + self.fan_out))
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            self.bias = np.zeros((1, self.fan_out))
        self.init_weights = copy.deepcopy(self.weights)
        self.init_bias = copy.deepcopy(self.bias)

    def update(self, learning_rate, batch_size, alpha, lambd, nesterov):
        self.deltas_weights /= batch_size
        self.deltas_bias /= batch_size
 
        # weights and bias update
        velocity_w =  alpha * self.velocity_w - learning_rate * self.deltas_weights
        velocity_b =  alpha * self.velocity_b - learning_rate * self.deltas_bias
        self.weights -= lambd * self.weights
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
        delta_w = np.outer(self.input, delta) 
        self.deltas_weights += delta_w
        self.deltas_bias += delta 
        return delta_i
