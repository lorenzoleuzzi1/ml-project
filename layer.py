import numpy as np
from utils import ACTIVATIONS, ACTIVATIONS_DERIVATIVES

class Layer():

    def __init__(self, fan_in, fan_out, activation):
        self.input = None
        self.output = None
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = ACTIVATIONS[activation]
        self.activation_prime = ACTIVATIONS_DERIVATIVES[activation]
        self.weights_init(activation)
        self.deltas_weights = np.zeros(shape = (fan_in, fan_out))
        self.deltas_bias = np.zeros(shape = (1, fan_out))
        self.deltas_weights_prev = np.zeros(shape = (fan_in, fan_out)) #previous weights used for the momentum

    def set_weights(self, w, b):
        self.weights = w
        self.bias = b

    def weights_init(self, activation):
        '''
        Qui https://link.springer.com/article/10.1007/s12065-022-00795-y per identity usa il secondo
        Per softplus, softmax, logistic, identity
        '''
        if activation == 'relu' or activation == 'leaky_relu': #TODO: softplus??
            # He inizialization [https://arxiv.org/abs/1502.01852]
            self.weights = np.random.normal(scale=np.sqrt(2 / self.fan_in), size=(self.fan_in, self.fan_out))
            self.bias = np.zeros((1, self.fan_out))
        else:
            # Xavier initialization [https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
            factor = 6.0
            if activation == "logistic":
                factor = 2.0 # TODO: perchè? ha media 1/2, le assunzioni per cui viene derivata non valgono
            bound = np.sqrt(factor / (self.fan_in + self.fan_out))
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            self.bias = np.zeros((1, self.fan_out))

    def update(self, learning_rate, batch_size, alpha, lambd, nesterov):
        self.deltas_weights /= batch_size
        self.deltas_bias /= batch_size
 
        dw =  alpha * self.deltas_weights_prev - learning_rate * self.deltas_weights # classic momentum
        self.weights -= lambd*self.weights
        if nesterov:
            self.weights += alpha * dw - learning_rate * self.deltas_weights #nesterov and update
        else:
            self.weights += dw

        #self.weights -= lambd * self.weights #weight decay Tickonov
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

    def backward_propagation(self, delta_j): 
        delta = np.dot(delta_j, self.activation_prime(self.net))
        delta_i = np.dot(delta, np.transpose(self.weights)) 
        #weights_error = np.dot(np.transpose(self.input), delta) 
        delta_w = np.outer(self.input, delta) 

        #accumalte deltas
        self.deltas_weights += delta_w
        self.deltas_bias += delta 

        return delta_i
