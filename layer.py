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
        self.wights = w
        self.bias = b

    def weights_init(self, activation):
        '''
        Ad ogni modo dobbiamo evitare 0, valori troppo alti (quando lo sono? > 1?), 
        pesi tutti uguali (quando con i seguenti approcci queste condizioni non sono soddisfatte?)
        # TODO: per identity, logisitc e sofplus è giusto il secondo metodo?
        Qui https://link.springer.com/article/10.1007/s12065-022-00795-y per identity usa il secondo
        Per softplus e logistic??
        '''
        # TODO: usiamo uniform o normal? nel paper di He sembra sia equivalente...
        if activation == 'relu' or activation == 'leaky_relu':
            # He inizialization [https://arxiv.org/abs/1502.01852]
            self.weights = np.random.normal(scale=np.sqrt(2 / self.fan_in), size=(self.fan_in, self.fan_out))
            self.bias = np.zeros((1, self.fan_out))
        else:
            # Xavier initialization [https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
            factor = 6.0
            if activation == "logistic":
                factor = 2.0 # TODO: perchè? usato in scikit learn nel paper originale non si trova...
            bound = np.sqrt(factor / (self.fan_in + self.fan_out))
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            self.bias = np.zeros((1, self.fan_out)) # REVIEW: nel paper sembra 0, in scikit learn inizializzano come per i weights

    """def update(self, learning_rate, batch_size, alpha, lambd, nesterov):

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
        self.deltas_bias.fill(0)"""
    def update(self, learning_rate, batch_size, alpha, lambd, nesterov):
        # TODO: ricontrollare
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

    # computes dE/dW, dE/dB for a given error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, error): # REVIEW: rename error --> delta_j
        delta = np.dot(error, self.activation_prime(self.net))
        sum_w_delta = np.dot(delta, np.transpose(self.weights)) # REVIEW: rename sum_w_delta --> delta_i
        #weights_error = np.dot(np.transpose(self.input), delta) # dE/dW
        weights_error = np.outer(self.input, delta) # REVIEW: rename weights_error --> delta_w
        # dBias = delta
        #accumalte deltas
        self.deltas_weights += weights_error
        self.deltas_bias += delta

        return sum_w_delta
