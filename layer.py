import numpy as np

class Layer():

    def __init__(self, first, fan_in, fan_out, weights_init, activation, activation_prime):
        self.input = None
        self.output = None
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = activation
        self.activation_prime = activation_prime
        self.weights_init(weights_init)
        self.deltas_weights = np.zeros(shape = (fan_in, fan_out))
        self.deltas_bias = np.zeros(shape = (1, fan_out))
        self.deltas_weights_prev = np.zeros(shape = (fan_in, fan_out)) #previous weights used for the momentum

    def set_weights(self, w, b):
        self.wights = w
        self.bias = b

    def weights_init(self, method):
        ''' 
        TODO: 
        GlorotBengio method
        https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        scikit learn usa GlorotBengioNorm per "identity", "tanh", "relu" e "softmax".
        Per leaky_relu and softplus sono okay? Credo di sì dal momento che la 
        derivata di softplus derivative è simile a quella di tanh.
        Questa initizializzazione viene fatto per evitare la saturazione 
        (avviene quando la derivata della funzione di attivazione è vicina a 0).
        Per linear, relu, leaky_relu questo metodo funziona comunque?
        Ad ogni modo dobbiamo evitare 0, valori troppo alti (quando lo sono? > 1?), 
        pesi tutti uguali (quando con i seguenti approcci queste condizioni non sono soddisfatte?)
        Io userei He per relu/leaky relu e GlorotBengioNorm negli altri casi.
        TODO: cercare con identity cosa si usa!
        Qui
        https://link.springer.com/article/10.1007/s12065-022-00795-y
        con linear Xavier (alcuni usano questo nome per riferirsi a GlorotBengioNorm, altri a GlorotBengio) funziona meglio.
        '''
        if method == 'GlorotBengioNorm':
            factor = 6.0
            if self.activation == "logistic":
                factor = 2.0 # TODO: why?
            bound = np.sqrt(factor / (self.fan_in + self.fan_out))
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            self.bias = np.random.uniform(-bound, bound, (1, self.fan_out)) # TODO: maybe 0?
        elif method == 'GlorotBengio':
            bound = 1 / np.sqrt(self.fan_in)
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            self.bias = np.zeros((1, self.fan_out))
        elif method == 'He': # for relu / leaky https://arxiv.org/abs/1502.01852
            # TODO: uniform o normal?
            self.weights = np.random.normal(scale=np.sqrt(2 / self.fan_in), size=(self.fan_in, self.fan_out)) # TODO: simile a sopra?
            self.bias = np.random.normal(scale=np.sqrt(2 / self.fan_in), size=(1, self.fan_out)) # TODO: maybe 0?
        elif method == 'Micheli': # TODO: good for standardized data, not for output layer, not if fan_in too large (how large?)
            bound = 0.7 * (2 / self.fan_in)
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            self.bias = np.random.uniform(-0.1, 0.1, (1, self.fan_out))
        elif method == 'normal':
            self.weights = np.random.normal(size=(self.fan_in, self.fan_out))
            self.bias = np.random.normal(size=(1, self.fan_out))
        elif method == 'our':
            self.weights = np.random.rand(self.fan_in, self.fan_out) - 0.5
            self.bias = np.random.rand(1, self.fan_out) - 0.5

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
    def backward_propagation(self, error):        
        delta = np.dot(error, self.activation_prime(self.net))
        sum_w_delta = np.dot(delta, np.transpose(self.weights))
        weights_error = np.dot(np.transpose(self.input), delta) # dE/dW
        # dBias = delta
        #accumalte deltas
        self.deltas_weights += weights_error
        self.deltas_bias += delta

        return sum_w_delta
