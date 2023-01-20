import numpy as np
import copy
from utils import ACTIVATIONS, ACTIVATIONS_DERIVATIVES

class Layer():
    """
    A class representing a feed-forward fully connected layer of a 
    neural network.

    It implements the methods needed to propagate input values to the units of
    the next layer and to backpropagate the errors coming from them.

    Attributes
    ----------
    input: ndarray
        Inputs values to the units of the layer
    net: ndarray
        Net input values to the units of the layer
    output: ndarray
        Outputs values of the units of the layer
    fan_in: int
        Number of inputs to each unit of the layer
    fan_out: int
        Number of outputs of each unit of the layer
    activation: str
        Activation function name
    activation_fun: function
        Activation function
    activation_prime: function
        Derivative of the activation function
    weights: ndarray
        Current weights values associated to the incoming links of the units
    bias: ndarray
        Current bias value associated to each unit
    init_weights: ndarray
        Initial weights values associated to the incoming links of the units
    init_bias: ndarray
        Initial bias value associated to each unit
    deltas_weights: ndarray
        Gradient of the error w.r.t the weights
    deltas_bias: ndarray
        Gradient of the error w.r.t the biases
    velocity_w: ndarray
        Weights velocity term to apply momentum
    velocity_b: ndarray
        Bias velocity term to apply momentum
    """

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
        self.deltas_bias = np.zeros(shape = (fan_out))
        self.velocity_w = np.zeros(shape = (fan_in, fan_out))
        self.velocity_b = np.zeros(shape = (fan_out))

    def set_weights(self, weights, bias):
        """
        Set layer's weights and biases.

        Parameters
        ----------
        weights: ndarray
            Weights values to set
        bias: ndarray
            Biases values to set
        """

        self.weights = weights
        self.bias = bias
        self.init_weights = copy.deepcopy(weights)
        self.init_bias = copy.deepcopy(bias)

    def weights_init(self, distribution, bound):
        """
        Initialize weights and biases.

        Parameters
        ----------
        distribution: str
            The distribution from which weights and biases are sampled.
            If None 'Xavier' or 'He' initializations are used according
            to the activation function of the layer.
        bound: float
            The bound of the interval from which weights and biases are sampled.
            If distribution='uniform', weights and biases are uniformly 
            distributed over the interval [bound, bound).
            If distribution='normal' bound is the standard deviation of the
            normal distribution from which weights and biases are sampled.
            Only used when distribution is not None.

        References
        ----------
        Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of 
        training deep feedforward neural networks." 
        Proceedings of the thirteenth international conference on artificial 
        intelligence and statistics. 
        JMLR Workshop and Conference Proceedings, 2010.

        Bengio, Yoshua. "Practical recommendations for gradient-based training 
        of deep architectures." 
        Neural networks: Tricks of the trade.
        Springer, Berlin, Heidelberg, 2012. 437-478.

        He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
        human-level performance on imagenet classification."
        Proceedings of the IEEE international conference on computer vision.
        2015.
        """

        if distribution:
            if distribution == 'uniform':
                self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            else:
                self.weights = np.random.normal(scale=bound, size=(self.fan_in, self.fan_out))
            self.bias = np.zeros((self.fan_out))
        elif self.activation in ['relu', 'leaky_relu', 'softplus']:
            self.weights = np.random.normal(scale=np.sqrt(2 / self.fan_in), size=(self.fan_in, self.fan_out))
            self.bias = np.zeros((self.fan_out))
        else:
            factor = 6.0
            bound = np.sqrt(factor / (self.fan_in + self.fan_out))
            if self.activation in ['logistic', 'softmax']: 
                bound *= 4 
            self.weights = np.random.uniform(-bound, bound, (self.fan_in, self.fan_out))
            self.bias = np.zeros((self.fan_out))
        self.init_weights = copy.deepcopy(self.weights)
        self.init_bias = copy.deepcopy(self.bias)

    def update(self, learning_rate, batch_size, alpha, lambd, nesterov):
        """
        Update weights and biases with the accumulated deltas.

        Parameters
        ----------
        learning_rate: float
            Learning rate value
        batch_size: int
            Number of patterns in the batch
        alpha: float
            Momentum coefficient
        lambd: float
            L2 regularization coefficient
        nesterov: bool
            Wheter to apply Nesterov momentum
        """

        # normalize the accumulated deltas dividing by the batch size
        self.deltas_weights /= batch_size
        self.deltas_bias /= batch_size
 
        # compute weights and biases velocities to apply momentum
        velocity_w =  alpha * self.velocity_w - learning_rate * self.deltas_weights
        velocity_b =  alpha * self.velocity_b - learning_rate * self.deltas_bias

        # apply l2 regularization
        self.weights -= 2 * lambd * self.weights

        if nesterov: # apply Nesterov momentum
            self.weights += alpha * velocity_w - learning_rate * self.deltas_weights
            self.bias += alpha * velocity_b - learning_rate * self.deltas_bias
        else: # apply classical momentum
            self.weights += velocity_w
            self.bias += velocity_b
        
        # save velocities for the next update
        self.velocity_w  = velocity_w
        self.velocity_b = velocity_b

        # clear deltas for the next batch
        self.deltas_weights.fill(0)
        self.deltas_bias.fill(0)

    def forward_propagation(self, input_data):
        """
        Performs the forward pass.

        Parameters
        ----------
        input_data: ndarray
            Layer's inputs

        Returns
        -------
        output: ndarray
            Layer's outputs
        """

        self.input = input_data
        self.net = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation_fun(self.net)
        return self.output

    def backward_propagation(self, delta_j):
        """
        Performs the backward pass.

        Parameters
        ----------
        delta_j: ndarray
            Incoming error (from the units in the next layer j)

        Returns
        -------
        delta_i: ndarray
            Outcoming error (from the units in the current layer i)
        """

        act_prime = self.activation_prime(self.net)
        if act_prime.ndim == 2:
            delta = np.dot(delta_j, act_prime)
        else: # for a more efficient computation
            delta = delta_j * act_prime
        
        # compute outcoming error
        delta_i = np.dot(delta, np.transpose(self.weights))
        
        # cumulate deltas for weights and biases updates
        self.deltas_weights += np.outer(self.input, delta)
        self.deltas_bias += delta 

        return delta_i