import numpy as np
from sklearn.metrics import accuracy_score

#-----ACTIVATIONS----- 
# activation functions and their derivatives
# all take as input a numpy array with shape (1, #units)

def identity(x):
    return x

def identity_prime(x):
    return np.ones(x.shape)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x < 0, 0, 1) # arbitrarily 0 or 1 in 0

def leaky_relu(x): 
    return np.where(x >= 0, x, 0.01 * x)

def leaky_relu_prime(x): 
    return np.where(x >= 0, 1, 0.01)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_prime(x):
    l = logistic(x)
    return l * (1 - l)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    t = tanh(x)
    return 1 - t**2

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_prime(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x, axis=1)) # TODO: normalization?
    return e / np.sum(e, axis=1)

def softmax_prime(x):
    # TODO: 
    # https://www.haio.ir/app/uploads/2021/12/Neural-Networks-from-Scratch-in-Python-by-Harrison-Kinsley-Daniel-Kukiela-z-lib.org_.pdf
    # Chapter 9, page 46
    return 

ACTIVATIONS = {
    'identity': identity,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'logistic': logistic,
    'tanh': tanh,
    'softplus': softplus
}

ACTIVATIONS_DERIVATIVES = {
    'identity': identity_prime,
    'relu': relu_prime,
    'leaky_relu': leaky_relu_prime,
    'logistic': logistic_prime,
    'tanh': tanh_prime,
    'softplus': softplus_prime
}

#-----LOSSES FOR BACKPROP-----
# loss functions and their derivatives
# all take as input a numpy array with shape (1, #units_output)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size # derivative w.r.t. y_pred

# link above, somewhere
def logloss(x):
    # TODO:
    return

def logloss_prime(x):
    # TODO:
    return

LOSSES = {
    'mse': mse
}

LOSSES_DERIVATIVES = {
    'mse': mse_prime
}

#-----LOSSES TO EVALUATE PERFORMANCE-----
# all take as input numpy arrays with shape (#samples, #tagets_per_sample)

def mse_score(y_true, y_pred):
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis=1))

def mee_score(y_true, y_pred):
    return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=1)))

def accuracy(y_true, y_pred):
    return accuracy_score(y_true=y_true, y_pred=y_pred)

#-----OTHERS-----
def unison_shuffle(x, y):
    seed = np.random.randint(0, 100000) 
    np.random.seed(seed) 
    np.random.shuffle(x)
    np.random.seed(seed) 
    np.random.shuffle(y)
    return x, y

# utility temporary function
def f_pred(pred):
    flattened_pred = np.empty(len(pred))
    for i in range(len(pred)):
        if pred[i][0] > 0:
            flattened_pred[i] = 1
        else:
            flattened_pred[i] = -1
    return flattened_pred