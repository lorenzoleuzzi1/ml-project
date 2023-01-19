import numpy as np
from scipy.special import xlogy
from sklearn.metrics import accuracy_score
import pickle

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
    exps = np.exp(x - np.max(x)) # subtracting the maximum avoids overflow
    return exps / np.sum(exps)

def softmax_prime(x):
    f = softmax(x)
    return np.diagflat(f) - np.outer(f, f)

ACTIVATIONS = {
    'identity': identity,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'logistic': logistic,
    'tanh': tanh,
    'softplus': softplus,
    'softmax': softmax
}

ACTIVATIONS_DERIVATIVES = {
    'identity': identity_prime,
    'relu': relu_prime,
    'leaky_relu': leaky_relu_prime,
    'logistic': logistic_prime,
    'tanh': tanh_prime,
    'softplus': softplus_prime,
    'softmax': softmax_prime
}

ACTIVATIONS_THRESHOLDS = {
    #TODO
    #'identity': 0,
    #'relu': 0,
    #'leaky_relu': 0,
    'logistic': 0.5,
    'tanh': 0,
    #'softplus': ln(1+e^0)?,
    'softmax': 0.5
}

#-----LOSSES AND METRICS-----
# loss functions and their derivatives

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def mee(y_true, y_pred):
    if len(y_true.shape) == 1:
        return np.sqrt(np.sum(np.power(y_true.reshape(y_true.shape[0]) - y_pred, 2)))
    else:
        return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=1)))

def mee_prime(y_true, y_pred):
    f = mee(y_true, y_pred)
    if f == 0: 
        return np.zeros(y_true.shape)
    else : 
        return (y_pred - y_true) / f

def log_loss(y_true, y_pred):
    eps = np.finfo(y_pred.dtype).eps  # machine precision for that type
    y_pred = np.clip(y_pred, eps, 1 - eps) # if lower than eps replaced with eps, if higher than 1-eps replaced with 1-eps
    return -xlogy(y_true, y_pred).sum() / y_pred.shape[0]

def log_loss_prime(y_true, y_pred):
    return - y_true / y_pred

LOSSES = {
    'mse': mse,
    'mee': mee,
    'logloss': log_loss
}

LOSSES_DERIVATIVES = {
    'mse': mse_prime,
    'mee': mee_prime,
    'logloss': log_loss_prime
}

EVALUATION_METRICS = {
    'mse': mse,
    'mee': mee,
    'logloss': log_loss,
    'accuracy': accuracy_score
}

def save_obj(obj, path):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()