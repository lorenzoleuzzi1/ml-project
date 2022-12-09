import numpy as np

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
    return np.mean(np.power(y_true - y_pred, 2)) # REVIEW: to follow Micheli np.sum(np.power(y_true - y_pred, 2)) / 2 => half_se

def mse_prime(y_true, y_pred): # REVIEW: to follow Micheli (y_pred - y_true) => half_se_prime
    return 2 * (y_pred - y_true) / y_true.size # derivative w.r.t. y_pred

def ee(y_true, y_pred):
    return np.sqrt(np.dot(y_true - y_pred, y_true - y_pred)) # np.sqrt(np.sum(np.power(y_true - y_pred, 2)))

def ee_prime(y, z):
    return (y - z) / np.sqrt(np.dot(y - z, y - z))

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
# REVIEW: we could use the ones of scikit learn?
#         No because scikit learn computes the mean two times (to compare different tasks), 
#         in the slides it is computed once...
def mse_score(y_true, y_pred):
    if len(y_true.shape) != 2 and len(y_true.shape) != 1:
        raise ValueError("Invalid shape")
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(y_true.shape[0], 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(y_pred.shape[0], 1) 
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis=1))

def mee_score(y_true, y_pred):
    if len(y_true.shape) != 2 and len(y_true.shape) != 1:
        raise ValueError("Invalid shape")
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(y_true.shape[0], 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(y_pred.shape[0], 1) 
    return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=1)))

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