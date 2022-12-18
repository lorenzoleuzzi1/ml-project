import numpy as np
import matplotlib.pyplot as plt
from math import floor
from sklearn.metrics import accuracy_score

#-----ACTIVATIONS----- 
# activation functions and their derivatives
# all take as input a numpy array with shape (1, #units)

def identity(x):
    return x

def identity_prime(x):
    diag = np.ones(x.shape)
    return np.diagflat(diag)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    diag = np.where(x < 0, 0, 1) # arbitrarily 0 or 1 in 0
    return np.diagflat(diag)

def leaky_relu(x): 
    return np.where(x >= 0, x, 0.01 * x)

def leaky_relu_prime(x): 
    diag = np.where(x >= 0, 1, 0.01)
    return np.diagflat(diag)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_prime(x):
    l = logistic(x)
    diag = l * (1 - l)
    return np.diagflat(diag)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    t = tanh(x)
    diag = 1 - t**2
    return np.diagflat(diag)

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_prime(x):
    diag = 1 / (1 + np.exp(-x))
    return np.diagflat(diag)

def softmax(x):
    #e = np.exp(x - np.max(x, axis=1)) # TODO: normalization?
    #return e / np.sum(e, axis=1)
    return np.exp(x) / sum(np.exp(x))

def softmax_prime(x):
    f = softmax(x) 
    return np.diagflat(f) - np.dot(f, np.transpose(f))

def logloss(x):
    # TODO:
    return

def logloss_prime(x):
    # TODO:
    return

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

#-----LOSSES FOR BACKPROP-----
# loss functions and their derivatives
# all take as input a numpy array with shape (1, #units_output)

# returns a scalar
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2)) # REVIEW: to follow Micheli np.sum(np.power(y_true - y_pred, 2)) / 2 => half_sse

# returns a numpy array with shape (1, #units_output)
def mse_prime(y_true, y_pred): # REVIEW: to follow Micheli (y_pred - y_true) => half_sse_prime
    df = 2 * (y_pred - y_true) / y_true.size 
    #return 2 * (y_pred - y_true) / y_true.size # derivative w.r.t. y_pred
    return df

# returns a scalar
def ee(y_true, y_pred): # TODO: is equivalent to mse?
    return np.sqrt(np.sum(np.power(y_true - y_pred, 2)))

# returns a numpy array with shape (1, #units_output)
def ee_prime(y_true, y_pred):
    e = ee(y_true=y_true, y_pred=y_pred)
    return (y_pred - y_true) / e

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
    if len(y_true.shape) != 2 and len(y_true.shape) != 1:
        raise ValueError("Invalid shape")
    n_targets = 1
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(y_true.shape[0], 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(y_pred.shape[0], 1) 
    else:
        n_targets = y_pred.shape[1]
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis=1)/n_targets)

# REVIEW: as in the slides here it does not divide by the number of components (in mse_score instead the division is done)
def mee_score(y_true, y_pred):
    if len(y_true.shape) != 2 and len(y_true.shape) != 1:
        raise ValueError("Invalid shape")
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(y_true.shape[0], 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(y_pred.shape[0], 1) 
    return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=1)))

def accuracy(y_true, y_pred):
    y_pred = flatten_pred(y_pred) # TODO: sistemare
    return accuracy_score(y_true=y_true, y_pred=y_pred) 
# TODO: accuracy_score ERROR:
# Classification metrics can't handle a mix of continuous-multioutput and multilabel-indicator targets

#-----OTHERS-----
def unison_shuffle(x, y):
    seed = np.random.randint(0, 100000) 
    np.random.seed(seed) 
    np.random.shuffle(x)
    np.random.seed(seed) 
    np.random.shuffle(y)
    return x, y

# utility temporary function
def flatten_pred(pred): # TODO: per regressione multipla non funziona, 
    # calcola flatten_pred solo per prima colonna e poi copia nelle successiva
    flattened_pred = np.empty(pred.shape)
    for i in range(len(pred)):
        if pred[i][0] > 0:
            flattened_pred[i] = 1
        else:
            flattened_pred[i] = -1
    return flattened_pred


#-----PLOT-----
def error_plot(tr_error, val_error):
    epochs = len(tr_error)
    epoch_vector = np.linspace(1, epochs, epochs)
    plt.figure()
    plt.plot(epoch_vector, tr_error, "b",
             label="Training error", linewidth=1.5)
    plt.plot(epoch_vector, val_error, "r--",
             label="Validation error", linewidth=1.5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.grid()
    plt.title("Training and validation error on monks 1 dataset")
    fig_name = "ml-project-ErrorPlot"
    plt.savefig(fig_name)


def accuracy_plot(tr_accuracy, val_accuracy):
    tr_epochs = len(tr_accuracy)
    val_epochs = len(val_accuracy)
    tr_epoch_vector = np.linspace(1, tr_epochs, tr_epochs)
    val_epoch_vector = np.linspace(1, tr_epochs, val_epochs)
    
    plt.figure()
    plt.plot(tr_epoch_vector, tr_accuracy, "b",
             label="Trainig accuracy", linewidth=1.5)
    plt.plot(val_epoch_vector, val_accuracy, "r--",
             label="Validation accuracy", linewidth=1.5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.title("Training and validation accuracy on monks 1 dataset")
    fig_name = "ml-project-AccuracyPlot"
    plt.savefig(fig_name)
    
