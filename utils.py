import numpy as np
import matplotlib.pyplot as plt
from math import floor

#-----ACTIVATIONS----- 
# activation functions and their derivatives

def identity(x):
    return x

def identity_prime(x):
    return 1

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return 0 if x < 0 else 1

def leaky_relu(x): 
    return x if x >= 0 else 0.01 * x

def leaky_relu_prime(x): 
    return 1 if x >= 0 else 0.01

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_prime(x):
    l = logistic(x)
    return l * (1 - l)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_prime(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def softmax_prime(x):
    # TODO: 
    return

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

#-----LOSSES-----
# loss functions and their derivatives

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2)) # TODO: mean is needed? yes needed 

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def rmse(y_true, y_pred): 
    return np.sqrt(mse(y_true, y_pred))

def rmse_prime(y_true, y_pred): # TODO: check if right
    return (1 / (2 * rmse(y_true, y_pred))) * mse_prime(y_true, y_pred)

def ee(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, 2)
    # same as 
    # np.sqrt(np.sum(np.power(y_true - y_pred, 2)))

def ee_prime(y_true, y_pred): # TODO: check if right
    return (y_true - y_pred) / ee(y_true, y_pred)

LOSSES = {
    'mse': mse,
    'rmse': rmse,
    'mee': ee
}

LOSSES_DERIVATIVES = {
    'mse': mse_prime,
    'rmse': rmse_prime,
    'mee': ee_prime
}

#-----LEARNING RATE-----
def fixed(learning_rate):
    return lambda x: learning_rate
    
def linear_decay(tau, starting_learning_rate):
    final_learning_rate = starting_learning_rate * 0.1

    def fun(epoch):
        alpha = epoch / tau
        learning_rate = (1 - alpha) * starting_learning_rate + alpha * final_learning_rate
        
        if epoch == 0:
            return starting_learning_rate
        
        if (learning_rate < final_learning_rate or epoch >= tau):
            return final_learning_rate
        else:
            return learning_rate

    return fun

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
        if pred[i][0][0] > 0:
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
    