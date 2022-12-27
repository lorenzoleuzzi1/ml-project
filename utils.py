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

def softmax(x): # TODO: softmax può essere usata con numero unità output layer > 1
    x = normalize(x)
    return np.exp(x) / np.sum(np.exp(x))

def softmax_prime(x):
    f = softmax(x) 
    return np.diagflat(f) - np.dot(np.transpose(f), f)

# TODO: ricontrollare se funziona con scalari / vettori, in generale se funziona e come usarla nel nostro progetto
def crossEntropy_fun(y_true, y_pred):
    y_pred = np.log(y_pred)
    return - np.dot(y_pred, y_true)

def crossEntropy_der(y, lb):
    y = 1. / y
    return - y * lb

def binaryCrossEntropy_fun(y, lb): # usare solo le y è una probabilità -> activ fun softmax
    if lb == 0:
        return - np.log(1. - y)
    else:
        return - np.log(y)

def binaryCrossEntropy_der(y, lb):
    if lb == 0:
        return 1 / (1. - y)
    else:
        return - 1. / y

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

#-----LOSSES AND METRICS-----
# loss functions and their derivatives

# returns a scalar
def mse(y_true, y_pred):
    return np.sum(np.power(y_true - y_pred, 2)) / y_true.size # TODO: use scikit learn?

# returns a numpy array with shape (1, #units_output)
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size  # derivative w.r.t. y_pred

# returns a scalar
def mee(y_true, y_pred): # TODO: when used as a loss is equivalent to mse? 
    axis = 1
    if len(y_true.shape) == 1: axis = 0
    return np.sqrt( np.sum(np.power(y_true - y_pred, 2), axis=axis) ) / y_true.size
    #  TODO: sostituire (?) con np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=axis)))

# returns a numpy array with shape (1, #units_output)
def mee_prime(y_true, y_pred):
    f = mee(y_true, y_pred)
    if f == 0: return np.zeros(y_true.shape)
    else : return (y_pred - y_true) / ( (y_true.size ** 2) * f )
    # TODO: sostituire (?) con (y_pred - y_true) / f (quando usato come loss la media non viene fatta in mee)

def mrmse(y_true, y_pred): # mean root mean square error
    axis = 1
    if len(y_true.shape) == 1: axis = 0
    return np.mean(np.sqrt(np.mean(np.power(y_true - y_pred, 2), axis=axis)))

def mrmse_prime(y_true, y_pred):
    return (y_pred - y_true) / mrmse(y_true, y_pred)
    # TODO: sostituire (?) (y_pred - y_true) / (y_true.size * mrmse(y_true, y_pred))

# link above, somewhere
def logloss(x):
    # TODO:
    return

def logloss_prime(x):
    # TODO:
    return

LOSSES = {
    'mse': mse,
    'mee': mee,
    'mrmse': mrmse
} # REVIEW: devono essere tutte "medie"

LOSSES_DERIVATIVES = {
    'mse': mse_prime,
    'mee': mee_prime,
    'mrmse': mrmse_prime
}


def accuracy(y_true, y_pred): # TODO: da rivedere sistemando la codifica dei target
    threshold = 0 # per tanh, per softmax 0.5
    return accuracy_score(y_true=y_true, y_pred=np.where(y_pred > threshold, 1, -1))

EVALUATION_METRICS = {
    'mse': mse,
    'mee': mee,
    'mrmse': mrmse,
    'accuracy': accuracy
} # REVIEW: devono essere tutte "medie"

#-----OTHERS-----
def unison_shuffle(x, y, seed):
    if seed == None:
        seed = np.random.randint(0, 100000)
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y

def validation_split(x_train, y_train, percentage):
    validation_size = int(len(x_train)/100 * percentage) #% for val
    x_val = x_train[:validation_size]
    y_val = y_train[:validation_size]      
    x_train = x_train[validation_size:]
    y_train = y_train[validation_size:]
    return x_train, y_train, x_val, y_val

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

def mean_and_std(data):
    mean = np.mean(data)
    dev = np.std(data)
    return mean, dev

def normalize(data):
    mean, std = mean_and_std(data)
    return (data - mean) / std

def check_inputs():
    pass

#-----PLOT-----
def fold_plot(type, tr_results, val_results, avg_tr, avg_val):
    fold_count=1
    plt.figure()
    for tr, val in zip(tr_results, val_results):
        plt.plot(np.trim_zeros(tr, 'b'), color="blue",
                label="training {} fold {}".format(type, fold_count), alpha = 0.2)
        plt.plot(np.trim_zeros(val, 'b'), color="green",
                label="talidation {} fold {}".format(type, fold_count), alpha = 0.2)
        fold_count += 1
    plt.plot(avg_tr, color="blue",label="training {} avg".format(type))
    plt.plot(avg_val, color="green",label="validation {} avg".format(type))
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.grid()
    plt.title("k-fold training and validation {}".format(type))
    fig_name = "plot_k-fold_{}".format(type)
    plt.savefig(fig_name)

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