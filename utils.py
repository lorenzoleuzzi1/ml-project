import numpy as np
from scipy.special import xlogy
import matplotlib.pyplot as plt
from math import floor
from sklearn.metrics import accuracy_score
import json

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
    """f = softmax(x) 
    return np.diagflat(f) - np.dot(np.transpose(f), f)"""

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

# returns a scalar
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))
    """axis = 1
    if len(y_true.shape) == 1: axis = 0
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis=axis) / y_true.shape[axis])"""

# returns a numpy array with shape (1, #units_output)
def mse_prime(y_true, y_pred):
    # GIULIA: mettere controllo sugli assi, le derivate sono uguali
    return 2 * (y_pred - y_true) / y_true.size
    """axis = 1
    if len(y_true.shape) == 1: axis = 0
    return 2 * (y_pred - y_true) / y_true.shape[axis] # derivative w.r.t. y_pred"""

# returns a scalar
def mee(y_true, y_pred):
    if len(y_true.shape) == 1:
        return np.sqrt(np.sum(np.power(y_true.reshape(y_true.shape[0]) - y_pred, 2)))
    else:
        return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=1)))

# returns a numpy array with shape (1, #units_output)
def mee_prime(y_true, y_pred):
    f = mee(y_true, y_pred)
    if f == 0: 
        return np.zeros(y_true.shape)
    else : 
        return (y_pred - y_true) / f

def logloss(y_true, y_pred):
    p = logistic(y_pred)
    return np.mean( -sum(y_true * np.log(p)) )
    # TODO: così va bene per la multiclassificazione s.s.s. abbiamo un neurone di output per ogni classe
    # se decidiamo che per classificazione binaria volgiamo un solo neurone di output dobbiamo allora sistemare logloss
    # distinguendo caso multiclasse e caso binario -> io non lo farei perché è una roba in più da controllare, ed è uno sbatti
    # ma comunque sono due righe da aggiugere a codice

def logloss_prime(y_true, y_pred):
    p = logistic(y_pred)
    inv = 1/p
    return -(y_true * inv)

def log_loss(y_true, y_pred):
    eps = np.finfo(y_pred.dtype).eps  # machine precision for that type
    y_pred = np.clip(y_pred, eps, 1 - eps) # if lower than eps replaced with eps, if higher than 1-eps replaced with 1-eps
    return -xlogy(y_true, y_pred).sum() / y_pred.shape[0]

def log_loss_prime(y_true, y_pred):
    return - y_true / y_pred # TODO: if prob has a comp. near 0? (/0)

LOSSES = {
    'mse': mse,
    'mee': mee,
    'logloss': log_loss # logloss
}

LOSSES_DERIVATIVES = {
    'mse': mse_prime,
    'mee': mee_prime,
    'logloss': log_loss_prime # logloss_prime
}

EVALUATION_METRICS = {
    'mse': mse,
    'mee': mee,
    'logloss': log_loss, # logloss,
    'accuracy': accuracy_score
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

def mean_and_std(data):
    mean = np.mean(data)
    dev = np.std(data)
    return mean, dev

def normalize(data):
    mean, std = mean_and_std(data)
    return (data - mean) / std

def check_inputs():
    pass

def write_json(data: dict, file_path: str):
    with open(file_path, 'r+') as f:
        file_data = json.load(f)
        file_data['results'].append(data)
        f.seek(0)
        json.dump(file_data, f, indent = 4)

def read_json(file_path: str):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data['results']

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
    tr_epochs = len(tr_error)
    tr_epoch_vector = np.linspace(1, tr_epochs, tr_epochs)
    val_epochs = len(val_error)
    val_epoch_vector = np.linspace(1, tr_epochs, val_epochs)
    plt.figure()
    plt.plot(tr_epoch_vector, tr_error, "b",
             label="Training error", linewidth=1.5)
    plt.plot(val_epoch_vector, val_error, "r--",
             label="Validation error", linewidth=1.5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.grid()
    plt.title("Training and validation error on monks 1 dataset")
    fig_name = "ml_project_error_plot"
    plt.savefig(fig_name)


def accuracy_plot(tr_accuracy, val_accuracy):
    tr_epochs = len(tr_accuracy)
    val_epochs = len(val_accuracy)
    tr_epoch_vector = np.linspace(1, tr_epochs, tr_epochs)
    val_epoch_vector = np.linspace(1, tr_epochs, val_epochs)
    
    plt.figure()
    plt.plot(tr_epoch_vector, tr_accuracy, "b",
             label="Trainig score", linewidth=1.5)
    plt.plot(val_epoch_vector, val_accuracy, "r--",
             label="Validation score", linewidth=1.5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.grid()
    plt.title("Training and validation score on monks 1 dataset")
    fig_name = "ml_project_score_plot"
    plt.savefig(fig_name)

def regression2_plots(true, pred):
    import numpy as np
    from scipy.stats import gaussian_kde
    xy = np.vstack([true[:,0],  pred[:, 0]])
    z = gaussian_kde(xy)(xy)
    plt.scatter(true[:,0], pred[:, 0], c=z)
    plt.xlabel("True y1")
    plt.ylabel("Predicted y1")
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints)
    plt.show()

    xy = np.vstack([true[:,0], true[:,0]-pred[:, 0]])
    z = gaussian_kde(xy)(xy)
    plt.scatter(true[:,0], true[:,0]-pred[:, 0], c=z)
    plt.xlabel("True y1")
    plt.ylabel("True y1 - Pred y1")
    plt.axhline(y=0)
    plt.show()

    xy = np.vstack([true[:,1], pred[:, 1]])
    z = gaussian_kde(xy)(xy)
    plt.scatter(true[:,1], pred[:, 1], c=z)
    plt.xlabel("True y2")
    plt.ylabel("Predicted y2")
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints)
    plt.show()

    xy = np.vstack([true[:,1],true[:,1]-pred[:, 1]])
    z = gaussian_kde(xy)(xy)
    plt.scatter(true[:,1], true[:,1]-pred[:, 1], c=z)
    plt.xlabel("True y2")
    plt.ylabel("True y2 - Pred y2")
    plt.axhline(y=0)
    plt.show()

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    for i in range(10):
        axes[0].scatter(true[i,0], pred[i, 0])
        axes[0].annotate(str(i), (true[i,0], pred[i, 0]))
        xpoints = ypoints = axes[0].get_xlim()
        axes[0].plot(xpoints, ypoints)
        axes[0].set_xlabel("True y1")
        axes[0].set_ylabel("Predicted y1")
        axes[1].scatter(true[i,1], pred[i, 1])
        axes[1].annotate(str(i), (true[i, 1], pred[i, 1]))
        xpoints = ypoints = axes[1].get_xlim()
        axes[1].plot(xpoints, ypoints)
        axes[1].set_xlabel("True y2")
        axes[1].set_ylabel("Predicted y2")
    plt.show()