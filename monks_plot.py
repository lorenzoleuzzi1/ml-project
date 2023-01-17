import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from validation import k_fold_cross_validation, grid_search_cv
from network import Network
from utils import error_plot, accuracy_plot
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

MONKS1_TRAIN_PATH = './datasets/monks-1.train'
MONKS1_TEST_PATH = './datasets/monks-1.test'
MONKS2_TRAIN_PATH = './datasets/monks-2.train'
MONKS2_TEST_PATH = './datasets/monks-2.test'
MONKS3_TRAIN_PATH = './datasets/monks-3.train'
MONKS3_TEST_PATH = './datasets/monks-3.test'

def read_monks(path, one_hot_encoding=True, target_rescaling=True):
    data = pd.read_csv(path, sep=" ", header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy() # int 64
    if one_hot_encoding:
        data = OneHotEncoder().fit_transform(data).toarray() # float 64
    if target_rescaling:
        targets[targets == 0] = -1 # int 64
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

def plot_monks_curves(net, data_set_name):
    print("MSE train %f" % net.train_losses[net.best_epoch])
    print("MSE test %f" % net.val_losses[net.best_epoch])
    print("ACCURACY train %f" % net.train_scores[net.best_epoch])
    print("ACCURACY test %f" % net.val_scores[net.best_epoch])

    plt.figure()
    plt.plot(net.train_losses, label="Training", color="blue")
    plt.plot(net.val_losses, 'r--', label='Test')
    plt.vlines(x=net.best_epoch, ymin=0, ymax=net.val_losses[net.best_epoch], color='black', linestyle='dashed', linewidth=0.8)
    plt.plot(net.best_epoch, 0.015, 'k*', label='Backtracked epoch')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.ylim(0, max(max(net.train_losses), max(net.val_losses)))
    plt.savefig("%s_loss_curves.pdf" %data_set_name, bbox_inches="tight")

    plt.figure()
    plt.plot(net.train_scores, label="Training", color="blue")
    plt.plot(net.val_scores, 'r--', label="Test")
    plt.vlines(x=net.best_epoch, ymin=0, ymax=net.val_scores[net.best_epoch], color='black', linestyle='dashed', linewidth=0.8)
    plt.plot(net.best_epoch, 0.015, 'k*', label='Backtracked epoch')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig("%s_accuracy_curves.pdf" %data_set_name, bbox_inches="tight")

# MONKS1
X_train1, y_train1 = read_monks(MONKS1_TRAIN_PATH)
X_test1, y_test1 = read_monks(MONKS1_TEST_PATH)

net1 = Network( #TODO: mettere parametri buoni
    hidden_layer_sizes=[2],
    activation_out='tanh',
    classification=True,
    activation_hidden='tanh',
    epochs = 200, # diminuire!
    lambd=0,
    learning_rate = "fixed", # ?, tau
    batch_size=16, # 4
    learning_rate_init=0.005, # 0.001 0.01
    alpha=0.9,
    nesterov=False,
    early_stopping=False,
    evaluation_metric='accuracy',
    verbose=True,
    loss='mse',
    tol=0.00001,
    metric_decrease_tol=0.000001,
    random_state=None,
    stopping_patience=30,
    reinit_weights=True,
    weights_dist='uniform',
    weights_bound=0.7
    )

net1.fit(X_train1, y_train1, X_test1, y_test1)
plot_monks_curves(net1, 'monks1')

# TODO: fare per tutte e 4 i monks
