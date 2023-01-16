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

X_train1, y_train1 = read_monks(MONKS1_TRAIN_PATH)
X_test1, y_test1 = read_monks(MONKS1_TEST_PATH)

net1 = Network(
    hidden_layer_sizes=[3],
    activation_out='logistic',
    classification=True,
    activation_hidden='tanh',
    epochs = 400, # diminuire!
    lambd=0,
    learning_rate = "fixed", # ?, tau
    batch_size=16,
    learning_rate_init=0.01,
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
    weights_dist=None,
    weights_bound=None
    )

"""activation_out='tanh',
    classification=True,
    activation_hidden='tanh',
    epochs = 100, 
    batch_size = 1,
    lambd=0,
    learning_rate = "fixed",
    learning_rate_init=0.001,
    #nesterov=True, 
    early_stopping=True,
    evaluation_metric='accuracy',
    verbose=True,
    loss='mse',
    validation_frequency=1,
    validation_size=0.1,
    tol=1e-4,
    random_state=0"""

net1.fit(X_train1, y_train1, X_test1, y_test1)

print("MSE train %f" % net1.train_losses[net1.best_epoch])
print("MSE test %f" % net1.val_losses[net1.best_epoch])
print("ACCURACY train %f" % net1.train_scores[net1.best_epoch])
print("ACCURACY test %f" % net1.val_scores[net1.best_epoch])

plt.plot(net1.train_losses, label="Training", color="blue")
plt.plot(net1.val_losses, 'r--', label='Test')
plt.axvline(x=net1.best_epoch, ymin=0, ymax=net1.val_losses[net1.best_epoch], color='black', linestyle='dashed', linewidth=0.8)
plt.legend()
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.savefig("monks1_loss_curves.pdf", bbox_inches="tight")

plt.figure()
plt.plot(net1.train_scores, label="Training", color="blue")
plt.plot(net1.val_scores, 'r--', label="Test")
plt.axvline(x=net1.best_epoch, ymin=0, ymax=net1.val_scores[net1.best_epoch], color='black', linestyle='dashed', linewidth=0.8)
plt.legend()
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("monks1_accuracy_curves.pdf", bbox_inches="tight")