import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import accuracy
from network import Network
from utils import flatten_pred
from cross_validation import cross_validation


MONKS1_TRAIN_PATH = './datasets/monks-1.train'
MONKS1_TEST_PATH = './datasets/monks-1.test'
MONKS2_TRAIN_PATH = './datasets/monks-2.train'
MONKS2_TEST_PATH = './datasets/monks-2.test'
MONKS3_TRAIN_PATH = './datasets/monks-3.train'
MONKS3_TEST_PATH = './datasets/monks-3.test'

TRAIN_PATH = MONKS1_TRAIN_PATH
TEST_PATH = MONKS1_TEST_PATH

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
    return (data, targets)

X_train, y_train = read_monks(TRAIN_PATH)
X_test, y_test = read_monks(TEST_PATH)

# cross validation
network = Network('tanh', True, batch_size=1, learning_rate_init=0.002, evaluation_metric='accuracy', verbose=True)
cross_validation(network, X_train, y_train, X_test, y_test, k_out=3, k_inn=3, nested=True)

# net = Network(activation_out='tanh', classification=True, activation_hidden='tanh', epochs= 200, batch_size=1, 
# learning_rate = "linear_decay", learning_rate_init=0.002, nesterov=True, early_stopping=True)
# all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train) # without early stopping
# # #all_train_errors, all_val_errors, tr_accuracy, val_accuracy = net.fit(X_train, y_train) # with early stopping
# pred = net.predict(X_test)
# print(accuracy(y_pred=pred, y_true=y_test))
#error_plot(all_train_errors, all_val_errors) # with early stopping
#accuracy_plot(tr_accuracy, val_accuracy) # with early stopping

