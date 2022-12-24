import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import accuracy
from network import Network

MONKS1_TRAIN_PATH = './datasets/monks-1.train'
MONKS1_TEST_PATH = './datasets/monks-1.test'

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

############## TEST 1 ##############
net = Network(activation_out='tanh', classification=True, activation_hidden='tanh', epochs= 1000, batch_size=1, 
learning_rate = "linear_decay", learning_rate_init=0.002, nesterov=False, early_stopping=True, reinit_weights=False, random_state=0, verbose=False)
all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train) # with early stopping, otw returns 2 args
pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))

weights, bias = net.get_weights()
print("weights")
print(weights)
print("bias")
print(bias)

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

net.set_weights(init_weights, init_bias)
all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train)

weights, bias = net.get_weights()
print("weights")
print(weights)
print("bias")
print(bias)

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))

############## TEST 2 ##############
"""net = Network(activation_out='tanh', classification=True, activation_hidden='tanh', epochs= 1000, batch_size=1, 
learning_rate = "linear_decay", learning_rate_init=0.002, nesterov=False, early_stopping=True, reinit_weights=True, random_state=0, verbose=False)
all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train) # with early stopping, otw returns 2 args
pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))

weights, bias = net.get_weights()
print("weights")
print(weights)
print("bias")
print(bias)

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train)

weights, bias = net.get_weights()
print("weights")
print(weights)
print("bias")
print(bias)

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))"""

############## TEST 3 ##############
"""net = Network(activation_out='tanh', classification=True, activation_hidden='tanh', epochs= 1000, batch_size=1, 
learning_rate = "linear_decay", learning_rate_init=0.002, nesterov=False, early_stopping=True, reinit_weights=True, random_state=None, verbose=False)
all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train) # with early stopping, otw returns 2 args
pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))

weights, bias = net.get_weights()
print("weights")
print(weights)
print("bias")
print(bias)

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train)

weights, bias = net.get_weights()
print("weights")
print(weights)
print("bias")
print(bias)

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))"""

############## TEST 4 ##############
"""net = Network(activation_out='tanh', classification=True, activation_hidden='tanh', epochs= 1000, batch_size=1, 
learning_rate = "linear_decay", learning_rate_init=0.002, nesterov=False, early_stopping=True, reinit_weights=False, random_state=None, verbose=False)
all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train) # with early stopping, otw returns 2 args
pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

net.set_weights(init_weights, init_bias)
all_train_errors, tr_accuracy, _, _ = net.fit(X_train, y_train)

init_weights, init_bias = net.get_init_weights()
print("init weights")
print(init_weights)
print("init bias")
print(init_bias)

pred = net.predict(X_test)
print(accuracy(y_pred=pred, y_true=y_test))"""