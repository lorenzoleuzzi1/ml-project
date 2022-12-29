import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from network import Network
from utils import flatten_pred, error_plot, accuracy_plot
from cross_validation import cross_validation
from utils import error_plot, accuracy_plot
import matplotlib.pyplot as plt

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
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

X_train, y_train = read_monks(TRAIN_PATH)
X_test, y_test = read_monks(TEST_PATH)

import time
start = time.time()

net = Network(activation_out='tanh', classification=True, activation_hidden='tanh', epochs = 1000, batch_size = 32, 
    learning_rate = "fixed", learning_rate_init=0.05, nesterov=True, early_stopping=True, stopping_patience = 1000)
tr_errors, tr_accuracy, val_errors, val_accuracy = net.fit(X_train, y_train) 
pred = net.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=pred))

plt.plot(tr_errors, label="training", color="blue")
#plt.plot(val_errors, label= "validation", color="green")
plt.plot(val_accuracy, label="score",color="red")
plt.legend(loc="upper right")
plt.show()
end = time.time()
#print(end - start)

