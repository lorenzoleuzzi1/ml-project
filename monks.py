import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from network import Network
from utils import linear_decay, error_plot, accuracy_plot
from cross_validation import *


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

# utility temporary function
def flatten_pred(pred):
    flattened_pred = np.empty(len(pred))
    for i in range(len(pred)):
        if pred[i][0][0] > 0:
            flattened_pred[i] = 1
        else:
            flattened_pred[i] = -1
    return flattened_pred

X_train, y_train = read_monks(TRAIN_PATH)
X_test, y_test = read_monks(TEST_PATH)

"""net = Network(activation_out='tanh', epochs= 300, batch_size=32, learning_rate_fun=linear_decay(200, 0.1))
all_train_errors, all_val_errors, tr_accuracy, val_accuracy = net.fit(X_train, y_train, X_test, y_test)
pred = net.predict(X_test)

error_plot(all_train_errors, all_val_errors)
accuracy_plot(tr_accuracy, val_accuracy)"""

# cross validation
avg_tr_error, avg_val_error, avg_tr_accuracy, avg_val_accuracy, accuracy, pred = cross_validation(X_train, y_train, X_test, k=5, epochs=300)

error_plot(avg_tr_error, avg_val_error)
accuracy_plot(avg_tr_accuracy, avg_val_accuracy)

for p, y in zip(pred, y_test):
    print("pred: {} expected: {}".format(p,y))

flattened_pred = flatten_pred(pred)
print(accuracy_score(y_true=y_test, y_pred=flattened_pred))