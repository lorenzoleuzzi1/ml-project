import pandas as pd
from utils import error_plot, accuracy_plot
from network import Network

CUP_TRAIN_PATH = './datasets/ML-CUP22-TR.csv'
CUP_TEST_PATH = './datasets/ML-CUP22-TS.csv'

def read_tr_cup(path):
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    targets = data[data.columns[-2:]].to_numpy()
    data.drop(data.columns[-2:], axis=1, inplace=True)
    data = data.to_numpy()
    return (data, targets)

def read_ts_cup(path):
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy()
    return data

X_train, y_train = read_tr_cup(CUP_TRAIN_PATH)
X_test = read_ts_cup(CUP_TEST_PATH)

net = Network(activation_out='softmax', activation_hidden='softmax', epochs= 1000, batch_size=32, learning_rate = "linear_decay", learning_rate_init=0.05, nesterov=True)
all_train_errors, all_val_errors, tr_accuracy, val_accuracy = net.fit(X_train, y_train)
pred = net.predict(X_test)
#print(accuracy(y_pred=pred, y_true=y_test))
error_plot(all_train_errors, all_val_errors)
accuracy_plot(tr_accuracy, val_accuracy)