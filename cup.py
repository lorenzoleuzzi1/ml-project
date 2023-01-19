from network import Network
from utils import mse, mee #TODO: score
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

CUP_TRAIN_CSV_PATH = './datasets/ML-CUP22-TR.csv'
CUP_BLIND_TEST_CSV_PATH = './datasets/ML-CUP22-TS.csv'
CUP_BLIND_TEST_PATH = './datasets/CUP_BLIND_TS.pkl'
CUP_DEV_PATH = './datasets/CUP_DEV.pkl'
CUP_TEST_PATH = './datasets/CUP_TS.pkl'

def read_tr_cup(): # TODO: valuta se cambiare nomi
    data = pd.read_csv(CUP_TRAIN_CSV_PATH, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    targets = data[data.columns[-2:]].to_numpy()
    data.drop(data.columns[-2:], axis=1, inplace=True)
    data = data.to_numpy()
    return (data, targets)

def read_ts_cup():
    data = pd.read_csv(CUP_BLIND_TEST_CSV_PATH, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy()
    return data

def load_blind_test_cup():
    file = open(CUP_BLIND_TEST_PATH, 'rb')
    return pkl.load(file)

def load_dev_set_cup():
    file = open(CUP_DEV_PATH, 'rb')
    dev_set = pkl.load(file)
    return dev_set['X_dev'], dev_set['y_dev']

def load_internal_test_cup():
    file = open(CUP_TEST_PATH, 'rb')
    test_set = pkl.load(file)
    return test_set['X_test'], test_set['y_test']


def run_cup(config):
    #TODO: validation/test + curves
    print(f"Running cup with the following configuration:\n{config}")
    X_train, y_train = load_dev_set_cup()

    net = Network(**config)

    net.fit(X_train, y_train)
    X_test, y_test = load_internal_test_cup()
    pred = net.predict_outputs(X_test)
    print(mse(y_test, pred))
    print(mee(y_test, pred))
    # regression2_plots(y_test, pred)
    plt.plot(net.train_losses_reg)
    plt.plot(net.train_scores)
    plt.show()

