import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl

CUP_TRAIN_CSV_PATH = './datasets/ML-CUP22-TR.csv'
CUP_BLIND_TEST_CSV_PATH = './datasets/ML-CUP22-TS.csv'
CUP_BLIND_TEST_PATH = './datasets/CUP_BLIND_TS.pkl'
CUP_DEV_PATH = './datasets/CUP_DEV.pkl'
CUP_TEST_PATH = './datasets/CUP_TS.pkl'

def read_tr_cup(path): # TODO: valuta se cambiare nomi
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

def main():
    print("ciao")
    X_blind_test = read_ts_cup(CUP_BLIND_TEST_CSV_PATH)
    X_train, y_train = read_tr_cup(CUP_TRAIN_CSV_PATH)
    X_dev, X_test, y_dev, y_test = train_test_split(X_train, y_train, test_size=0.20, shuffle=True, random_state=0)

    file = open(CUP_BLIND_TEST_PATH, 'wb')
    pkl.dump(X_blind_test, file)
    file.close()

    dev_set={}
    dev_set['X_dev'] = X_dev
    dev_set['y_dev'] = y_dev
    file = open(CUP_DEV_PATH, 'wb')
    pkl.dump(dev_set, file)
    file.close()

    test_set={}
    test_set['X_test'] = X_test
    test_set['y_test'] = y_test
    file = open(CUP_TEST_PATH, 'wb')
    pkl.dump(test_set, file)
    file.close()

if __name__ == '__main__':
    main()