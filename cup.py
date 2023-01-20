from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import json

def load_train_cup(): 
    CUP_TRAIN_CSV_PATH = './datasets/ML-CUP22-TR.csv'
    data = pd.read_csv(CUP_TRAIN_CSV_PATH, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    targets = data[data.columns[-2:]].to_numpy()
    data.drop(data.columns[-2:], axis=1, inplace=True)
    data = data.to_numpy()
    return (data, targets)

def load_test_cup():
    CUP_BLIND_TEST_CSV_PATH = './datasets/ML-CUP22-TS.csv'
    data = pd.read_csv(CUP_BLIND_TEST_CSV_PATH, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy()
    return data

def load_blind_test_cup():
    CUP_BLIND_TEST_PATH = './datasets/CUP_BLIND_TS.pkl'
    file = open(CUP_BLIND_TEST_PATH, 'rb')
    return pkl.load(file)

def load_dev_set_cup():
    CUP_DEV_PATH = './datasets/CUP_DEV.pkl'
    file = open(CUP_DEV_PATH, 'rb')
    dev_set = pkl.load(file)
    return dev_set['X_dev'], dev_set['y_dev']

def load_internal_test_cup():
    CUP_TEST_PATH = './datasets/CUP_TS.pkl'
    file = open(CUP_TEST_PATH, 'rb')
    test_set = pkl.load(file)
    return test_set['X_test'], test_set['y_test']

def plot_cup_curves(net, name):
    plt.figure()
    plt.plot(net.train_losses, label="Development", color="blue", linewidth=1.2)
    plt.plot(net.val_losses, 'r--', label='Internal Test', linewidth=1.2)
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel(f"Loss ({net.loss.upper()})")
    plt.savefig(f"./plots/cup/{name}_loss.pdf", bbox_inches="tight")

    plt.figure()
    plt.plot(net.train_scores, label="Development", color="blue", linewidth=1.2)
    plt.plot(net.val_scores, 'r--', label="Internal Test", linewidth=1.2)
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel(f"Score ({net.evaluation_metric.upper()})")
    plt.savefig(f"./plots/cup/{name}_score.pdf", bbox_inches="tight")

def run_cup(config):
    print(f"Running cup with the following configuration:\n{config}")
    X_train, y_train = load_dev_set_cup()
    X_test, y_test = load_internal_test_cup()
    net = NeuralNetwork(**config)
    net.verbose = True
    
    net.fit(X_train, y_train, X_test, y_test)
    
    scores = net.score(X_test, y_test, ["mse", "mee"])
    print(scores)
    plot_cup_curves(net, "cup")

def best_models_assessment(configs):
    X_train, y_train = load_dev_set_cup()
    X_test, y_test = load_internal_test_cup()
    scores = []
    print(f"Assessing {len(configs)} models")

    # loop through best configurations
    for i, config in enumerate(configs):
        print(f"Running cup with the following configuration:\n{config}")
        X_train, y_train = load_dev_set_cup()
        net = NeuralNetwork(**config)

        net.fit(X_train, y_train)
        score = net.score(X_test, y_test, ['mse', 'mee'])
        print(score)
        scores.append({f"model_{i}" : score})
    
    return scores
   
