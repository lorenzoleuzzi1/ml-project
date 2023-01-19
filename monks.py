import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
 
def read_train_monks(monks_name, one_hot_encoding=True):
    MONKS_TRAIN_PATH = f"./datasets/{monks_name[:7]}.train"
    data = pd.read_csv(MONKS_TRAIN_PATH, sep=" ", header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy() 
    if one_hot_encoding:
        data = OneHotEncoder().fit_transform(data).toarray()
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

def read_test_monks(monks_name, one_hot_encoding=True):
    MONKS_TEST_PATH = f"./datasets/{monks_name[:7]}.test"
    data = pd.read_csv(MONKS_TEST_PATH, sep=" ", header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy() 
    if one_hot_encoding:
        data = OneHotEncoder().fit_transform(data).toarray() 
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

def plot_monks_curves(net, name):
    plt.figure()
    plt.plot(net.train_losses, label="Training", color="blue", linewidth=1.2)
    plt.plot(net.val_losses, 'r--', label='Test', linewidth=1.2)
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel(f"Loss ({net.loss.upper()})")
    plt.savefig(f"./plots/monks/{name}_loss.pdf", bbox_inches="tight")

    plt.figure()
    plt.plot(net.train_scores, label="Training", color="blue", linewidth=1.2)
    plt.plot(net.val_scores, 'r--', label="Test", linewidth=1.2)
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel(f"Score ({net.evaluation_metric.upper()})")
    plt.ylim(0, 1.05)
    plt.savefig(f"./plots/monks/{name}_score.pdf", bbox_inches="tight")

def run_monks(config):
    monks_name = config.get("name")
    config.pop("name")
    
    print(f"Running {monks_name} with the following configuration:\n{config}")
    
    X_train, y_train = read_train_monks(monks_name)
    X_test, y_test = read_test_monks(monks_name)
    net = NeuralNetwork(**config)
    net.activation_out = 'tanh'
    net.fit(X_train, y_train, X_test, y_test)
    
    scores = net.score(X_test, y_test, ["accuracy", "mse", "mee"])
    print(scores)
    plot_monks_curves(net, monks_name)
    


