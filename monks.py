import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from validation import read_csv_results
from network import Network

def read_monks(path, one_hot_encoding=True):
    data = pd.read_csv(path, sep=" ", header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy() # int 64
    if one_hot_encoding:
        data = OneHotEncoder().fit_transform(data).toarray() # float 64
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

def plot_monks_curves(net, data_set_name):
    plt.figure()
    plt.plot(net.train_losses, label="Training", color="blue")
    plt.plot(net.val_losses, 'r--', label='Test')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.savefig("./monks_curves/%s_loss_curves.pdf" %data_set_name, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(net.train_scores, label="Training", color="blue")
    plt.plot(net.val_scores, 'r--', label="Test")
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Score (Accuracy)")
    plt.ylim(0, 1.05)
    plt.savefig("./monks_curves/%s_accuracy_curves.pdf" %data_set_name, bbox_inches="tight")
    plt.show()

def run_monks(config):
    monks_name = config.get("name")
    config.pop("name")
    print(f"Running {monks_name} with the following configuration:\n{config}")

    MONKS_TRAIN_PATH = f"./datasets/{monks_name[:7]}.train"
    MONKS_TEST_PATH = f"./datasets/{monks_name[:7]}.test"
    
    X_train, y_train = read_monks(MONKS_TRAIN_PATH)
    X_test, y_test = read_monks(MONKS_TEST_PATH)
    net = Network(**config)
    # net.batch_size = 4
    # net.lambd = 0.0001
    net.fit(X_train, y_train, X_test, y_test)
    scores = net.score(X_test, y_test, ["accuracy", "mse", "mee"])
    print(scores)
    plot_monks_curves(net, monks_name)
    


