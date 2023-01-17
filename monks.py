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
    return data, targets

def plot_monks_curves(net, data_set_name):
    print("MSE train %f" % net.train_losses[net.best_epoch])
    print("MSE test %f" % net.val_losses[net.best_epoch])
    print("ACCURACY train %f" % net.train_scores[net.best_epoch])
    print("ACCURACY test %f" % net.val_scores[net.best_epoch])

    plt.figure()
    plt.plot(net.train_losses, label="Training", color="blue")
    plt.plot(net.val_losses, 'r--', label='Test')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.savefig("./monks_curves/%s_loss_curves.pdf" %data_set_name, bbox_inches="tight")

    plt.figure()
    plt.plot(net.train_scores, label="Training", color="blue")
    plt.plot(net.val_scores, 'r--', label="Test")
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.savefig("./monks_curves/%s_accuracy_curves.pdf" %data_set_name, bbox_inches="tight")

def run_monks(monks : str):
    print(f"Running {monks}")
    if monk == "monks-3reg":
        monk = "monks-3"
    
    MONKS_TRAIN_PATH = f"./datasets/{monks}.train"
    MONKS_TEST_PATH = f"./datasets/{monks}.test"

    X_train, y_train = read_monks(MONKS_TRAIN_PATH)
    X_test, y_test = read_monks(MONKS_TEST_PATH)

    config = read_csv_results("monks.csv")
    net = Network(**config)
    
    net.fit(X_train, y_train, X_test, y_test)
    plot_monks_curves(net, monks)
    
    print('accuracy_test = %f' % net.score(X_test, y_test, 'accuracy'))
    print('accuracy_train = %f' %net.train_scores[net.best_epoch])
    print('mse_train = %f' %net.train_losses[net.best_epoch])
    print('mse_test = %f' %net.val_losses[net.best_epoch])
    print('accuracy_test_internal = %f' %net.val_scores[net.best_epoch])


