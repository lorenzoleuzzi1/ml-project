import pandas as pd
from utils import error_plot, accuracy_plot
from sklearn.model_selection import train_test_split
from network import Network
import matplotlib.pyplot as plt
from utils import regression2_plots

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
X_blind_test = read_ts_cup(CUP_TEST_PATH)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

net = Network(
    activation_out='identity',
    classification=False,
    activation_hidden='leaky_relu',
    loss='mse',
    evaluation_metric='mee',
    epochs=50,
    batch_size=32, 
    learning_rate = "fixed",
    learning_rate_init=0.001,
    nesterov=False,
    stopping_patience=30,
    early_stopping=True
    )

net.fit(X_train, y_train)
print(net.score(X_test, y_test))

#regression2_plots(y_test, pred)

#print(accuracy(y_pred=pred, y_true=y_test))
#plt.plot(net.train_losses, label="training loss", color="blue")
#plt.plot(net.train_scores, label="training score", color="green")
#plt.plot(net.val_losses, label="training loss", color="red")
#plt.show()