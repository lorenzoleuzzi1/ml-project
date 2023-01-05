import pandas as pd
from utils import error_plot, accuracy_plot
from sklearn.model_selection import train_test_split
from network import Network
import matplotlib.pyplot as plt

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
    activation_hidden='tanh',
    loss='mse',
    epochs=100,
    batch_size=32, 
    learning_rate = "fixed",
    learning_rate_init=0.001,
    nesterov=False,
    stopping_patience=10,
    early_stopping=False
    )
net.fit(X_train, y_train)
pred = net.predict(X_test)

plt.scatter(y_test[:,0], pred[:, 0])
plt.xlabel("True y1")
plt.ylabel("Predicted y1")
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints)
plt.show()

plt.scatter(y_test[:,0], y_test[:,0]-pred[:, 0])
plt.xlabel("True y1")
plt.ylabel("True y1 - Pred y1")
plt.axhline(y=0)
plt.show()

plt.scatter(y_test[:,1], pred[:, 1])
plt.xlabel("True y2")
plt.ylabel("Predicted y2")
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints)
plt.show()

plt.scatter(y_test[:,1], y_test[:,1]-pred[:, 1])
plt.xlabel("True y2")
plt.ylabel("True y2 - Pred y2")
plt.axhline(y=0)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
for i in range(10):
    axes[0].scatter(y_test[i,0], pred[i, 0])
    axes[0].annotate(str(i), (y_test[i,0], pred[i, 0]))
    xpoints = ypoints = axes[0].get_xlim()
    axes[0].plot(xpoints, ypoints)
    axes[0].set_xlabel("True y1")
    axes[0].set_ylabel("Predicted y1")
    axes[1].scatter(y_test[i,1], pred[i, 1])
    axes[1].annotate(str(i), (y_test[i, 1], pred[i, 1]))
    xpoints = ypoints = axes[1].get_xlim()
    axes[1].plot(xpoints, ypoints)
    axes[1].set_xlabel("True y2")
    axes[1].set_ylabel("Predicted y2")
plt.show()

#print(accuracy(y_pred=pred, y_true=y_test))
#plt.plot(net.train_losses, label="training loss", color="blue")
#plt.plot(net.train_scores, label="training score", color="green")
#plt.plot(net.val_losses, label="training loss", color="red")
#plt.show()