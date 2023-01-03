import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_linnerud, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from network import Network
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#from utils import mse_score
from utils import mse

# DATASET infos: https://scikit-learn.org/stable/datasets/toy_dataset.html

################## MULTI CLASSIFICATION TEST ##################
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

scikit_net = MLPClassifier(
    hidden_layer_sizes=(5, 5),
    activation='tanh', # for hidden layers
    solver='sgd', 
    alpha=0.0001, # our lambd
    batch_size=32, 
    learning_rate='constant', 
    learning_rate_init=0.001, 
    max_iter=500,
    shuffle=True, 
    tol=0.0005,
    momentum=0.9, # our alpha
    nesterovs_momentum=False,
    early_stopping=False
    )
scikit_net.fit(X_train, y_train)
plt.plot(scikit_net.loss_curve_, label="training loss", color="blue")
plt.legend(loc="upper right")
plt.show()
pred = scikit_net.predict(X_test)
print(accuracy_score(y_pred=pred, y_true=y_test))

y_train = y_train.reshape(y_train.shape[0], 1)
net = Network(
	activation_out='softmax', 
	activation_hidden='tanh',
	classification=True,
	early_stopping=False,
	batch_size=32,
	epochs=500,
	hidden_layer_sizes=[5, 5],
	loss='logloss',
	evaluation_metric='mse',
	learning_rate='fixed',
	learning_rate_init=0.001
	)
tr_loss, tr_score = net.fit(X_train, y_train)
pred = net.predict(X_test)
print(accuracy_score(y_pred=pred, y_true=y_test))

plt.plot(tr_loss, label="training loss", color="blue")
plt.plot(tr_score, label= "training score", color="green")
plt.legend(loc="upper right")
plt.show()

################## MULTI REGRESSOR TEST ##################
"""X, y = load_diabetes(return_X_y=True)
X_new = X[:,1:]
y2 = X[:,0] # first feature as additional target
y_new = np.column_stack((y,y2))
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=42)
net = Network(activation_out='identity', classification=False, hidden_layer_sizes=[3, 3])
net.fit(X_train, y_train)
y_pred = net.predict(X_test)
print(mse(y_pred, y_test))"""

################## CLASSIFICATION TEST ##################
"""X, y = load_breast_cancer(return_X_y=True)
y = y.reshape(y.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
net = Network(activation_out='tanh', classification=True, hidden_layer_sizes=[3, 3])
net.fit(X_train, y_train)
y_pred = net.predict(X_test)
y_flatten_pred = f_pred(y_pred)
print(accuracy_score(y_true=y_test, y_pred=y_flatten_pred))"""