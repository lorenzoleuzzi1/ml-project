import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_linnerud, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from network import Network
from sklearn.metrics import accuracy_score
from utils import f_pred
from utils import mse_score

# DATASET infos: https://scikit-learn.org/stable/datasets/toy_dataset.html

################## MULTI REGRESSOR TEST ##################
X, y = load_diabetes(return_X_y=True)
X_new = X[:,1:]
y2 = X[:,0] # first feature as additional target
y_new = np.column_stack((y,y2))
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=42)
net = Network(activation_out='identity', hidden_layer_sizes=[3, 3])
net.fit(X_train, y_train, X_test, y_test)
y_pred = net.predict(X_test)
print(mse_score(y_pred, y_test))

################## CLASSIFICATION TEST ##################
"""X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
net = Network(activation_out='tanh', hidden_layer_sizes=[3, 3])
net.fit(X_train, y_train, X_test, y_test)
y_pred = net.predict(X_test)
y_flatten_pred = f_pred(y_pred)
print(accuracy_score(y_true=y_test, y_pred=y_flatten_pred))"""