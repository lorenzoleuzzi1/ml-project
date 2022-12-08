import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_linnerud, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from network import Network
from sklearn.metrics import accuracy_score

# utility function
def flatten_pred(pred):
    flattened_pred = np.empty(len(pred))
    for i in range(len(pred)):
        if pred[i][0][0] > 0:
            flattened_pred[i] = 1
        else:
            flattened_pred[i] = -1
    return flattened_pred

# DATASET infos: https://scikit-learn.org/stable/datasets/toy_dataset.html

X, y = load_breast_cancer(return_X_y=True) # replace with one of the load utility imported above
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

net = Network(activation_out='tanh') # NOTE: identity for regression
net.fit(X_train, y_train)
y_pred = net.predict(X_test)
y_flatten_pred = flatten_pred(y_pred) # NOTE: keep attention on target encoding / net output encoding
print(accuracy_score(y_true=y_test, y_pred=y_flatten_pred))