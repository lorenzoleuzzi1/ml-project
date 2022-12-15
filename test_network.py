import numpy as np
from sklearn.model_selection import ParameterGrid
from network import Network

###### TESTED RUN TIME ERRORS ONLY ######

grid = ParameterGrid([
    {
        'activation_out': ['identity', 'relu', 'leaky_relu', 'logistic', 'tanh', 'softplus'],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[3], [3, 3]],
        'learning_rate': ['fixed', 'linear_decay'],
        'batch_size': [1, 2]
    }
])

###### SINGLE TARGET ######

# training data
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([0, 1, 1, 0])
# test data
x_test = np.array([[0,0]])

for params in grid:
    net = Network(**params)
    net.fit(x_train, y_train, x_train, y_train)
    out = net.predict(x_test)


###### MULTI TARGET ######
"""
# training data
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0,0], [0,1], [1,0], [1,1]])
# test data
x_test = np.array([[0,0]])

for params in grid:
    net = Network(**params)
    net.fit(x_train, y_train, x_train, y_train)
    out = net.predict(x_test)
"""