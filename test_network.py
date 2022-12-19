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
        'batch_size': [1.0]
    }
])

###### SINGLE TARGET ######
"""
# training data
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([0, 1, 1, 0])
# test data
x_test = np.array([[0,0]])

for params in grid:
    net = Network(**params)
    net.fit(x_train, y_train, x_train, y_train)
    out = net.predict(x_test)
"""

###### MULTI TARGET ######

# TODO: con i primi 4 dati il validation set è vuoto!!!
# training data
x_train = np.array([[0,0], [0,1], [1,0], [1,1], [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9]])
y_train = np.array([[0,0], [0,1], [1,0], [1,1], [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9]])
# test data
x_test = np.array([[0,0]])
"""
for params in grid:
    net = Network(**params)
    net.fit(x_train, y_train)
    out = net.predict(x_test)
    """
    
net = Network(activation_out='tanh', epochs= 1000, batch_size=2, learning_rate = "linear_decay", learning_rate_init=0.05, nesterov=True)
net.fit(x_train, y_train)
print(net.predict(x_test))