from sklearn.model_selection import ParameterGrid
from network import Network

# Currently randomly set for test purposes
grid = ParameterGrid([
    {
        'learning_rate': [0.001, 0.01],
        'alpha': [0.1, 0.2],
        'lambd': [0.01, 0.1, 0.2],
        'epochs': [100, 1000],
        'batch_size': [1],
        'hidden_layer_sizes': [[3], [3, 3]],
        'activation_hidden': ['tanh', 'logistic'],
        'activation_out': ['tanh', 'logistic'],
        'loss': ['mse', 'rmse']
    }
])
# TODO: add
# 'tau': []
# 'init_learning_rate': []

for params in grid:
    print(params)
    print("----------")
    net = Network(**params)
    # fit & validation