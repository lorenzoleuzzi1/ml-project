from sklearn.model_selection import ParameterGrid
from network import Network

n_samples = 50 # random value

# Temporary set as follows for test purposes
grid = ParameterGrid([
    {
        'activation_out': ['tanh', 'logistic'],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[10], [3, 3]],
        'loss': ['mse', 'rmse'],
        'epochs': [100, 200, 500, 1000],
        'learning_rate_schedule': ['fixed'],
        'learning_rate_init': [0.001, 0.05, 0.01, 0.1, 0.5],
        'batch_size': [1, n_samples/4, n_samples/2, n_samples], # TODO: if real?
        'lambd': [0.0001, 0.001, 0.01, 0.1],
        'nesterov': [True, False],
        'alpha': [0.5, 0.7, 0.9]
    },
    {
        'activation_out': ['tanh', 'logistic'],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[10], [3, 3]],
        'loss': ['mse', 'rmse'],
        'epochs': [100,200, 500, 1000],
        'learning_rate_schedule': ['linear_decay'],
        'learning_rate_init': [0.001, 0.05, 0.01, 0.1, 0.5],
        'tau': [100], # depends on epochs (must be less)
        'batch_size': [1, n_samples/4, n_samples/2, n_samples], # TODO: if real?
        'lambd': [0.0001, 0.001, 0.01, 0.1],
        'nesterov': [True, False],
        'alpha': [0.5, 0.7, 0.9],
    }
])

for params in grid:
    print(params)
    print("----------")
    net = Network(**params)
    # fit & validation