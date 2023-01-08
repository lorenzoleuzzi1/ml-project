from sklearn.model_selection import ParameterGrid
from network import Network

grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'relu'],
        'hidden_layer_sizes': [[30], [120], [60,60]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200, 800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001, 0.01],
        'alpha': [0, 0.8],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'relu'],
        'hidden_layer_sizes': [[30], [120], [60,60]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200, 800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001, 0.01],
        'alpha': [0.5, 0.8],
        'verbose': [False],
        'nesterov' : [True],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'relu'],
        'hidden_layer_sizes': [[30], [120], [60,60]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200, 800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.00001, 0.0001, 0.001, 0.01],
        'alpha': [0, 0.8],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [1],
        'learning_rate_init': [0.0001, 0.0005, 0.001]
    }
    ]
)

print(len(grid))