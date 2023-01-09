from sklearn.model_selection import ParameterGrid
import numpy as np

grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
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
        'stopping_patience': [30],
        'validation_size': [0.1],
        'validation_frequency': [1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
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
        'stopping_patience': [30],
        'validation_size': [0.1],
        'validation_frequency': [1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
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
        'stopping_patience': [30],
        'validation_size': [0.1],
        'validation_frequency': [1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [1],
        'learning_rate_init': [0.0001, 0.0005, 0.001]
    }
    ]
)

print(len(grid))
#print(grid)
grid_splitted = np.array_split(grid, 3)
grid_splitted[0] # 
# grid_search_cv(grid, X_train, y_train, k)


# TODO: 
# lambda range (doppi)?
# parametri indipendnti
