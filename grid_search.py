from sklearn.model_selection import ParameterGrid
from network import Network

grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic', 'relu'], #identity?
        'hidden_layer_sizes': [[10], [10, 10], [50], [100, ], [50, 50], [100, 100]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500, 1000],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0, 0.0001, 0.001, 0.01, 0.1], # rivedi dopo aver capito come scalare lamda, quando alto serve early stopping?
        'alpha': [0.5, 0.7, 0.9],
        'verbose': [False],
        'nesterov' : [True, False],
        'early_stopping': [True, False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [1],
        'learning_rate_init': [0.0001, 0.0005, 0.001]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic', 'relu'], #identity?
        'hidden_layer_sizes': [[10], [10, 10], [50], [100, ], [50, 50], [100, 100]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500, 1000],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0, 0.0001, 0.001, 0.01, 0.1], # rivedi dopo aver capito come scalare lamda, quando alto serve early stopping?
        'alpha': [0.5, 0.7, 0.9],
        'verbose': [False],
        'nesterov' : [True, False],
        'early_stopping': [True, False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [1.0],
        'learning_rate_init': [0.1, 0.01]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic', 'relu'], #identity?
        'hidden_layer_sizes': [[10], [10, 10], [50], [100], [50, 50], [100, 100]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500, 1000],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0, 0.0001, 0.001, 0.01, 0.1], # rivedi dopo aver capito come scalare lamda, quando alto serve early stopping?
        'alpha': [0.5, 0.7, 0.9],
        'verbose': [False],
        'nesterov' : [True, False],# TODO togliere ????
        'early_stopping': [True, False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32, 256], #faremo 5 o 3 fold cross alidation 1500/5*4 => 1200 1500/3*2=1000, 
        'learning_rate_init': [0.01, 0.001]
    }
]
)

print(len(grid))