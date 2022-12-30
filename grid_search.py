from sklearn.model_selection import ParameterGrid
from validation import cross_validation
from network import Network

grid = ParameterGrid(
    {   
        #---fixed TODO: ricerca ad occhio per migliori
        'activation_out': ['tanh'],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10]],
        'loss': ['mse'],
        'epochs': [200],
        'learning_rate_init': [0.001], 
        'tau' : [200],
        'tol' : [0.0005],
        #---to tune
        'learning_rate': ['fixed', 'linear_decay'],
        'batch_size': [1, 32, 1.0],
        'lambd': [0.0001, 0.001, 0.01, 0.1],
        'alpha': [0.5, 0.7, 0.9]
    }
)

K = 3

print(len(grid))





