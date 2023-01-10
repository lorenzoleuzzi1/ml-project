"""from cup_parsing import load_dev_set_cup
from network import Network

X_dev, y_dev = load_dev_set_cup()
net = Network(activation_out='identity', classification=False, random_state=0)
net.fit(X_dev, y_dev)"""

from cup_parsing import load_dev_set_cup
from network import Network
from validation import k_fold_cross_validation
from sklearn.model_selection import ParameterGrid
import time

FILE_SCORES = 'topology_scores.csv'

X_dev, y_dev = load_dev_set_cup()

# 6 4 5 3 0 1 2
# [30, 30, 30] [30, 30] [60, 60] [10, 10] [30] [60] [120]
grid = ParameterGrid(
    {'activation_hidden': ['tanh'],
        'activation_out': ['identity'],
        'alpha': [0.5],
        'batch_size': [32],
        'classification': [False],
        'early_stopping': [False],
        'epochs': [500],
        'evaluation_metric': ['mee'],
        'hidden_layer_sizes': [[30], [60], [120], [10, 10], [30,30], [60, 60], [30,30,30]],
        'lambd': [0.0001],
        'learning_rate': ['linear_decay'],
        'learning_rate_init': [0.01],
        'loss': ['mse'],
        'metric_decrease_tol': [0.001],
        'nesterov': [True],
        'random_state': [None],
        'reinit_weights': [True],
        'stopping_patience': [5],
        'tau': [200],
        'tol': [0.0001],
        'validation_size': [0.1],
        'verbose': [False],
        'weights_dist': [None]
        }
    )

file_scores=open(FILE_SCORES, 'w')

i = 0
for parameters in grid:
    t = time.time()
    net = Network( 
        activation_out = parameters['activation_out'],
        classification = parameters['classification'],
        activation_hidden = parameters['activation_hidden'],
        hidden_layer_sizes =  parameters['hidden_layer_sizes'],
        loss = parameters['loss'],
        evaluation_metric = parameters['evaluation_metric'],
        epochs = parameters['epochs'],
        learning_rate = parameters['learning_rate'],
        learning_rate_init = parameters['learning_rate_init'],
        tau = parameters['tau'],
        batch_size = parameters['batch_size'],
        lambd = parameters['lambd'],
        alpha = parameters['alpha'],
        verbose = parameters['verbose'],
        nesterov = parameters['nesterov'],
        early_stopping = parameters['early_stopping'],
        stopping_patience = parameters['stopping_patience'],
        validation_size = parameters['validation_size'],
        tol = parameters['tol'],
        random_state = parameters['random_state'],
        reinit_weights = parameters['reinit_weights'],
        weights_dist = parameters['weights_dist'],
        metric_decrease_tol = parameters['metric_decrease_tol']
    )
    net.fit(X_dev, y_dev)
    
    scores = k_fold_cross_validation(net, X_dev, y_dev, 3)
    
    file_scores.write('----------------------------------------CONFIGURATION %d----------------------------------------' % i)
    file_scores.write('\n')
    file_scores.write(str(scores))
    file_scores.write('\n \n')
    
    print('configuration %d done. time = %f' %(i, time.time()-t))
    i += 1

# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# [1, 2, 3, 4] PRIMO FOLD
# [5, 6, 7, 8] SECONDO FOLD
# [9, 10, 11, 12] TERZO FOLD

# MODELLO1
# allenato su fold 1,2 testato su fold 3
	# shuffle di [1,2,3,4,5,6,7,8] => prima epoca [2,1,4,3,8,7,5,6]
# allenato su fold 1,3 testato su fold 2
	# shuffle di [1,2,3,4,9,10,11,12] => prima epoca [2,1,4,3,12,11, 9, 10]
# allenato su fold 2,3 testato su fold 1

# MODELLO2
# allenato su fold 1,2 testato su fold 3
	# shuffle di [1,2,3,4,5,6,7,8] => prima epoca [2,1,4,3,8,7,5,6]
# allenato su fold 1,3 testato su fold 2
# allenato su fold 2,3 testato su fold 1


# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# [1, 2, 3, 4] PRIMO FOLD
# [5, 6, 7, 8] SECONDO FOLD
# [9, 10, 11, 12] TERZO FOLD

# MODELLO1
# allenato su fold 1,2 testato su fold 3
	# shuffle di [1,2,3,4,5,6,7,8] => prima epoca [2,1,4,3,8,7,5,6]
# allenato su fold 1,3 testato su fold 2
	# shuffle di [1,2,3,4,9,10,11,12] => prima epoca [2,1,4,3,12,11, 9, 10]
# allenato su fold 2,3 testato su fold 1

# MODELLO2
# allenato su fold 1,2 testato su fold 3
	# shuffle di [1,2,3,4,5,6,7,8] => prima epoca [2,1,4,3,8,7,5,6]
# allenato su fold 1,3 testato su fold 2
# allenato su fold 2,3 testato su fold 1
