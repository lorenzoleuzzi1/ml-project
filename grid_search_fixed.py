from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from network import Network
from validation import k_fold_cross_validation

CUP_TRAIN_PATH = './datasets/ML-CUP22-TR.csv'
CUP_TEST_PATH = './datasets/ML-CUP22-TS.csv'
FILE_PARAMETERS = 'parameters.csv'
FILE_SCORES = 'scores.csv'

def read_tr_cup(path):
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    targets = data[data.columns[-2:]].to_numpy()
    data.drop(data.columns[-2:], axis=1, inplace=True)
    data = data.to_numpy()
    return (data, targets)

def read_ts_cup(path):
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy()
    return data

""" 
DEFAULT VALUE:
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],                  range: ['tanh', 'logistic', 'relu']
        'hidden_layer_sizes': [[10, 10]],               range: [[10], [20], [10, 10], [30,30] [50], [100], [50, 50], [100, 100]]
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],                                range: [100, 500, 1000]
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],              range: ['fixed', 'linear_decay']
        'lambd': [0.0001],                              range: [0, 0.0001, 0.001, 0.01, 0.1]
        'alpha': [0],                                   range: [0], [0.5, 0.7, 0.9]
        'verbose': [False],
        'nesterov': [False],                            range: [False], [True, False] with 'alpha'
        'early_stopping': [False],                      range: [True, False]
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],                              range: [1], [1.0], [32, 256] with 'learning_rate_init'
        'learning_rate_init': [0.01]                     range: [0.0001, 0.0005, 0.001], [0.1, 0.01], [0.01, 0.001]
    }
"""

fixed_grid = ParameterGrid([
    # ---------------------'activation_hidden'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh', 'logistic', 'relu'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'hidden_layer_sizes'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10], [20], [30, 30], [50], [100], [50, 50], [100, 100]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'epochs'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [200, 1000],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'learning_rate'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.0001],
        'alpha': [0],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'lambd'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0, 0.001, 0.01, 0.1],
        'alpha': [0],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'alpha'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0.5, 0.7, 0.9],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'nesterov'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0.5],
        'verbose': [False],
        'nesterov': [True],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'early_stopping'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0.5],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [True],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [32],
        'learning_rate_init': [0.01]
    },
    # ---------------------'batch_size'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0.5],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [1, 32, 256],
        'learning_rate_init': [0.001]
    },
    # ---------------------'learning_rate_init'---------------------
    {
        'activation_out': ['identity'],
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0.5],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
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
        'classification': [False],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[10, 10]],
        'loss': ['mse'],
        'evaluation_metric': ['mee'],
        'epochs': [500],
        'tau': [200],
        'tol': [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001],
        'alpha': [0.5],
        'verbose': [False],
        'nesterov': [False],
        'early_stopping': [False],
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [1],
        'learning_rate_init': [0.0001, 0.0005]
    }
])

"""
print(len(grid))
i = 0
for param1 in grid:
    j = 0
    for param2 in grid:
        if param1 == param2:
            print('%d uguale %d' %(i, j))
            #print(param1)
            #print(param2)
        j += 1
    i += 1
    print('---------------------')"""

file_parameters = open(FILE_PARAMETERS, 'w')
i = 0
for parameters in fixed_grid:
    file_parameters.write('----------------------------------------CONFIGURATION %d----------------------------------------' % i)
    file_parameters.write('\n')
    if i == 0:
        file_parameters.write('default value')
    elif i == 1:
        file_parameters.write('activation_hidden = logistic')
    elif i == 2:
        file_parameters.write('activation_hidden = relu')
    elif i == 3:
        file_parameters.write('hidden_layer_size = [10]')
    elif i == 4:
        file_parameters.write('hidden_layer_size = [20]')
    elif i == 5:
        file_parameters.write('hidden_layer_size = [30, 30]')
    elif i == 6:
        file_parameters.write('hidden_layer_size = [50]')
    elif i == 7:
        file_parameters.write('hidden_layer_size = [100]')
    elif i == 8:
        file_parameters.write('hidden_layer_size = [50, 50]')
    elif i == 9:
        file_parameters.write('hidden_layer_size = [100, 100]')
    elif i == 10:
        file_parameters.write('epochs = 100')
    elif i == 11:
        file_parameters.write('epochs = 1000')
    elif i == 12:
        file_parameters.write('learning_rate = fixed')
    elif i == 13:
        file_parameters.write('lambd = 0')
    elif i == 14:
        file_parameters.write('lambd = 0.001')
    elif i == 15:
        file_parameters.write('lambd = 0.01')
    elif i == 16:
        file_parameters.write('lambd = 0.1')
    elif i == 17:
        file_parameters.write('alpha = 0.5')
    elif i == 18:
        file_parameters.write('alpha = 0.7')
    elif i == 19:
        file_parameters.write('alpha = 0.9')
    elif i == 20:
        file_parameters.write('nesterov = True')
    elif i == 21:
        file_parameters.write('early_stopping = True')
    elif i == 22:
        file_parameters.write('batch_size = 1')
    elif i == 23:
        file_parameters.write('batch_size = 32')
    elif i == 24:
        file_parameters.write('batch_size = 256')
    elif i == 25:
        file_parameters.write('learning_rate_init = 0.1, with batch_size = 1.0')
    elif i == 26:
        file_parameters.write('learning_rate_init = 0.01, with batch_size = 1.0')
    elif i == 27:
        file_parameters.write('learning_rate_init = 0.0001, with batch_size = 1')
    elif i == 28:
        file_parameters.write('learning_rate_init = 0.0005, with batch_size = 1')
    
    file_parameters.write('\n')
    file_parameters.write(str(parameters))
    file_parameters.write('\n \n')
    i += 1
file_parameters.close()

X_train, y_train = read_tr_cup(CUP_TRAIN_PATH)
#X_blind_test = read_ts_cup(CUP_TEST_PATH)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
file_scores = open(FILE_SCORES, 'w')

i = 0
for parameters in fixed_grid:
    print('start configuration %d in [0, 28]' %i)
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
        validation_frequency = parameters['validation_frequency'],
        random_state = parameters['random_state'],
        reinit_weights = parameters['reinit_weights'],
        weights_dist = parameters['weights_dist'],
        metric_decrease_tol = parameters['metric_decrease_tol']
    )
    
    print('net configuration %d in [0, 28] initialized' %i)
    
    # net.fit(X_train, y_train)
    # pred = net.predict(X_test)
    # scores = net.train_scores
    
    scores = k_fold_cross_validation(net, X_train, y_train, 3)
    
    file_scores.write('----------------------------------------CONFIGURATION %d----------------------------------------' % i)
    file_scores.write('\n')
    file_scores.write(str(scores))
    file_scores.write('\n \n')
    
    print('configuration %d in [0, 28] done' %i)
    i += 1
    
file_scores.close()