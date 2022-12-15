from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
from network import Network

def get_param_grid(n_samples):
    # Temporary set as follows for test purposes
    grid = ParameterGrid([
        {
            'activation_out': ['tanh', 'logistic'],
            'activation_hidden': ['tanh', 'logistic'],
            'hidden_layer_sizes': [[10], [3, 3]],
            'loss': ['mse', 'rmse'],
            'epochs': [100, 200, 500, 1000],
            'learning_rate': ['fixed'],
            'learning_rate_init': [0.001, 0.05, 0.01, 0.1, 0.5],
            'batch_size': [1, n_samples/4, n_samples/2, n_samples], # TODO: if real?
            'lambd': [0.0001, 0.001, 0.01, 0.1],
            'alpha': [0.5, 0.7, 0.9]
        },
        {
            'activation_out': ['tanh', 'logistic'],
            'activation_hidden': ['tanh', 'logistic'],
            'hidden_layer_sizes': [[10], [3, 3]],
            'loss': ['mse', 'rmse'],
            'epochs': [100,200, 500, 1000],
            'learning_rate': ['linear_decay'],
            'learning_rate_init': [0.001, 0.05, 0.01, 0.1, 0.5],
            'tau': [100], # depends on epochs (must be less)
            'batch_size': [1, n_samples/4, n_samples/2, n_samples], # TODO: if real?
            'lambd': [0.0001, 0.001, 0.01, 0.1],
            'alpha': [0.5, 0.7, 0.9],
        }
    ])
    return grid

N_SPLITS = 5
N_TRIALS = 5 

X, y = load_breast_cancer(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.33)
cv = KFold(n_splits=N_SPLITS, shuffle=True)

# TODO: EARLY STOPPING?
# TODO: how to evaluate variance?

# GRID SEARCH
grid = get_param_grid(len(X_dev) / N_SPLITS)
# param_scores = []
# best_score = min value
# best_params = {} 
for params in grid:
    # CROSS VALIDATION
    # cv_scores = []
    for split, (train_index , test_index) in enumerate(cv.split(X_dev, y_dev)):
        X_train, X_val = X_dev[train_index, :], X_dev[test_index, :]
        y_train, y_val = y_dev[train_index] , y_dev[test_index]
        # trials_scores = []
        for i in range(N_TRIALS):
            net = Network(**params)
            net.fit(X_train, y_train, X_val, y_val)
            y_pred = net.predict(X_val)
            # trials_scores.append(metric(y_pred, y_val))
        # mean_score_trials = mean(trials_scores)
        # cv_scores.append(mean_score_trials)
    # mean_score_cv = mean(cv_scores)
    # param_scores.append(mean_score_cv)
    # if mean_score_cv is better than best_score:
    #   best_score = mean_score_cv
    #   best_params = params

###### RETRAIN WITH RANDOM INIT WEIGHTS ###### (could be risky, if unliky we could fall in local minimum)
# net = Network(**best_params)
# net.fit(X_dev, y_dev)
# y_pred = net.predict(X_test)
# risk = metric(y_pred, y_test)

# OR

###### RETRAIN ENSEMBLE ######
# with init weights which gave the best score in each split?
# with k nets using random init weights and building ensemble?