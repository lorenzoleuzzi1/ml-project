import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, KFold
from neural_network import NeuralNetwork

def k_fold_cross_validation(network, X_train, y_train, k, shuffle=True):
    """
    Perform k-fold cross validation on a neural network.
    The dataset is divided into k folds, k-1 of which are used for training and the remaining one for testing.
    This process is repeated k times, with each fold being used as the test set once.

    Parameters:
        - network (NeuralNetwork): an instance of the NeuralNetwork class that will be trained and evaluated.
        
        - X_train (np.array): the training data.
        
        - y_train (np.array): the target data for the training set.
        
        - k (int): the number of folds to use for cross validation.
        
        - shuffle (bool): whether to shuffle the data before splitting it into folds. Default is True.

    Returns:
        - results (dict): a dict of the result obtained in the k-fold cross validation process.
    """
    if k <= 1:
        print('Number of folds k must be more than 1')
        raise ValueError("k must be more than 1")

    if network.classification:
        kf = StratifiedKFold(n_splits=k, shuffle=shuffle)
    else:
        kf = KFold(n_splits=k, shuffle=shuffle) 
    
    folds_metrics = []
    y_preds = []
    y_trues = []

    # loop through folds
    for train_index, validation_index in kf.split(X_train, y_train):
        metrics = []
       
        # stratified K-fold split
        X_train_fold, X_val_fold = X_train[train_index], X_train[validation_index] 
        y_train_fold, y_val_fold = y_train[train_index], y_train[validation_index] 

        # fold training
        network.fit(X_train_fold, y_train_fold)
        metrics.append(network.train_losses)
        metrics.append(network.train_scores)
        
        if network.early_stopping == True:
            metrics.append(network.val_losses)
            metrics.append(network.val_scores)
        
        # fold validation
        metric_values = network.score(X_val_fold, y_val_fold, [network.loss, network.evaluation_metric])
        y_pred = network.preds

        y_preds.append(y_pred)
        y_trues.append(y_val_fold)

        best_epoch = network.best_epoch
        metrics.append(metric_values[network.loss])
        metrics.append(metric_values[network.evaluation_metric])
        metrics.append(best_epoch)

        folds_metrics.append(metrics)
     
    # --------------results processing--------------
    # fold metrics contains for every fold in this order:
    #   - training losses and scores
    #   - if early stopping internal validation loss and score 
    #   - test score for network.loss
    #   - test score for network.evaluation_metric
    #   - best epoch 
    best_metrics = np.zeros(shape = (len(folds_metrics[0]), k))
    # retrive the best (at the end of epoch) value for every fold
    for i, fold in enumerate(folds_metrics):
        best_epoch = fold[-1]  
        for j, values in enumerate(fold):
            if isinstance(values, list):
                best_metrics[j][i] = values[best_epoch]
            else:           
                best_metrics[j][i] = values #single value

    # means and stds contains in this order mean and std of the following metrics over the fold:
    #   - training losses in [0]
    #   - training scores in [1]
    #   - if early stopping internal validation losses
    #   - if ealry stopping internal validation scores
    #   - test score for network.loss in [-3]
    #   - test score for network.evaluation_metric in [-2]
    #   - epoch in [-1]
    means = []
    stds = []

    for best_metric in best_metrics:
        means.append(np.mean(best_metric))
        stds.append(np.std(best_metric))

    # mean results
    results = {
        'tr_%s_mean'%network.loss : means[0],
        'tr_%s_dev'%network.loss : stds[0],
        'tr_%s_mean'%network.evaluation_metric : means[1],
        'tr_%s_dev'%network.evaluation_metric : stds[1],
        'val_%s_mean'%network.loss : means[-3],
        'val_%s_dev'%network.loss : stds[-3],
        'val_%s_mean'%network.evaluation_metric : means[-2],
        'val_%s_dev'%network.evaluation_metric : stds[-2],
    }
     
    # results for each fold
    for i in range(k):
        results['split%d_tr_%s'%(i, network.loss)] = best_metrics[0][i]
        results['split%d_tr_%s'%(i, network.evaluation_metric)] = best_metrics[1][i]
        results['split%d_val_%s'%(i, network.loss)] = best_metrics[-3][i]
        results['split%d_val_%s'%(i, network.evaluation_metric)] = best_metrics[-2][i]
        results['split%d_best_epoch'%i] = best_metrics[-1][i]
    
    results['y_preds'] = y_preds
    results['y_trues'] = y_trues
    # -----------------------------------
    
    return results

def grid_search_cv(grid, X, y, k, results_path):
    """
    Perform a grid search cross-validation on a neural network.
    The grid search explores a specified set of hyperparameters by training and evaluating the 
    network for each combination of parameters.
    
    Parameters:
        - grid (dict): a dictionary containing the hyperparameters to explore and the possible values for each one.
        
        - X (np.array): the data to train and evaluate the network on.
        
        - y (np.array): the target data for the input set.
        
        - k (int): the number of folds to use for cross validation.
        
        - results_path (str): the path to save the results of the grid search.
    """
    metric = grid[0]['evaluation_metric']
    loss = grid[0]['loss']
    for param in grid:
        if param['evaluation_metric'] != metric:
            raise ValueError("Evaluation metric must be the same for each configuration.")
        if param['loss'] != loss:
            raise ValueError("Loss must be the same for each configuration.")
    
    print(f"Starting grid search - exploring {len(grid)} configs")
    
    df_scores = pd.DataFrame(columns=[])

    # loop through each configuration in the grid search grid
    for i, config in enumerate(grid):
        print(f"{i+1}/{len(grid)}")
        # create a NeuralNetwork object with the configuration
        network = NeuralNetwork(**config)
        
        # perform kfold cross validation
        cv_results = k_fold_cross_validation(network, X, y, k, shuffle=False)
        
        cv_results.pop('y_preds')
        cv_results.pop('y_trues')
        cv_results['params'] = json.dumps(config)
        df_scores = pd.concat([df_scores, pd.DataFrame([cv_results])], ignore_index=True)
    
    print("Grid search finished.")
    # save results
    df_scores.to_csv(results_path)
