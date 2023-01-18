import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, KFold
from network import Network
from utils import mse, mee

# TODO: evaluation_metric come lista? se passi mee, mse li calcola entrambi sui validation folds
# TODO: aggiungere parametro shuffle (teniamo falso nella grid search! vogliamo confrontare modelli diversi su stessi dati)
def k_fold_cross_validation(network, X_train, y_train, k, evaluation_metric):
    if k <= 1:
        print('Number of folds k must be more than 1')
        raise ValueError("k must be more than 1")

    if network.classification:
        kf = StratifiedKFold(n_splits=k, shuffle=True) # TODO: shuffle
    else:
        kf = KFold(n_splits=k, shuffle=False) # TODO: shuffle
    
    folds_metrics = []
    for train_index, validation_index in kf.split(X_train, y_train):
        metrics = []
       
        #-----stratified K-fold split-----
        X_train_fold, X_val_fold = X_train[train_index], X_train[validation_index] 
        y_train_fold, y_val_fold = y_train[train_index], y_train[validation_index] 
        
        # --------------fold train--------------
        network.fit(X_train_fold, y_train_fold)
        metrics.append(network.train_losses)
        metrics.append(network.train_scores)
        
        if network.early_stopping == True:
            metrics.append(network.val_losses)
            metrics.append(network.val_scores)
        
        # --------------fold validation--------------
        Y_pred = network._predict_outputs(X=X_val_fold)
        net_score = mse(y_true=y_val_fold, y_pred=Y_pred)
        val_score = mee(y_true=y_val_fold, y_pred=Y_pred)
        # TODO: modificare score in modo che gli passo lista

        best_epoch = network.best_epoch
        metrics.append(net_score)
        metrics.append(val_score)
        metrics.append(best_epoch)

        folds_metrics.append(metrics)
     
    # --------------results--------------
    # fold metrics contains for every fold in this order:
    #   - train losses and scores
    #   - if early stopping validation loss and score 
    #   - test score network.evaluation_metric
    #   - test score evaluation_metric (della chiamata a questo metodo)
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
    #   - train losses (pos [0])
    #   - train scores (pos [1])
    #   - if early stopping internal validation losses
    #   - if ealry stopping internal validation scores
    #   - validation score [-2]
    #   - epoch [-1]
    means = []
    stds = []

    for best_metric in best_metrics:
        means.append(np.mean(best_metric))
        stds.append(np.std(best_metric))

    results = {
        'tr_mse_mean' : means[0],
        'tr_mse_dev' : stds[0],
        'tr_%s_mean'%network.evaluation_metric : means[1], # TODO: medie degli score della lista passata a questa funzione
        'tr_%s_dev'%network.evaluation_metric : stds[1], 
        'val_%s_mean'%evaluation_metric : means[-3],
        'val_%s_dev'%evaluation_metric : stds[-3],
        'val_%s_mean'%network.evaluation_metric : means[-2],
        'val_%s_dev'%network.evaluation_metric : stds[-2],
    }
     
    #   - train losses and scores
    #   - if early stopping validation loss and score 
    #   - test score network.evaluation_metric
    #   - test score evaluation_metric (della chiamata a questo metodo)
    #   - best epoch 
    for i in range(k):
        results['split%d_tr_mse'%i] = best_metrics[0][i]
        results['split%d_tr_%s'%(i,network.evaluation_metric)] = best_metrics[1][i]
        results['split%d_val_%s'%(i, evaluation_metric)] = best_metrics[-3][i]
        results['split%d_val_%s'%(i, network.evaluation_metric)] = best_metrics[-2][i]  # TODO: qui salva tutti gli score della lista passata a questa funzione
        results['split%d_best_epoch'%i] = best_metrics[-1][i]

    return results

def grid_search_cv(grid, X, y, k, results_path, evaluation_metric):
    metric = None
    for param in grid: # TODO: necessario?
        if metric==None:
            metric = param['evaluation_metric'] # TODO: servono altri check? dovremo metterli ovunque...
        elif param['evaluation_metric'] != metric:
            raise ValueError("Evaluation metric must be the same for each configuration.")
    
    print(f"starting grid search - exploring {len(grid)} configs")
    df_scores = pd.DataFrame(columns=[])
    for i, config in enumerate(grid):
        print(f"{i+1}/{len(grid)}")
        network = Network(**config)
        cv_results = k_fold_cross_validation(network, X, y, k, evaluation_metric)
        cv_results['params'] = json.dumps(config)
        df_scores = pd.concat([df_scores, pd.DataFrame([cv_results])], ignore_index=True)
    
    df_scores.to_csv(results_path)

def read_grid_search_results(path):
    df = pd.read_csv(path, sep=",")
    for i in range(len(df['params'])):
        params_as_json_string = df['params'][i]
        params_as_dictionary = json.loads(params_as_json_string)
        df.at[i,'params'] = params_as_dictionary
    return df
