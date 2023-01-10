import numpy as np
from utils import fold_plot, mean_and_std
from sklearn.model_selection import StratifiedKFold, KFold
from network import Network
from utils import write_json, read_json
from scipy.stats import rankdata
from multiprocessing import Process
import pandas as pd

JSON_PATH = 'monks_cv_results.json'

def k_fold_cross_validation(network, X_train, y_train, k):
    if k <= 1:
        print('Number of folds k must be more than 1')
        raise ValueError("k must be more than 1")

    if network.classification:
        kf = StratifiedKFold(n_splits=k, shuffle=True) 
    else:
        kf = KFold(n_splits=k, shuffle=False) # NOTE: to compare different models on the same splits
    
    folds_metrics = []
    i = 1
    for train_index, validation_index in kf.split(X_train, y_train): # TODO: farlo con numpy? (possiamoe eliminare stratified)
        metrics = []
        print("{} fold".format(i))
       
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
        score = network.score(X_test=X_val_fold, Y_test=y_val_fold)
        best_epoch = network.best_epoch
        metrics.append(score)
        metrics.append(best_epoch)
        print("{} fold VL score = {}".format(i, score))    

        folds_metrics.append(metrics)
        i+=1
     
    # --------------results--------------
    # fold metrics contains for every fold in this order:
    #   - train losses and scores
    #   - if early stopping validation loss and score 
    #   - test score
    #   - best epoch 
    best_metrics = np.zeros(shape = (len(folds_metrics[0]), k))
    # retrive the best (at the end of epoch) value for every fold
    for i, fold in enumerate(folds_metrics):  
        best_epoch = fold[-1]  
        for j, values in enumerate(fold):
            if isinstance(values, list):
                # best_metrics[j][i] = np.trim_zeros(values, 'b')[-1] # a list of values, take the last one
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
        'tr_loss_mean' : means[0],
        'tr_loss_dev' : stds[0],
        'tr_score_mean': means[1],
        'tr_score_dev': stds[1],
        'val_score_mean' : means[-2],
        'val_score_dev' : stds[-2]
    }
    for i in range(k):
        results['split%d_tr_loss'%i] = best_metrics[0][i]
        results['split%d_tr_score'%i] = best_metrics[1][i]
        results['split%d_val_score'%i] = best_metrics[-2][i]
        results['split%d_best_epoch'%i] = best_metrics[-1][i]

    print("---K-fold results---")
    for k, v in zip(results.keys(), results.values()):
        print(f"{k} : {v}")

    return results

def nested_cross_validation(grid, X_train, y_train, k):
    
    # inner kfold
    # per multiprocessing dividere la griglia in n parti ogni parte va ad un Process
    # Process(target=grid_search_cv, args=(grid, X_train, y_train, k))
    # n_processi = 3
    # splitted_grid = np.array_split(grid, n_processi)
    # for partial_grid in splitted_grid:
    #     Process(target=grid_search_cv, args=(grid, X_train, y_train, k)).start()
    
    grid_search_cv(grid, X_train, y_train, k)
    
   
    data = read_json(JSON_PATH)
    print(f"starting outer cv - exploring {len(grid)} configs")
    i = 1
   
    for config in data:
        #outerfold
        print(f"{i}/{len(grid)}")
        network = config_to_network(config.get("config"))
        nested_results = k_fold_cross_validation(network, X_train, y_train, k)
        print("------")
        print(config)
        print(f"outer score: {nested_results.get('val_score')} +/- {nested_results.get('val_score_dev')}")
        #TODO: save results into json?
        i += 1

def grid_search_cv(grid, X, y, k, results_path): # TODO: clean the following code (assuming gs will be executed on a single machine)
    metric = None
    for param in grid:
        if metric==None:
            metric = param['evaluation_metric']
        elif param['evaluation_metric'] != metric:
            raise ValueError("Evaluation metric must be the same for each configuration.")
    print(f"starting grid search - exploring {len(grid)} configs")
    df_scores = pd.DataFrame(columns=[])
    for i, config in enumerate(grid):
        print(f"{i}/{len(grid)}")
        network = Network(**config)
        cv_results = k_fold_cross_validation(network, X, y, k)
        cv_results['params'] = config
        df_scores = pd.concat([df_scores, pd.DataFrame([cv_results])], ignore_index=True)
    # if config['evaluation_metric'] == 'accuracy':
    #     df_scores['val_score_mean_rank'] = rankdata(1-df_scores['val_score_mean'], method='dense')
    #     df_scores['tr_score_mean_rank'] = rankdata(1-df_scores['tr_score_mean'], method='dense')
    # else:
    #     df_scores['val_score_mean_rank'] = rankdata(df_scores['val_score_mean'], method='dense')
    #     df_scores['tr_score_mean_rank'] = rankdata(df_scores['tr_score_mean'], method='dense')
    
    # df_scores['tr_loss_mean_rank'] = rankdata(df_scores['tr_loss_mean'], method='dense')
    # columns_order = [
    #     'val_score_mean',
    #     'val_score_mean_rank',
    #     'tr_loss_mean',
    #     'tr_loss_mean_rank',
    #     'tr_score_mean',
    #     'tr_score_mean_rank'
    # ]
    # for i in range(k):
    #     columns_order.append('split%d_val_score'%i)
    #     columns_order.append('split%d_tr_loss'%i)
    #     columns_order.append('split%d_tr_score'%i)
    #     columns_order.append('split%d_best_epoch'%i)
    # columns_order.append('params')
    # df_scores = df_scores[columns_order]
    # df_scores.sort_values(by=['val_score_mean_rank'])
    df_scores.to_csv(results_path)

def mean_std_dev(data_fold):
    """return average and std deviation for each epoch"""
    k = data_fold.shape[0]
    epochs = data_fold.shape[1]
    mean = [] # init
    std_dev = []
    
    for j in range(epochs):
        mu = 0
        N = 0
        for i in range(k):
            mu += data_fold[i][j]
            if data_fold[i][j] != 0: N +=1
        if N == 0: break
        else:
            mu = mu / N
            mean.append(mu)
            sigma = 0
            for i in range(k):
                sigma += np.power((data_fold[i][j] - mu), 2)
            sigma = np.power((sigma / N), 0.5)
            std_dev.append(sigma)
    
    return mean, std_dev

def config_to_network(config: dict):
    activation_out = config.get("activation_out")
    classification = config.get("classification")
    activation_hidden = config.get("activation_hidden")
    hidden_layer_sizes = config.get("hidden_layer_sizes")
    loss = config.get("loss")
    evaluation_metric = config.get("evaluation_metric")
    epochs = config.get("epochs")
    learning_rate = config.get("learning_rate")
    learning_rate_init = config.get("learning_rate_init")
    tau = config.get("tau")
    lambd = config.get("lambd")
    alpha = config.get("alpha")
    early_stopping = config.get("early_stopping")
    stopping_patience = config.get("stopping_patience")
    validation_size = config.get("validation_size")
    nesterov = config.get("nesterov")
    validation_frequency = config.get("validation_frequency")
    tol = config.get("tol")
    batch_size = config.get("batch_size")
    
    network = Network(
            activation_out=activation_out, classification=classification, activation_hidden=activation_hidden,
            hidden_layer_sizes=hidden_layer_sizes, epochs=epochs, learning_rate=learning_rate, evaluation_metric=evaluation_metric,
            learning_rate_init=learning_rate_init, tau=tau, batch_size=batch_size, lambd=lambd,alpha=alpha, tol=tol, loss=loss,
            nesterov=nesterov, early_stopping=early_stopping, stopping_patience=stopping_patience, validation_size=validation_size,
            validation_frequency=validation_frequency
            )
   
    return network
