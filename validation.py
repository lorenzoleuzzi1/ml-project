import numpy as np
from utils import fold_plot, mean_and_std
from sklearn.model_selection import StratifiedKFold
from network import Network
from utils import write_json, read_json
from multiprocessing import Process

JSON_PATH = 'monks_cv_results.json'

def cross_validation(network, X_train, y_train, k):
    if k <= 1:
        print('Number of folds k must be more than 1')
        raise ValueError("k must be more than 1")

    skf = StratifiedKFold(n_splits=k, shuffle=True) 
    network.verbose = False
    # init error and score vectors    
    tr_error_fold = []
    es_val_error_fold = []
    tr_score_fold = []
    es_val_score_fold = []
    val_score_fold = []
    i = 1
    for train_index, validation_index in skf.split(X_train, y_train):
        print("{} fold".format(i))
       
        #-----stratified K-fold split-----
        X_train_fold, X_val_fold = X_train[train_index], X_train[validation_index] 
        y_train_fold, y_val_fold = y_train[train_index], y_train[validation_index] 
        

        # --------------fold train--------------
        tr_error, val_error, tr_score, val_score = network.fit(X_train_fold, y_train_fold)
        
        #difference between max epochs and epochs actually done (because early stopping)
        epoches_difference = network.epochs - len(tr_error) 
        if epoches_difference > 0:
            tr_error.extend(np.zeros(epoches_difference))
            val_error.extend(np.zeros(epoches_difference))
            tr_score.extend(np.zeros(epoches_difference))
            val_score.extend(np.zeros(epoches_difference))    

        tr_error_fold.append(tr_error)        
        es_val_error_fold.append(val_error)
        tr_score_fold.append(tr_score)      
        es_val_score_fold.append(val_score) 

        # --------------fold validation--------------
        pred = network.predict(X_val_fold)
        score = network.evaluate(Y_true=y_val_fold, Y_pred=pred)
        val_score_fold.append(score) 
        print("{} fold VL score = {}".format(i, score))    
        i+=1
    
    
    # --------------results--------------
    
    last_tr_error = []
    last_es_val_error = []
    last_tr_score = []
    last_es_val_score = []
    last_val_score = []
    
    # create an array of the last results for each K-fold
    for tr_error, val_error, tr_score, val_score, test_score \
        in zip(tr_error_fold, es_val_error_fold, tr_score_fold, es_val_score_fold, val_score_fold):       
        last_tr_error.append(np.trim_zeros(tr_error, 'b')[-1])
        last_es_val_error.append(np.trim_zeros(val_error, 'b')[-1])
        last_tr_score.append(np.trim_zeros(tr_score, 'b')[-1])
        last_es_val_score.append(np.trim_zeros(val_score, 'b')[-1])
        last_val_score.append(test_score)

    # mean and std dev of the last results (of each K-fold) 
    mean_tr_error, std_tr_error = mean_and_std(last_tr_error) 
    mean_es_val_error, std_es_val_error = mean_and_std(last_es_val_error)
    mean_tr_score, std_tr_score = mean_and_std(last_tr_score)
    mean_es_val_score, std_es_val_score = mean_and_std(last_es_val_score)
    mean_val_score, std_val_score = mean_and_std(last_val_score)

    # # average of the learning curve  TODO: rename
    # avg_tr_error, dev_tr_error = mean_std_dev(np.array(tr_error_fold))
    # avg_es_val_error, dev_val_error = mean_std_dev(np.array(es_val_error_fold))
    # avg_tr_score, dev_tr_score = mean_std_dev(np.array(tr_score_fold))
    # avg_es_val_score, dev_val_score = mean_std_dev(np.array(es_val_score_fold))


    # # plot results
    # fold_plot("error", tr_error_fold, es_val_error_fold, avg_tr_error, avg_es_val_error) 
    # fold_plot("score", tr_score_fold, es_val_score_fold, avg_tr_score, avg_es_val_score) 
    
    #-----print results-----
    print("-----CV-----")
    print("TR error - mean: {} std: {}  \nES VL error - mean {} std: {}"
        .format(mean_tr_error, std_tr_error, mean_es_val_error, std_es_val_error))
    print("TR score - mean: {} std: {}  \nES VL score - mean {} std: {}"
        .format(mean_tr_score, std_tr_score, mean_es_val_score, std_es_val_score))
    print("VL score - mean: {} std: {}".format(mean_val_score, std_val_score))

    results = {
        'tr_loss' : mean_tr_error,
        'tr_loss_std' : std_tr_error,
        'val_score' : mean_val_score,
        'val_score_dev' : std_val_score
    }
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
        nested_results = cross_validation(network, X_train, y_train, k)
        print("------")
        print(config)
        print(f"outer score: {nested_results.get('val_score')} +/- {nested_results.get('val_score_dev')}")
        #TODO: save results into json?
        i += 1

def grid_search_cv(grid, X_train, y_train, k):
    print(f"starting grid search - exploring {len(grid)} configs")
    i = 1
    for config in grid:
        print(f"{i}/{len(grid)}")
        network = config_to_network(config)
        cv_results = cross_validation(network, X_train, y_train, k)
        cv_results.update({'config' : config})
        write_json(cv_results, JSON_PATH)
        i += 1


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
