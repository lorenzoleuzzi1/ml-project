import numpy as np
from sklearn.metrics import accuracy_score
from network import Network
from utils import fold_plot, flatten_pred, mean_and_std
from sklearn.model_selection import StratifiedKFold

def cross_validation(X_train, y_train, X_test, y_test, k, epochs):
    # if k <= 1:
    #     print('Number of folds k must be more than 1')
    #     exit()
    
    skf = StratifiedKFold(n_splits=k, shuffle=True) 

    # init error and score vectors    
    tr_error_fold = []
    val_error_fold = []
    tr_score_fold = []
    val_score_fold = []
    test_score_fold = []

    for train_index, validation_index in skf.split(X_train, y_train):
        #-----stratified K-fold split-----
        print("FOLD")
        #split training set for validation
        X_train_fold, X_val_fold = X_train[train_index], X_train[validation_index] 
        y_train_fold, y_val_fold = y_train[train_index], y_train[validation_index] 

        # --------------train--------------
        net = Network(activation_out='tanh')
        tr_error, val_error, tr_score, val_score = net.fit(X_train_fold, y_train_fold) 
        
        #fill with 0s
        for _ in range(epochs - len(tr_error)):
            tr_error.append(0)
            val_error.append(0)
            tr_score.append(0)
            val_score.append(0)    

        tr_error_fold.append(tr_error)        
        val_error_fold.append(val_error)
        tr_score_fold.append(tr_score)      
        val_score_fold.append(val_score) 

        # --------------outer validation--------------
        pred = net.predict(X_val_fold)
        flattened_pred = flatten_pred(pred)
        score = accuracy_score(y_true=y_val_fold, y_pred=flattened_pred)
        test_score_fold.append(score) 
    
    
    # --------------results--------------
    
    last_tr_error = []
    last_val_error = []
    last_tr_score = []
    last_val_score = []
    last_outer_val_score = []
    #create an array of the last results for each K-fold
    for tr_error, val_error, tr_score, val_score, test_score \
        in zip(tr_error_fold, val_error_fold, tr_score_fold, val_score_fold, test_score_fold):       
        last_tr_error.append(np.trim_zeros(tr_error, 'b')[-1])
        last_val_error.append(np.trim_zeros(val_error, 'b')[-1])
        last_tr_score.append(np.trim_zeros(tr_score, 'b')[-1])
        last_val_score.append(np.trim_zeros(val_score, 'b')[-1])
        last_outer_val_score.append(test_score)

    #mean and std dev of the last results (of each K-fold) TODO: better way without appending and trimming zeros
    mean_tr_error, std_tr_error = mean_and_std(last_tr_error) 
    mean_val_error, std_val_error = mean_and_std(last_val_error)
    mean_tr_score, std_tr_score = mean_and_std(last_tr_score)
    mean_val_score, std_val_score = mean_and_std(last_val_score)
    mean_outer_val_score, std_outer_val_score = mean_and_std(last_outer_val_score)

    #average of the learning curve  TODO: rename
    avg_tr_error, dev_tr_error = mean_std_dev(np.array(tr_error_fold))
    avg_val_error, dev_val_error = mean_std_dev(np.array(val_error_fold))
    avg_tr_score, dev_tr_score = mean_std_dev(np.array(tr_score_fold))
    avg_val_score, dev_val_score = mean_std_dev(np.array(val_score_fold))


    # plot results
    fold_plot("error", tr_error_fold, val_error_fold, avg_tr_error, avg_val_error) 
    fold_plot("score", tr_score_fold, val_score_fold, avg_tr_score, avg_val_score) 
    
    #-----test-----
    # predict and score
    pred_on_test_data = net.predict(X_test)
    flattened_pred_on_test_data = flatten_pred(pred_on_test_data)
    
    # for p, y in zip(pred_on_test_data, y_test):
    #     print("pred: {} expected: {}".format(p[0],y))

    
    #-----print results-----
    print("TR error - mean: {} std: {}  \nVL error - mean {} std: {}".format(mean_tr_error, std_tr_error, mean_val_error, std_val_error))
    print("TR score - mean: {} std: {}  \nVL score - mean {} std: {}".format(mean_tr_score, std_tr_score, mean_val_score, std_val_score))
    print("outer VL score - mean: {} std: {}".format(mean_outer_val_score, std_outer_val_score))
    print("TEST score: {}%".format(accuracy_score(y_true=y_test, y_pred=flattened_pred_on_test_data) * 100)) #TODO: non per forza accuracy

    #TODO: return results?

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