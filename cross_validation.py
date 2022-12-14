import numpy as np
from network import Network
from utils import linear_decay

# TODO: gestire accuracy_score + usare maschere e/o pulire codice + MOMENTUM(?)

def cross_validation(data, targets, X_test, k, epochs):
    data = np.array(data, dtype=np.int64) 
    targets = np.array(targets, dtype=np.int64)
    
    # split dataset into k fold:   
    data_folds = np.array(np.array_split(data, k), dtype=object) # list of folds containing the data (numpy tensor)
    target_folds = np.array(np.array_split(targets, k), dtype=object) # list of folds containing the targets (numpy matrix)
    
    # init error and accuracy vector:    
    tr_error_fold = []
    val_error_fold = []
    tr_accuracy_fold = []
    val_accuracy_fold = []
    pred = []

    # cross validation
    for i in range(k):
        # create validation set and training set
        tr_data, tr_targets, val_data, val_targets = create_sets(data_folds, target_folds, i)

        net = Network(activation_out='tanh', epochs=300, batch_size=32, learning_rate_fun=linear_decay(200, 0.1))
        
        # train
        tr_error, val_error, tr_accuracy, val_accuracy = net.fit(tr_data, tr_targets, val_data, val_targets)  # error and accuracy values for each epoch        
        # reshape
        for i in range(epochs - len(tr_error)):
            # TODO: se non funzionano maschere per ufunc numpy, togliere e mettere sistemare mean_std
            tr_error.append(0)
            val_error.append(0)
            tr_accuracy.append(0)
            val_accuracy.append(0)   
        # convert to numpy object
        tr_error = np.array(tr_error)
        val_error = np.array(val_error)
        tr_accuracy = np.array(tr_accuracy)
        val_accuracy = np.array(val_accuracy)   
          
        # test
        pred.append(net.predict(val_data)) # TODO: usarlo!!!!!

        # update errors and accuracy: 
        tr_error_fold.append(tr_error)        
        val_error_fold.append(val_error)
        tr_accuracy_fold.append(tr_accuracy)      
        val_accuracy_fold.append(val_accuracy) 
        # TODO: gestire accuracy_score -> tr_accuracy.append(accuracy_score(y_true=y_train, y_pred=predict_tr)) 

    # convert to numpy object
    tr_error_fold = np.array(tr_error_fold, dtype=object)
    val_error_fold = np.array(val_error_fold, dtype=object)
    tr_accuracy_fold = np.array(tr_accuracy_fold, dtype=object)
    val_accuracy_fold = np.array(val_accuracy_fold, dtype=object)
    
    # results i.e. average and std deviation of error and accuracy for each epoch
    #TODO: vedere se si riescono ad usare funzioni np inizializzando maschere con np.ufunc  
    """#where_bool = nnz(tr_error_fold)
    where_bool = np.ufunc.reduce(tr_error_fold, axis=0, dtype=None, out=None, keepdims=False, where=True)
    avg_tr_error = np.mean(tr_error_fold, axis=0, where=where_bool)  # average  
    dev_tr_error = np.std(tr_error_fold, axis=0, where=where_bool)  # standard deviation
    avg_val_error = np.mean(val_error_fold, axis=0)
    dev_val_error = np.std(val_error_fold)    
    avg_tr_accuracy = np.mean(tr_accuracy_fold, axis=0)
    dev_tr_accuracy = np.std(tr_accuracy_fold)
    avg_val_accuracy = np.mean(val_accuracy_fold, axis=0)
    dev_val_accuracy = np.std(val_accuracy_fold)"""
    avg_tr_error, dev_tr_error = mean_std(tr_error_fold)
    avg_val_error, dev_val_error = mean_std(val_error_fold)
    avg_tr_accuracy, dev_tr_accuracy = mean_std(tr_accuracy_fold)
    avg_val_accuracy, dev_val_accuracy = mean_std(val_accuracy_fold)
    
    pred_on_test_data = net.predict(X_test) #???????

    return avg_tr_error, avg_val_error, avg_tr_accuracy, avg_val_accuracy, pred_on_test_data

def create_sets(data_folds, target_folds, val_idx):

    # validation fold
    val_data = data_folds[val_idx]
    val_data = np.array(val_data, dtype=np.float32)

    val_targets = target_folds[val_idx]
    val_targets = np.array(val_targets, dtype=np.float32)

    # training fold
    if val_idx != 0:
        tr_data = data_folds[0]
        tr_targets = target_folds[0]
        start = 1
        idx = 1
    else:
        tr_data = data_folds[1]
        tr_targets = target_folds[1]
        start = 0
        idx = 0

    for fold in data_folds[start:]:
        if idx != val_idx:
            tr_data = np.concatenate((tr_data, fold)) #concatenate matrices
        idx += 1
    tr_data = np.array(tr_data, dtype=np.float32)

    idx = start
    for fold in target_folds[start:]:
        if idx != val_idx:
            tr_targets = np.concatenate((tr_targets, fold))
        idx += 1
    tr_targets = np.array(tr_targets, dtype=np.float32)

    return tr_data, tr_targets, val_data, val_targets

def nnz(fold):
    # TODO: se inutile togliere!!!
    # returns an array of booleans of the same size as the input
    # true where the element are nnz
    
    fold_bool = np.ones(fold.shape, dtype=np.bool8)
    i = 0
    
    for arr in fold:
        j = 0
        for elem in arr:
            if elem == 0:
                fold_bool[i][j] = 0
            j += 1
        i += 1
    
    return fold_bool

def mean_std(data_fold):
    # return average and std deviation for each epoch
    k = data_fold.shape[0]
    epochs = data_fold.shape[1]
    mean = [] # init
    std = []
    
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
            std.append(sigma)
    
    return mean, std