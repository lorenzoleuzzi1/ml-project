import numpy as np
from network import Network
from utils import linear_decay

def cross_validation(data, targets, X_test, k, epochs):
    data = np.array(data, dtype=int)
    targets = np.array(targets, dtype=int)
    
    # split dataset into k fold:   
    data_folds = np.array(np.array_split(data, k), dtype=object) # list of folds containing the data
    target_folds = np.array(np.array_split(targets, k), dtype=object) # list of folds containing the targets
    
    # init error and accuracy vector:
    tr_error = np.ones(epochs)  # sum of error values for each epoch
    val_error = np.ones(epochs)
    
    tr_error_fold = []
    val_error_fold = []

    # cross validation
    for i in range(k):
        # create validation set and training set
        tr_data, tr_targets, val_data, val_targets = create_sets(data_folds, target_folds, i)

        net = Network(activation_out='tanh', epochs=300, batch_size=32, learning_rate_fun=linear_decay(200, 0.1))
        
        # train
        [tr_error, val_error] = net.fit(tr_data, tr_targets, val_data, val_targets)  # error and metric values for each epoch
        # test
        pred = net.predict(val_data)

        # update errors:
        tr_error += tr_error    
        val_error += val_error
     
        tr_error_fold.append(tr_error)        
        val_error_fold.append(val_error)    
        
    # average of errors and metrics ( k == number of folds )
    tr_error /= k
    val_error /= k
    print("training error: {0}, val error:{1}".format(tr_error, val_error))
    
    # results i.e. average and std deviation of error
    avg_tr_error = np.mean(tr_error_fold)  # average
    dev_tr_error = np.std(tr_error_fold)  # standard deviation
    avg_val_error = np.mean(val_error_fold)
    dev_val_error = np.std(val_error_fold)
    
    pred = net.predict(X_test)

    return tr_error, val_error, tr_error_fold, val_error_fold, pred


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
