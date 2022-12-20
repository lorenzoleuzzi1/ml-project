import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from network import Network
from utils import error_plot, accuracy_plot, flatten_pred

# TODO: rappresentare graficamente: accuracy_fold + dev std di accuracy e error per ogni epoca (o forse no?)

def _cross_validation(X_train, y_train, X_test, y_test, k, epochs):
    if k <= 1:
        print('Number of folds k must be more than 1')
        exit()

    X_train, y_train = shuffle(X_train, y_train) # random reorganize the order of the data
    
    # TODO: divisione in fold con StratifiedKFold (posso usarlo?)
    
    number_used_data = len(X_train) - (len(X_train) % k) # numbers of data and target used in cross validation
    data = []
    targets = []
    for i in range(number_used_data):
        data.append(X_train[i])
        targets.append(y_train[i])
        
    # convert to numpy object        
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
    test_accuracy_fold = []

    # --------------cross validation--------------
    for fold in range(k):
        # create validation set and training set
        print("FOLD {}".format(k))
        tr_data, tr_targets, val_data, val_targets = create_sets(data_folds, target_folds, fold)

        net = Network(activation_out='tanh', epochs=epochs, batch_size=8, learning_rate = "linear_decay", learning_rate_init=0.05, nesterov=True)
        
        # --------------train--------------
        # return error and accuracy values for each epoch 
        tr_error, val_error, tr_accuracy, val_accuracy = net.fit(tr_data, tr_targets)        
        
        # reshape
        for _ in range(epochs - len(tr_error)):
            tr_error.append(0)
            val_error.append(0)
            tr_accuracy.append(0)
        for _ in range(epochs - len(val_accuracy)):
            val_accuracy.append(0)   
        # convert to numpy object
        tr_error = np.array(tr_error)
        val_error = np.array(val_error)
        tr_accuracy = np.array(tr_accuracy)
        val_accuracy = np.array(val_accuracy)   

        # update errors and accuracy: 
        tr_error_fold.append(tr_error)        
        val_error_fold.append(val_error)
        tr_accuracy_fold.append(tr_accuracy)      
        val_accuracy_fold.append(val_accuracy) 
          
        # --------------test--------------
        pred = net.predict(val_data)
        flattened_pred = flatten_pred(pred)
        accuracy = accuracy_score(y_true=val_targets, y_pred=flattened_pred)
        test_accuracy_fold.append(accuracy) # update accuracy
    #---------------------------------------------
        
    # convert to numpy object
    tr_error_fold = np.array(tr_error_fold, dtype=object)
    val_error_fold = np.array(val_error_fold, dtype=object)
    tr_accuracy_fold = np.array(tr_accuracy_fold, dtype=object)
    val_accuracy_fold = np.array(val_accuracy_fold, dtype=object)
    test_accuracy_fold = np.array(test_accuracy_fold, dtype=object)
    
    # --------------results--------------
    # i.e. average and std deviation of error and accuracy for each epoch
    avg_tr_error, dev_tr_error = mean_std_dev(tr_error_fold)
    avg_val_error, dev_val_error = mean_std_dev(val_error_fold)
    avg_tr_accuracy, dev_tr_accuracy = mean_std_dev(tr_accuracy_fold)
    avg_val_accuracy, dev_val_accuracy = mean_std_dev(val_accuracy_fold)
    
    # plot results
    error_plot(avg_tr_error, avg_val_error)
    accuracy_plot(avg_tr_accuracy, avg_val_accuracy)
    
    # predict and accuracy
    pred_on_test_data = net.predict(X_test)
    
    for p, y in zip(pred_on_test_data, y_test):
        print("pred: {} expected: {}".format(p[0],y))

    flattened_pred_on_test_data = flatten_pred(pred_on_test_data)
    print("accuracy: {}%".format(accuracy_score(y_true=y_test, y_pred=flattened_pred_on_test_data) * 100))

def create_sets(data_folds, target_folds, val_idx):
    """create k set of folds"""
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
            tr_data = np.concatenate((tr_data, fold)) # concatenate matrices
        idx += 1
    tr_data = np.array(tr_data, dtype=np.float32)

    idx = start
    for fold in target_folds[start:]:
        if idx != val_idx:
            tr_targets = np.concatenate((tr_targets, fold))
        idx += 1
    tr_targets = np.array(tr_targets, dtype=np.float32)

    return tr_data, tr_targets, val_data, val_targets

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