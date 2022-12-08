import numpy as np

import monks
import plot
from network import Network
from layer import Layer
from utils import tanh, tanh_prime, mse, mse_prime


def main_cv(dataset, k, epochs, learning_rate, batch_size):
    """[tr_error, tr_metric, val_error, val_metric, tr_error_fold, tr_metric_fold, val_error_fold,
        val_metric_fold] = cross_validation(dataset, k, epochs, learning_rate, batch_size)"""
    [tr_error, val_error, tr_error_fold, val_error_fold] = cross_validation(dataset, k, epochs, learning_rate, batch_size)

    # average of errors and metrics ( k == number of folds )
    tr_error /= k
    #tr_metric /= k
    val_error /= k
    #val_metric /= k

    # results i.e. average and std deviation of error and metric
    # training
    avg_tr_error = np.mean(tr_error_fold)  # average
    dev_tr_error = np.std(tr_error_fold)  # standard deviation
    #avg_tr_metric = np.mean(tr_metric_fold)
    #dev_tr_metric = np.std(tr_metric_fold)

    # validation
    avg_val_error = np.mean(val_error_fold)
    dev_val_error = np.std(val_error_fold)
    #avg_val_metric = np.mean(val_metric_fold)
    #dev_val_metric = np.std(val_metric_fold)

    plot.plot_cv()


def cross_validation(dataset, k, epochs, learning_rate, batch_size):

    if dataset in ('monks-1.train', 'monks-2.train', 'monks-3.train'):
        data, targets = monks.read_monk(path=dataset)
    else:
        raise ValueError("Dataset not found")

    # split dataset into k fold
    # list of folds containing the data
    data_folds = np.array(np.array_split(data, k), dtype=object)
    # list of folds containing the targets
    target_folds = np.array(np.array_split(targets, k), dtype=object)

    # cross validation
    for i in range(k):
        # create validation set and training set
        tr_data, tr_targets, val_data, val_targets = create_sets(
            data_folds, target_folds, i)

        # network
        net = Network()
        net.add(Layer(2, 3, tanh, tanh_prime))
        net.add(Layer(3, 1, tanh, tanh_prime))

        # train
        net.use(mse, mse_prime)
        # TODO: qua non ritorna niente -> ma ho bisogno output per calcolare
        # [tr_error_value, tr_metric_value, val_error_value, val_metric_value]
        [tr_error_value, val_error_value] = net.fit(tr_data, tr_targets, epochs=100,
            learning_rate=0.1, batch_size=2)  # error and metric values for each epoch

        # test
        out = net.predict(val_data)

        # update errors end metrics
        # sum of error values for each epoch
        tr_error = np.ones(epochs)  # init
        tr_error += tr_error_value
        #tr_metric = np.ones(epochs)
        #tr_metric += tr_metric_value
        val_error = np.ones(epochs)
        val_error += val_error_value
        #val_metric = np.ones(epochs)
        #val_metric += val_metric_value

        # list of error in fold the fold -> to plot
        tr_error_fold = []  # init
        tr_error_fold.append(tr_error_value) # TODO: tr_error_value[-1] forse?
        #tr_metric_fold = []
        #tr_metric_fold.append(tr_metric_value) # performance of the model on the training set
        val_error_fold = []
        val_error_fold.append(val_error_value)
        #val_metric_fold = []
        #val_metric_fold.append(val_metric_value)

    return tr_error, val_error, tr_error_fold, val_error_fold
    # tr_error, tr_metric, val_error, val_metric, tr_error_fold, tr_metric_fold, val_error_fold, val_metric_fold


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
            tr_data = np.concatenate(tr_data, fold)
        idx += 1
    tr_data = np.array(tr_data, dtype=np.float32)

    idx = start
    for fold in target_folds[start:]:
        if idx != val_idx:
            tr_targets = np.concatenate(tr_targets, fold)
        idx += 1
    tr_targets = np.array(tr_targets, dtype=np.float32)

    return tr_data, tr_targets, val_data, val_targets
