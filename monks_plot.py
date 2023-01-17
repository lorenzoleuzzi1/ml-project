import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from validation import k_fold_cross_validation, grid_search_cv
from network import Network
from utils import error_plot, accuracy_plot
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

MONKS1_TRAIN_PATH = './datasets/monks-1.train'
MONKS1_TEST_PATH = './datasets/monks-1.test'
MONKS2_TRAIN_PATH = './datasets/monks-2.train'
MONKS2_TEST_PATH = './datasets/monks-2.test'
MONKS3_TRAIN_PATH = './datasets/monks-3.train'
MONKS3_TEST_PATH = './datasets/monks-3.test'

def read_monks(path, one_hot_encoding=True, target_rescaling=True):
    data = pd.read_csv(path, sep=" ", header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy() # int 64
    if one_hot_encoding:
        data = OneHotEncoder().fit_transform(data).toarray() # float 64
    if target_rescaling:
        targets[targets == 0] = -1 # int 64
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

def plot_monks_curves(net, data_set_name):
    print("MSE train %f" % net.train_losses[net.best_epoch])
    print("MSE test %f" % net.val_losses[net.best_epoch])
    print("ACCURACY train %f" % net.train_scores[net.best_epoch])
    print("ACCURACY test %f" % net.val_scores[net.best_epoch])

    plt.figure()
    plt.plot(net.train_losses, label="Training", color="blue")
    plt.plot(net.val_losses, 'r--', label='Test')
    #plt.vlines(x=net.best_epoch, ymin=0, ymax=net.val_losses[net.best_epoch], color='black', linestyle='dashed', linewidth=0.8)
    #plt.plot(net.best_epoch, 0.015, 'k*', label='Backtracked epoch')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    #plt.ylim(-0.05, 1)
    plt.savefig("./monks_curves/%s_loss_curves.pdf" %data_set_name, bbox_inches="tight")

    plt.figure()
    plt.plot(net.train_scores, label="Training", color="blue")
    plt.plot(net.val_scores, 'r--', label="Test")
    #plt.vlines(x=net.best_epoch, ymin=0, ymax=net.val_scores[net.best_epoch], color='black', linestyle='dashed', linewidth=0.8)
    #plt.plot(net.best_epoch, 0.015, 'k*', label='Backtracked epoch')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.savefig("./monks_curves/%s_accuracy_curves.pdf" %data_set_name, bbox_inches="tight")

# MONKS1
X_train1, y_train1 = read_monks(MONKS1_TRAIN_PATH)
X_test1, y_test1 = read_monks(MONKS1_TEST_PATH)

net1 = Network(
    hidden_layer_sizes=[3],
    activation_out='tanh',
    classification=True,
    activation_hidden='logistic',
    epochs = 400,
    lambd=0,
    learning_rate = "fixed",
    batch_size=8,
    learning_rate_init=0.05,
    alpha=0.9,
    nesterov=False,
    early_stopping=False,
    evaluation_metric='accuracy',
    verbose=False,
    loss='mse',
    tol=0.00001,
    metric_decrease_tol=0.000001,
    random_state=None,
    stopping_patience=30,
    reinit_weights=True,
    weights_dist='uniform',
    weights_bound=0.7
    )

# MONKS2
X_train2, y_train2 = read_monks(MONKS2_TRAIN_PATH)
X_test2, y_test2 = read_monks(MONKS2_TEST_PATH)

net2 = Network(
    hidden_layer_sizes=[4],
    activation_out='logistic',
    classification=True,
    activation_hidden='logistic',
    epochs = 100,
    lambd=0,
    learning_rate = "fixed",
    batch_size=1,
    learning_rate_init=0.05,
    alpha=0.9,
    nesterov=False,
    early_stopping=False,
    evaluation_metric='accuracy',
    verbose=False,
    loss='mse',
    tol=0.00001,
    metric_decrease_tol=0.000001,
    random_state=None,
    stopping_patience=30,
    reinit_weights=True,
    weights_dist='uniform',
    weights_bound=0.7
    )

# MONKS3
net3 = Network( #11,tanh,logistic,32,[4],0,0.005,0.7,uniform
    hidden_layer_sizes=[4],
    activation_out='logistic',
    classification=True,
    activation_hidden='tanh',
    epochs = 200,
    lambd=0,
    learning_rate = "fixed",
    batch_size=32,
    learning_rate_init=0.05,
    alpha=0.9,
    nesterov=False,
    early_stopping=False,
    evaluation_metric='accuracy',
    verbose=False,
    loss='mse',
    tol=0.00001,
    metric_decrease_tol=0.000001,
    random_state=None,
    stopping_patience=30,
    reinit_weights=True,
    weights_dist='uniform',
    weights_bound=0.7 
    )

net3_reg = Network(
    hidden_layer_sizes=[4],
    activation_out='logistic',
    classification=True,
    activation_hidden='tanh',
    epochs = 200,
    lambd=0.01,
    learning_rate = "fixed",
    batch_size=32,
    learning_rate_init=0.05,
    alpha=0.9,
    nesterov=False,
    early_stopping=False,
    evaluation_metric='accuracy',
    verbose=False,
    loss='mse',
    tol=0.00001,
    metric_decrease_tol=0.000001,
    random_state=None,
    stopping_patience=30,
    reinit_weights=True,
    weights_dist='uniform',
    weights_bound=0.7 
    )

X_train3, y_train3 = read_monks(MONKS3_TRAIN_PATH)
X_test3, y_test3 = read_monks(MONKS3_TEST_PATH)

n_trials = 5

for i in range(n_trials):
    net1.fit(X_train1, y_train1, X_test1, y_test1)
    plot_monks_curves(net1, 'monks1_%d' %i)
    
    print('accuracy_test = %f' % net1.score(X_test1, y_test1, 'accuracy'))
    print('accuracy_train = %f' %net1.train_scores[net1.best_epoch])
    print('mse_train = %f' %net1.train_losses[net1.best_epoch])
    print('mse_test = %f' %net1.val_losses[net1.best_epoch])
    print('accuracy_test_internal = %f' %net1.val_scores[net1.best_epoch])
    
for i in range(n_trials):
    net2.fit(X_train2, y_train2, X_test2, y_test2)
    plot_monks_curves(net2, 'monks2_%d' %i)
    
    print('accuracy_test = %f' % net2.score(X_test2, y_test2, 'accuracy'))
    print('accuracy_train = %f' %net2.train_scores[net2.best_epoch])
    print('mse_train = %f' %net2.train_losses[net2.best_epoch])
    print('mse_test = %f' %net2.val_losses[net2.best_epoch])
    print('accuracy_test_internal = %f' %net2.val_scores[net2.best_epoch])
    
for i in range(n_trials):
    net3.fit(X_train3, y_train3, X_test3, y_test3)
    plot_monks_curves(net3, 'monks3_%d' %i)
    
    print('accuracy_test = %f' % net3.score(X_test3, y_test3, 'accuracy'))
    print('accuracy_train = %f' %net3.train_scores[net3.best_epoch])
    print('mse_train = %f' %net3.train_losses[net3.best_epoch])
    print('mse_test = %f' %net3.val_losses[net3.best_epoch])
    print('accuracy_test_internal = %f' %net3.val_scores[net3.best_epoch])
    
for i in range(n_trials):
    net3_reg.fit(X_train3, y_train3, X_test3, y_test3)
    plot_monks_curves(net3_reg, 'monks3_reg_%d' %i)
    
    print('accuracy_test = %f' % net3_reg.score(X_test3, y_test3, 'accuracy'))
    print('accuracy_train = %f' %net3_reg.train_scores[net3_reg.best_epoch])
    print('mse_train = %f' %net3_reg.train_losses[net3_reg.best_epoch])
    print('mse_test = %f' %net3_reg.val_losses[net3_reg.best_epoch])
    print('accuracy_test_internal = %f' %net3_reg.val_scores[net3_reg.best_epoch])
