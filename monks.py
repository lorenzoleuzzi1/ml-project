import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from validation import nested_cross_validation, cross_validation
from network import Network
from utils import error_plot, accuracy_plot
from sklearn.neural_network import MLPClassifier
#from cross_validation import cross_validation
import matplotlib.pyplot as plt


MONKS1_TRAIN_PATH = './datasets/monks-1.train'
MONKS1_TEST_PATH = './datasets/monks-1.test'
MONKS2_TRAIN_PATH = './datasets/monks-2.train'
MONKS2_TEST_PATH = './datasets/monks-2.test'
MONKS3_TRAIN_PATH = './datasets/monks-3.train'
MONKS3_TEST_PATH = './datasets/monks-3.test'

TRAIN_PATH = MONKS1_TRAIN_PATH
TEST_PATH = MONKS1_TEST_PATH

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

grid = ParameterGrid(
    {   
        #---fixed TODO: ricerca ad occhio per migliori
        'activation_out': ['tanh'],
        'classification' : [True],
        'activation_hidden': ['tanh'],
        'hidden_layer_sizes': [[3]],
        'loss': ['mse'],
        'evaluation_metric' : ['accuracy'], 
        'epochs': [200],
        'learning_rate_init': [0.002], 
        'tau' : [200],
        'lambd' : [0.0001],
        'alpha' : [0.9],
        'nesterov' : [True],
        'early_stopping' : [True],
        'stopping_patience' : [20],
        'validation_size' : [0.1],
        'tol' : [0.0005], 
        'validation_frequency' : [4],
        #---to tune
        'learning_rate': ['fixed', 'linear_decay'],
        'batch_size': [1, 32],
    }
)

X_train, y_train = read_monks(TRAIN_PATH)
X_test, y_test = read_monks(TEST_PATH)

net = Network(
    hidden_layer_sizes=[3],
    activation_out='softmax',
    classification=True,
    activation_hidden='tanh',
    epochs = 200, 
    batch_size = 1.0,
    lambd=0,
    #lambd = 0.0001,
    learning_rate = "fixed",
    learning_rate_init=0.001,
    #nesterov=True, 
    early_stopping=False,
    evaluation_metric='accuracy',
    verbose=True,
    loss='logloss',
    validation_frequency=1,
    validation_size=0.1,
    tol=1e-4,
    random_state=0)

net.fit(X_train, y_train) # no early stopping
#tr_loss, val_loss, tr_score, val_score = net.fit(X_train, y_train) # early stopping
pred = net.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=pred))
print(net.get_current_weights())
plt.plot(net.train_losses, label="training loss", color="blue")
#plt.plot(tr_score, label="training score", color="green")
#plt.plot(val_loss, label="validation loss", color="red")
#plt.plot(val_score, label="validation score", color="black")
plt.legend(loc="upper right")
plt.title("OUR")
plt.show()

scikit_net = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='tanh', # for hidden layers
    solver='sgd', 
    alpha=0,
    #alpha=0.0001, # our lambd
    batch_size=32, 
    learning_rate='constant', 
    learning_rate_init=0.002, 
    max_iter=200,
    shuffle=True, 
    tol=0.0005,
    momentum=0.9, # our alpha
    nesterovs_momentum=False,
    validation_fraction=0.2,
    early_stopping=False,
    random_state = 0,
    )
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
scikit_net.fit(X_train, y_train)
plt.plot(scikit_net.loss_curve_, label="training loss", color="blue")
plt.title("SCIKIT LEARN")
plt.show()
pred = scikit_net.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=pred))