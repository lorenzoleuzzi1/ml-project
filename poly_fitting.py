import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from network import Network
from sklearn.neural_network import MLPRegressor
from utils import mse, mee

def visualize_weights(weights, bias):
    for w, b in zip(weights, bias): # for now biases are not displayed
        print("------")
        print(w)
        print(b)
        plt.imshow(w, vmax=2, vmin=-1)
        plt.colorbar()
        plt.show()

def random_poly(coefs, xinf, xsup, size):
    # generate size points equally spaced in [xinf, xsup]
    x = np.linspace(xinf, xsup, size)
    # compute ordinates
    exp = len(coefs) - 1
    y = np.zeros(size)
    for coef in coefs:
        y += coef * np.power(x, exp)
        exp -= 1
    return x, y

# polynomial coefficients (the size determines the degree of the polynomial)
coefs = [2, 1, -1, 0] # y = 2*x^3 + x^2 - x
#coefs = [0, 1, 0, 0] # y = x^2
# x interval extrems from where to sample points
xinf = -1
xsup = 1
# standard deviation of gaussian noise
sigma = 0.1
# dataset dimension
n_points = 60

x, y = random_poly(coefs, xinf, xsup, size=n_points)
plt.plot(
    x, 
    y, 
    '-', 
    label="true function")
# reshape x
x_2dim = x.reshape(x.size, 1)
# sample points from gaussian centered in 0
error = np.random.normal(0, sigma, n_points)
# add noise
y = y + error
# split data in train and test
X_train, X_test, y_train, y_test = train_test_split(
    x_2dim, 
    y, 
    test_size=0.33, 
    random_state=42)

# fit our net and get predictions for test points
net = Network(
    hidden_layer_sizes=[3, 3],
    activation_out='identity',
    classification=False,
    activation_hidden='tanh',
    lambd=0.0001,#1,#0.1
    batch_size=1,
    learning_rate='fixed',
    learning_rate_init=0.001,
    alpha=0.9
    )
net.fit(X_train, y_train.reshape(y_train.shape[0], 1))
y_pred = net.predict(X_test)
y_pred = y_pred.reshape(y_pred.shape[0])
our_mse = mse(y_test, y_pred)
our_mee = mee(y_test, y_pred)

weights, bias = net.get_current_weights()
visualize_weights(weights, bias)
net = Network(
    hidden_layer_sizes=[3, 3],
    activation_out='identity',
    classification=False,
    activation_hidden='tanh',
    lambd=1, # high labda, weights should be smaller...
    batch_size=1,
    learning_rate='fixed',
    learning_rate_init=0.001,
    alpha=0.9
    )
net.fit(X_train, y_train.reshape(y_train.shape[0], 1))
weights, bias = net.get_current_weights()
visualize_weights(weights, bias)

# fit scikit-learn net and get predictions for test points
scikit_net = MLPRegressor(
    hidden_layer_sizes=(3, 3),
    activation='tanh', # for hidden layers
    solver='sgd', 
    alpha=1,#0.1,#0.0001, # our lambd
    batch_size=1, 
    learning_rate='constant', 
    learning_rate_init=0.001, 
    max_iter=1000,
    shuffle=True, 
    tol=0.0005,
    momentum=0.9, # our alpha
    nesterovs_momentum=False,
    early_stopping=True
    )
scikit_net.fit(X_train, y_train)
scikit_y_pred = scikit_net.predict(X_test)
scikit_mse = mse(y_test, scikit_y_pred)
scikit_mee = mee(y_test, scikit_y_pred)

# plot train and test points
plt.plot(np.ravel(X_train), y_train, 'o', color='c', label="train")
plt.plot(np.ravel(X_test), y_test, 'o', color='b', label="test")

# plot predictions
plt.plot(
    np.ravel(X_test), 
    y_pred, 
    'o', 
    color='m',
    label="our predictions (MSE=%.2f, MEE=%.2f)"%(our_mse, our_mee))
plt.plot(
    np.ravel(X_test), 
    scikit_y_pred, 
    'o', 
    color='y',
    label="scikit-learn predictions (MSE=%.2f, MEE=%.2f)"%(scikit_mse, scikit_mee))

plt.legend()
plt.show()