import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from network import Network
from sklearn.neural_network import MLPRegressor
from utils import mse, mee

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

def compute_y_poly(x, coefs):
    exp = len(coefs) - 1
    y = 0
    for coef in coefs:
        y += coef * np.power(x, exp)
        exp -= 1
    return y

# polynomial coefficients (the size determines the degree of the polynomial)
coefs = [2, 2, -1, 0] # y = 2*x^3 + x^2 - x
#coefs = [0, 1, 0, 0] # y = x^2
# x interval extrems from where to sample points
xinf = -2
xsup = 1
# standard deviation of gaussian noise
sigma = 0.6
# dataset dimension
n_points = 25

# plot true function
x, y = random_poly(coefs, xinf, xsup, size=100)
plt.plot(
    x, 
    y, 
    '-', 
    label="true function")
x, y = random_poly(coefs, xinf, xsup, size=n_points)
# reshape x
x_2dim = x.reshape(x.size, 1)
# sample points from gaussian centered in 0
error = np.random.normal(0, sigma, n_points)
# add noise
y = y + error
"""# split data in train and test
X_train, X_test, y_train, y_test = train_test_split(
    x_2dim, 
    y, 
    test_size=0.33, 
    random_state=42)"""
X_train = x_2dim
y_train = y
X_test = np.linspace(xinf, xsup, 80)
X_test = X_test.reshape(X_test.shape[0], 1)
y_test = np.full_like(X_test, 0)
for i in range(X_test.shape[0]):
    y_test[i][0] = compute_y_poly(X_test[i][0], coefs)

# fit our net and get predictions for test points
net = Network(
    hidden_layer_sizes=[10, 10],
    activation_out='identity',
    classification=False,
    activation_hidden='tanh',
    lambd=0,
    batch_size=1,
    epochs = 3000,
    learning_rate='fixed',
    learning_rate_init=0.0001,
    alpha=0.9,
    early_stopping=False,
    tol=0,
    metric_decrease_tol=0,
    )
net.fit(X_train, y_train.reshape(y_train.shape[0], 1))
y_pred = net.predict(X_test)
y_pred_no_reg = y_pred.reshape(y_pred.shape[0])
no_reg_mse = mse(y_test, y_pred_no_reg)
no_reg_mee = mee(y_test, y_pred_no_reg)

net = Network(
    hidden_layer_sizes=[10, 10],
    activation_out='identity',
    classification=False,
    activation_hidden='tanh',
    lambd=0.001,
    batch_size=1,
    epochs = 3000,
    learning_rate='fixed',
    learning_rate_init=0.0001,
    alpha=0.9,
    early_stopping=False,
    tol=0,
    metric_decrease_tol=0
    )
net.fit(X_train, y_train.reshape(y_train.shape[0], 1))
y_pred = net.predict(X_test)
y_pred_reg = y_pred.reshape(y_pred.shape[0])
reg_mse = mse(y_test, y_pred_reg)
reg_mee = mee(y_test, y_pred_reg)

# fit scikit-learn net and get predictions for test points
"""scikit_net = MLPRegressor(
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
scikit_mee = mee(y_test, scikit_y_pred)"""

# plot train and test points
plt.plot(np.ravel(X_train), y_train, 'o', color='c', label="train")
#plt.plot(np.ravel(X_test), y_test, 'o', color='b', label="test")

# plot predictions
plt.plot(
    np.ravel(X_test), 
    y_pred_no_reg, 
    '-', 
    color='m',
    label="not reg predictions (MSE=%.2f, MEE=%.2f)"%(no_reg_mse, no_reg_mee))
plt.plot(
    np.ravel(X_test), 
    y_pred_reg, 
    '-', 
    color='y',
    label="reg predictions (MSE=%.2f, MEE=%.2f)"%(reg_mse, reg_mee))

plt.legend()
plt.show()