import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from network import Network

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

# sample points from gaussian centered in 0
error = np.random.normal(0, sigma, n_points)
# add noise
y = y + error

# fit net and get predictions for training points
net = Network(activation_out='identity') # TODO: try different settings
x_2dim = x.reshape(x.size, 1)
X_train, X_test, y_train, y_test = train_test_split(x_2dim, y, test_size=0.33, random_state=42)
net.fit(X_train, y_train, X_test, y_test)
y_pred = net.predict(X_test)

# plot train and test points
plt.plot(np.ravel(X_train), y_train, 'o', label="train")
plt.plot(np.ravel(X_test), y_test, 'o', color='b', label="test")

# plot predictions
plt.plot(np.ravel(X_test), y_pred, 'o', label="predictions")
plt.legend()
plt.show()