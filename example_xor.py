import numpy as np

from network import Network
from layer import Layer
from utils import tanh, tanh_prime, mse, mse_prime, fixed

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

print(x_train.shape)
net = Network(activation_out='tanh', epochs= 1000, batch_size=1, learning_rate_fun=fixed(0.01))
net.fit(x_train, y_train)

out = net.predict(x_train)
print(x_train)
print(y_train)
print(out)
