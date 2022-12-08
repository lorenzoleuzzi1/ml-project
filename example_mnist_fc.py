import numpy as np

from network import Network
from layer import Layer
from utils import tanh, tanh_prime, mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

import matplotlib.pyplot as plt

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
net = Network(mse, mse_prime)
net.add(Layer(28*28, 100, tanh, tanh_prime))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
#net.add(ActivationLayer(tanh, tanh_prime))
net.add(Layer(100, 50, tanh, tanh_prime))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
#net.add(ActivationLayer(tanh, tanh_prime))
net.add(Layer(50, 10, tanh, tanh_prime))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
#net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
#net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=30, learning_rate=0.1, batch_size = 32)

samples = 10
for test, true in zip(x_test[:samples], y_test[:samples]):
    image = np.reshape(test, (28, 28))
    plt.imshow(image, cmap='binary')
    plt.show()
    pred = net.predict(test) 
    idx = np.argmax(pred)
    idx_true = np.argmax(true)
    print('pred: %s, prob: %.2f, true: %d' % (idx, np.max(pred), idx_true))

ratio = sum([np.argmax(y) == np.argmax(net.predict(x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print("ratio: {}".format(ratio))
