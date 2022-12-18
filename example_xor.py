import numpy as np

from network import Network

# training data
x_train = np.array([[0,0], [0,1], [1,0], [1,1]]) # TODO: validation set vuoto!
y_train = np.array([0, 1, 1, 0])

print(x_train.shape)
net = Network(activation_out='tanh', epochs=100, batch_size=1)
net.fit(x_train, y_train)

out = net.predict(x_train)
print(out)
