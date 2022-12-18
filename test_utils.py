from utils import *
import numpy as np

# No asserts, need to call the functions and to check what they print

size = 2
input = np.random.rand(1, size)-0.5
print("IN:")
print(input)
true = np.random.rand(1, size)-0.5
print("TRUE:")
print(true)

def test_relu():
    f_net = relu(input)
    print("RELU(INPUT):")
    print(f_net)
    f_prime = relu_prime(input)
    print("RELU_PRIME(INPUT):")
    print(f_prime)

def test_leaky_relu():
    f_net = leaky_relu(input)
    print("LEAKY_RELU(INPUT):")
    print(f_net)
    f_prime = leaky_relu_prime(input)
    print("LEAKY_RELU_PRIME(INPUT):")
    print(f_prime)

def test_logistic():
    f_net = logistic(input)
    print("LOGISTIC(INPUT):")
    print(f_net)
    f_prime = logistic_prime(input)
    print("LOGISTIC_PRIME(INPUT):")
    print(f_prime)

def test_softplus():
    f_net = softplus(input)
    print("SOFTPLUS(INPUT):")
    print(f_net)
    f_prime = softplus_prime(input)
    print("SOFTPLUS_PRIME(INPUT):")
    print(f_prime)

def test_softmax():
    f_net = softmax(input)
    print("SOFTMAX(INPUT):")
    print(f_net)

def test_losses():
    pred = np.array([[1, 1], [3, 1]]) # mse flatten array (1 + 4 + 1)/4 == (1/2 + 5/2)/2 = 3/2 = 1.5
    true = np.array([[2, 1], [5, 2]]) # mee = (1+sqrt(5))/2
    #pred = np.array([[3,1]])
    #true = np.array([[5,2]])
    print(pred.shape)
    print("--MEE--")
    print(mee(true, pred))
    print("--MSE--")
    print(mse(true, pred))
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(y_pred=pred, y_true=true))

test_losses()