from utils import *
import numpy as np

# No asserts, need to call the functions and to check what they print

size = 1
input = np.random.rand(1, size)-0.5
print("IN:")
print(input)
true = np.random.rand(1, size)-0.5
print("TRUE:")
print(true)

def test_tanh():
    f_net = tanh(input)
    print("TANH(INPUT):")
    print(f_net)
    f_prime = tanh_prime(input)
    print("TANH_PRIME(INPUT):")
    print(f_prime)

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
    input = np.array([1, 2])
    f_net = softmax(input)
    print("SOFTMAX(INPUT):")
    print(f_net)
    f_prime = softmax_prime(input)
    print("SOFTMAX_PRIME(INPUT):")
    print(f_prime)

def test_losses():
    pred = np.array([[2,2]])
    true = np.array([3,2])
    print("--MEE prime--")
    print(mee_prime(true, pred)) # (y_pred - y_true) / f
    print("--MSE prime--")
    print(mse_prime(true, pred)) # 2 * (y_pred - y_true) / y_true.size
    print("--MSE --")
    print(mse(true, pred))
    print("--MEE--")
    print(mee(true, pred))
    #from sklearn.metrics import mean_squared_error
    #print(mean_squared_error(y_pred=pred, y_true=true))

test_losses()
# target    prob
# 1         1
# 1         0.7
# 0         0.7
# 1         1
# - ((0)+(1*log(0.7))+(1*log(0.3))+(0))/4
# -(log(0.7)+log(0.3))/4 = 
#print(log_loss(y_true=np.array([[1,0], [1,0], [1,0], [1,0]]), y_prob=np.array([[1,0],[0.7, 0.3],[0.3,0.7],[1,0]])))
#print(-(np.log(0.7)+np.log(0.3))/4)