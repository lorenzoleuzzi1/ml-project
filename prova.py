from utils import mse, mse_prime, logloss, logloss_prime
import numpy as np
#import pytorch.torch.nn as nn

"""
A = [[1,2,3], [1,2,3], [1,2,3]]
B = [[1,2,1], [1,1,3], [1,4,3]]

print(mse(A,B))
print(mse_prime(A,B))
print(np.mean(A))
"""

a = [1, 0, 0]
b = [1, 0, 0]
a = np.array(a)
b = np.array(b)
print(logloss(a, b))
print(logloss_prime(a, b))

"""loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()"""