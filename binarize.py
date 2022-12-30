import numpy as np
from sklearn import preprocessing

lb_zero = preprocessing.LabelBinarizer(pos_label=1, neg_label=0)
lb_minus1 = preprocessing.LabelBinarizer(pos_label=1, neg_label=-1)

"""lb_zero.fit([1, 2])
print(lb_zero.transform([1, 2]))
print("------")
lb_zero.fit([1, 2, 3])
print(lb_zero.transform([1, 2]))
print(lb_zero.inverse_transform(np.array([[0,0,1], [0,1,0]])))
print("------")
lb_minus1.fit([1, 2, 3])
print(lb_minus1.transform([1, 2]))
print(lb_minus1.inverse_transform(np.array([[0,0,1], [0,1,0]])))
print("------")
lb_zero.fit(["no", "sì"])
print(lb_zero.transform(["no", "sì"]))"""


Y = [np.array([0.6, 0.4, 0.3]), np.array([0.1, 0.4, 0.3]), np.array([0.4, 0.4, 0.3])]
B = np.max(Y, axis=1)
Y_new = []
for i in range(len(Y)):
    Y_new.append(np.where(Y[i] < B[i], 0.0, 1.0))
    j = 0
    s = int(sum(Y_new[i]) - 1)
    if s > 0:
        for j in range(s):
            if Y_new[i][j] == 1.0:
                Y_new[i][j] = 0.0
                j += 1
        
print(Y_new)