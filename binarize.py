import numpy as np
from sklearn import preprocessing

lb_zero = preprocessing.LabelBinarizer(pos_label=1, neg_label=0)
lb_minus1 = preprocessing.LabelBinarizer(pos_label=1, neg_label=-1)

lb_zero.fit([1, 2])
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
print(lb_zero.transform(["no", "sì"]))