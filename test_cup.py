from cup_parsing import load_dev_set_cup
from network import Network

X_dev, y_dev = load_dev_set_cup()
net = Network(activation_out='identity', classification=False, random_state=0)
net.fit(X_dev, y_dev)

# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# [1, 2, 3, 4] PRIMO FOLD
# [5, 6, 7, 8] SECONDO FOLD
# [9, 10, 11, 12] TERZO FOLD

# MODELLO1
# allenato su fold 1,2 testato su fold 3
	# shuffle di [1,2,3,4,5,6,7,8] => prima epoca [2,1,4,3,8,7,5,6]
# allenato su fold 1,3 testato su fold 2
	# shuffle di [1,2,3,4,9,10,11,12] => prima epoca [2,1,4,3,12,11, 9, 10]
# allenato su fold 2,3 testato su fold 1

# MODELLO2
# allenato su fold 1,2 testato su fold 3
	# shuffle di [1,2,3,4,5,6,7,8] => prima epoca [2,1,4,3,8,7,5,6]
# allenato su fold 1,3 testato su fold 2
# allenato su fold 2,3 testato su fold 1
