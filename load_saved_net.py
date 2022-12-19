import pickle

PATH = 'net.pkl'

# net must have been previously saved wuth net.save(PATH)

file = open(PATH, 'rb')
net = pickle.load(file)
# now net.predict(...) is available

# Print weights for test purposes
for l in net.layers:
	print(l.weights)