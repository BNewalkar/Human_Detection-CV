import numpy as np
import neural_net as model

# function to load saved data
def load_data(path):
	data = np.genfromtxt(path, delimiter=',', dtype=float)
	return data[:,:-1], data[:,-1].astype(int)

train_x, train_y = load_data("train.csv")
test_x, test_y = load_data("test.csv")

# MLP Training
# learning rate
lr = 0.01

# random weight initialization
w1 = np.random.normal(0, .1, size=(train_x.shape[1], 250))
w2 = np.random.normal(0, .1, size=(250,1))
b1 = np.random.normal(0, .1, size=(1,250))
b2 = np.random.normal(0, .1, size=(1,1))

mlp = model.MLP(w1, b1, w2, b2, lr)

# set epoch values
epoch = 400
steps = epoch*train_y.size

# training neural network
mlp.train(train_x, train_y, steps)

# evaluation function to calculate accuracy
def evaluate(solutions, real):
	if(solutions.shape != real.shape):
		raise ValueError("Output is wrong shape.")
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

# predicting on test data
solutions = mlp.predict(test_x)

# printing NN output
print("NN Output:")
print(solutions)

# printing predictions
print("Predictions")
solutions = np.round(solutions)
print(solutions)

# printing evaluation accuracy
print(evaluate(solutions, test_y))
