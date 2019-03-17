import numpy as np

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.fc1 = FCLayer(w1, b1, lr)
		self.rel = ReLU()
		self.fc2 = FCLayer(w2, b2, lr)
		self.sig = Sigmoid()

	# function to calculate mean squared error
	def MSE(self, prediction, target):
		return (0.5*(target-prediction)**2).sum()

	# function to calculate the error
	def MSEGrad(self, prediction, target):
		return -(target - prediction)

	# training neural network
	def train(self, X, y, steps):
		stop = False
		prev_loss = 0.0
		s = 0

		# training will end when either epoch iterations are completed or when error is very low
		while stop != True and s != steps:
			i = s % y.size
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.fc1.forward(xi)
			pred = self.rel.forward(pred)
			pred = self.fc2.forward(pred)
			pred = self.sig.forward(pred)
			loss = self.MSE(pred, yi)

			if round(abs(loss - prev_loss), 6) == 0.0:
				print("Epochs:", s/y.size+1)
				stop = True
				break

			prev_loss = loss
			grad = self.MSEGrad(pred, yi)
			grad = self.sig.backward(grad)
			grad = self.fc2.backward(grad)
			grad = self.rel.backward(grad)
			grad = self.fc1.backward(grad)
			s += 1

	# prediction using trained NN
	def predict(self, X):
		pred = self.fc1.forward(X)
		pred = self.rel.forward(pred)
		pred = self.fc2.forward(pred)
		pred = self.sig.forward(pred)
		# pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

 	# forward pass
	def forward(self, input):
		self.input = input
		h = np.dot(input, self.w) + self.b
		return h

	# backward pass
	def backward(self, gradients):
		input = self.input
		x_ = np.dot(gradients, self.w.T)
		self.w = self.w - np.dot(input.T, gradients)*self.lr
		self.b = self.b - gradients*self.lr
		return x_

class Sigmoid:

	def __init__(self):
		None

	def sigmoid_func(self, a):
		return 1/(1+np.exp(-a))

	#forward pass
	def forward(self, input):
		self.input = input
		sig_val = self.sigmoid_func(input)
		return sig_val

	# backward pass
	def backward(self, gradients):
		input = self.input
		sig_val_back = gradients*(1 - self.sigmoid_func(input))*self.sigmoid_func(input)
		return sig_val_back

class ReLU:
	def __init__(self):
		None

	# forward pass
	def forward(self, input):
		self.input = input
		input[input<0] = 0
		return input

	# backward pass
	def backward(self, gradients):
		input = self.input
		input[input < 0] = 0
		input[input > 0] = 1
		input *= gradients
		return input
