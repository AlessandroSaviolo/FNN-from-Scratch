import numpy as np
from utilities import plot_latent_features_2D, plot_latent_features_3D


def relu(x):															# activation function
	return max(0, x)


def relu_derivative(x):													# activation function derivative
	return 1 if x > 0 else 0


def sigmoid(x):															# activation function
	return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):												# activation function derivative
	return sigmoid(x) * (1 - sigmoid(x))


def mean_square_error(predictions, targets):							# loss function
	return np.sum([(predictions[i][0] - targets[i]) ** 2 for i in range(len(predictions))])


def cross_entropy(predictions, targets):								# loss function
	return np.sum([-targets[i] * np.log(predictions[i]) - (1 - targets[i]) * np.log(1 - predictions[i]) for i in range(predictions.shape[0])])


def compute_accuracy(predictions, targets):							# compute accuracy for classification task
	correct = 0
	for i in range(len(targets)):
		if targets[i] == predictions[i]:
			correct += 1
	return correct / float(len(targets)) * 100.0


class Regretter:

	def __init__(self, l0, l1, l2, l3):
		self.weights_0 = np.random.randn(l0, l1)			# weights from input layer to first hidden layer
		self.weights_1 = np.random.randn(l1, l2)			# weights from first hidden layer to second hidden layer
		self.weights_2 = np.random.randn(l2, l3)			# weights from second hidden layer to output layer
		self.relu = np.vectorize(relu)

	def regress(self, X_train, y_train, learning_rate, num_epochs, verbose):

		# forward propagation
		hidden_layer_1 = self.relu(np.dot(X_train, self.weights_0))				# relu as activation function
		hidden_layer_2 = self.relu(np.dot(hidden_layer_1, self.weights_1))			# relu as activation function
		output_layer = np.dot(hidden_layer_2, self.weights_2)

		error_history = [mean_square_error(output_layer, y_train)]

		if verbose:
			print('Initial Error:', error_history[0])

		for e in range(num_epochs):
			print('Training model --> Epoch', e)

			for p in range(output_layer.shape[0]):

				# backward propagation
				output_error = output_layer[p] - y_train[p] 					# error in output
				output_delta = output_error

				hidden_2_delta = np.zeros(len(self.weights_1[0]))  			# how much 2nd hidden layer weights contributed to output error
				for i in range(len(self.weights_1[0])):
					hidden_2_delta[i] = output_delta * self.weights_2[i][0] * relu_derivative(hidden_layer_2[p][i])

				hidden_1_delta = np.zeros(len(self.weights_0[0]))  			# how much 1st hidden layer weights contributed to output error
				for i in range(len(self.weights_0[0])):
					hidden_1_delta[i] = np.sum([self.weights_1[i][k] * hidden_2_delta * relu_derivative(hidden_layer_1[p][i]) for k in range(len(self.weights_1[0]))])

				for i in range(len(self.weights_2)):  							# adjusting (2nd hidden --> output) weights
					self.weights_2[i][0] -= learning_rate * output_delta * hidden_layer_2[p][i]

				for i in range(len(self.weights_1)):  							# adjusting (1st hidden --> 2nd hidden) weights
					for j in range(len(self.weights_1[0])):
						self.weights_1[i][j] -= learning_rate * hidden_2_delta[j] * hidden_layer_1[j][i]

				for i in range(len(self.weights_0)):  							# adjusting (input --> 1st hidden) weights
					for j in range(len(self.weights_0[0])):
						self.weights_0[i][j] -= learning_rate * hidden_1_delta[j] * X_train[j][i]

			# forward propagation: update predictions
			hidden_layer_1 = self.relu(np.dot(X_train, self.weights_0))
			hidden_layer_2 = self.relu(np.dot(hidden_layer_1, self.weights_1))
			output_layer = np.dot(hidden_layer_2, self.weights_2)

			error_history.append(mean_square_error(output_layer, y_train))
			if verbose:
				print('Error after training for', e, 'epochs:', error_history[-1])

		return error_history

	def predict(self, input_matrix):
		hidden_layer_1 = self.relu(np.dot(input_matrix, self.weights_0))			# forward propagation
		hidden_layer_2 = self.relu(np.dot(hidden_layer_1, self.weights_1))
		return np.dot(hidden_layer_2, self.weights_2)[0]


class Classifier:

	def __init__(self, l0, l1, l2):
		self.weights_0 = np.random.randn(l0, l1)						# weights from input layer to hidden layer
		self.weights_1 = np.random.randn(l1, l2)						# weights from hidden layer to output layer
		self.relu = np.vectorize(relu)
		self.sigmoid = np.vectorize(sigmoid)
		self.best_weights_0 = self.weights_0
		self.best_weights_1 = self.weights_1

	def classify(self, X_train, y_train, learning_rate, num_epochs, verbose):

		# forward propagation
		hidden_layer = self.relu(np.dot(X_train, self.weights_0))					# relu as activation function
		output_layer = self.sigmoid(np.dot(hidden_layer, self.weights_1))			# sigmoid as activation function

		error_history = [cross_entropy(output_layer, y_train)]
		acc_history = [compute_accuracy(np.array([0 if p < 0.5 else 1 for p in output_layer]), y_train)]

		if verbose:
			print('Initial Error:', error_history[0])
			print('Training accuracy:', acc_history[0])

		# save best accuracy for saving the best model found during training
		best_accuracy = 0

		for e in range(num_epochs):
			print('Training model --> Epoch', e)

			for p in range(output_layer.shape[0]):

				# backward propagation
				output_error = output_layer[p] - y_train[p] 							# error in output
				output_delta = output_error * sigmoid_derivative(output_layer[p])  	# applying derivative of sigmoid to error

				hidden_delta = np.zeros(len(self.weights_0[0]))  						# how much hidden layer weights contributed to output error
				for i in range(len(self.weights_0[0])):
					hidden_delta[i] = np.sum([self.weights_1[i][k] * output_delta * relu_derivative(hidden_layer[p][i]) for k in range(len(self.weights_1[0]))])

				for i in range(len(self.weights_1)):  									# adjusting (hidden --> output) weights
					self.weights_1[i][0] -= learning_rate * output_delta * hidden_layer[p][i]

				for i in range(len(self.weights_0)):  									# adjusting (input --> hidden) weights
					for j in range(len(self.weights_0[0])):
						self.weights_0[i][j] -= learning_rate * hidden_delta[j] * X_train[j][i]

			# forward propagation: update predictions
			hidden_layer = self.relu(np.dot(X_train, self.weights_0))  				# relu as activation function
			output_layer = self.sigmoid(np.dot(hidden_layer, self.weights_1))  		# sigmoid as activation function

			error_history.append(cross_entropy(output_layer, y_train))
			acc_history.append(compute_accuracy(np.array([0 if p < 0.5 else 1 for p in output_layer]), y_train))

			# save best model
			if acc_history[-1] > best_accuracy:
				best_accuracy = acc_history[-1]
				self.best_weights_0 = self.weights_0
				self.best_weights_1 = self.weights_1

			if verbose:
				print('Error after training for', e, 'epochs:', error_history[-1])
				print('Training accuracy after training for', e, 'epochs:', acc_history[-1])

				# plot the distribution of latent features at different training stage
				if e % 30 == 0:
					print('Plotting latent features at epoch', e)
					if hidden_layer.shape[1] == 2:
						plot_latent_features_2D(hidden_layer, [0 if p < 0.5 else 1 for p in output_layer], e)
					elif hidden_layer.shape[1] == 3:
						plot_latent_features_3D(hidden_layer, [0 if p < 0.5 else 1 for p in output_layer], e)

		return error_history, acc_history

	def predict(self, input_matrix):
		hidden_layer = self.relu(np.dot(input_matrix, self.best_weights_0))			# forward propagation																		# type == 0 -> classification
		return self.sigmoid(np.dot(hidden_layer, self.best_weights_1))[0]
