import numpy as np
from sklearn.model_selection import train_test_split
from utilities import load_dataset, plot_learning_curve, plot_latent_features_2D, plot_latent_features_3D


def preprocessing():
	df = load_dataset('ionosphere_csv.csv')						# load data
	X = (df.drop(['class'], 1))
	X = (X - X.min()) / (X.max() - X.min())  					# standardize data
	X = X.replace(np.NaN, 0)
	y = df['class'].transform(lambda x: 1 if x is 'g' else 0)			# one hot encode target
	return X, y


def compute_accuracy(predictions, targets):
	correct = 0
	for i in range(len(targets)):
		if targets[i] == predictions[i]:
			correct += 1
	return correct / float(len(targets)) * 100.0


def relu(x):																	# activation function
	return max(0, x)


def relu_derivative(x):									# activation function derivative
	return 1 if x > 0 else 0


def sigmoid(x):																	# activation function
	return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):								# activation function derivative
	return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy(predictions, targets):						# loss function
	return np.sum([-targets[i] * np.log(predictions[i]) - (1 - targets[i]) * np.log(1 - predictions[i]) for i in range(predictions.shape[0])])


class Classifier:

	def __init__(self, l0, l1, l2, l3, l4):
		self.weights_0 = np.random.randn(l0, l1)				# weights from input layer to first hidden layer
		self.weights_1 = np.random.randn(l1, l2)				# weights from first hidden layer to second hidden layer
		self.weights_2 = np.random.randn(l2, l3)  				# weights from second hidden layer to third hidden layer
		self.weights_3 = np.random.randn(l3, l4)				# weights from third hidden layer to output layer
		self.relu = np.vectorize(relu)
		self.sigmoid = np.vectorize(sigmoid)
		self.best_weights_0 = self.weights_0
		self.best_weights_1 = self.weights_1
		self.best_weights_2 = self.weights_2
		self.best_weights_3 = self.weights_3

	def classify(self, X_train, y_train, learning_rate, num_epochs, verbose):

		# forward propagation
		hidden_layer_1 = self.relu(np.dot(X_train, self.weights_0))			# relu as activation function
		hidden_layer_2 = self.relu(np.dot(hidden_layer_1, self.weights_1))		# relu as activation function
		hidden_layer_3 = self.relu(np.dot(hidden_layer_2, self.weights_2))  		# relu as activation function
		output_layer = self.sigmoid(np.dot(hidden_layer_3, self.weights_3))		# sigmoid as activation function

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
				output_error = output_layer[p] - y_train[p] 				# error in output
				output_delta = output_error * sigmoid_derivative(output_layer[p])  	# applying derivative of sigmoid to error

				hidden_3_delta = np.zeros(len(self.weights_2[0]))  			# how much 3rd hidden layer weights contributed to output error
				for i in range(len(self.weights_2[0])):
					hidden_3_delta[i] = self.weights_3[i][0] * output_delta * relu_derivative(hidden_layer_3[p][i])

				hidden_2_delta = np.zeros(len(self.weights_1[0]))  			# how much 2nd hidden layer weights contributed to output error
				for i in range(len(self.weights_1[0])):
					hidden_2_delta[i] = np.sum([self.weights_2[i][k] * hidden_3_delta * relu_derivative(hidden_layer_2[p][i]) for k in range(len(self.weights_2[0]))])

				hidden_1_delta = np.zeros(len(self.weights_0[0]))  			# how much 1st hidden layer weights contributed to output error
				for i in range(len(self.weights_0[0])):
					hidden_1_delta[i] = np.sum([self.weights_1[i][k] * hidden_2_delta * relu_derivative(hidden_layer_1[p][i]) for k in range(len(self.weights_1[0]))])

				for i in range(len(self.weights_3)):  							# adjusting (3rd hidden --> output) weights
					self.weights_3[i][0] -= learning_rate * output_delta * hidden_layer_3[p][i]

				for i in range(len(self.weights_2)):  							# adjusting (2nd hidden --> 3rd hidden) weights
					for j in range(len(self.weights_2[0])):
						self.weights_2[i][j] -= learning_rate * hidden_3_delta[j] * hidden_layer_2[j][i]

				for i in range(len(self.weights_1)):  							# adjusting (1st hidden --> 2nd hidden) weights
					for j in range(len(self.weights_1[0])):
						self.weights_1[i][j] -= learning_rate * hidden_2_delta[j] * hidden_layer_1[j][i]

				for i in range(len(self.weights_0)):  							# adjusting (input --> 1st hidden) weights
					for j in range(len(self.weights_0[0])):
						self.weights_0[i][j] -= learning_rate * hidden_1_delta[j] * X_train[j][i]

			# forward propagation: update predictions
			hidden_layer_1 = self.relu(np.dot(X_train, self.weights_0))  			# relu as activation function
			hidden_layer_2 = self.relu(np.dot(hidden_layer_1, self.weights_1))  		# relu as activation function
			hidden_layer_3 = self.relu(np.dot(hidden_layer_2, self.weights_2))  		# relu as activation function
			output_layer = self.sigmoid(np.dot(hidden_layer_3, self.weights_3))  		# sigmoid as activation function

			error_history.append(cross_entropy(output_layer, y_train))
			acc_history.append(compute_accuracy(np.array([0 if p < 0.5 else 1 for p in output_layer]), y_train))

			# save best model
			if acc_history[-1] > best_accuracy:
				best_accuracy = acc_history[-1]
				self.best_weights_0 = self.weights_0
				self.best_weights_1 = self.weights_1
				self.best_weights_2 = self.weights_2
				self.best_weights_3 = self.weights_3

			if verbose:
				print('Error after training for', e, 'epochs:', error_history[-1])
				print('Training accuracy after training for', e, 'epochs:', acc_history[-1])

			# plot the distribution of latent features at different training stage
			if e % 20 == 0:
				print('Plotting latent features at epoch', e)
				if hidden_layer_3.shape[1] == 2:
					plot_latent_features_2D(hidden_layer_3, [0 if p < 0.5 else 1 for p in output_layer], e)
				elif hidden_layer_3.shape[1] == 3:
					plot_latent_features_3D(hidden_layer_3, [0 if p < 0.5 else 1 for p in output_layer], e)

		return error_history, acc_history

	def predict(self, input_matrix):
		hidden_layer_1 = self.relu(np.dot(input_matrix, self.best_weights_0))
		hidden_layer_2 = self.relu(np.dot(hidden_layer_1, self.best_weights_1))
		hidden_layer_3 = self.relu(np.dot(hidden_layer_2, self.best_weights_2))
		return self.sigmoid(np.dot(hidden_layer_3, self.best_weights_3))


def classification(X, y, learning_rate, num_epochs, verbose):

	# split data into training and test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, shuffle=True)

	# create neural network by specifying the number of neurons in each layer
	network = Classifier(X_train.shape[1], 10, 10, 3, 1)

	# fit neural network and return the loss history
	ce_history, acc_history = network.classify(np.array(X_train), np.array(y_train), learning_rate, num_epochs, verbose)

	# predict training accuracy and error
	train_predictions = np.array([network.predict(np.array(row)) for _, row in X_train.iterrows()])
	train_accuracy = compute_accuracy(np.array([0 if p < 0.5 else 1 for p in train_predictions]), np.array(y_train))

	# predict test accuracy and error
	test_predictions = np.array([network.predict(np.array(row)) for _, row in X_test.iterrows()])
	test_accuracy = compute_accuracy(np.array([0 if p < 0.5 else 1 for p in test_predictions]), np.array(y_test))

	if verbose:
		print('Training accuracy:', train_accuracy)
		print('Test accuracy:', test_accuracy)
		print('Training error:', (100 - train_accuracy) * 1 / 100)
		print('Test error:', (100 - test_accuracy) * 1 / 100)

		# plot cross-entropy curve
		plot_learning_curve(ce_history, 'cross_entropy')

		# plot accuracy curve
		plot_learning_curve(acc_history, 'accuracy')
