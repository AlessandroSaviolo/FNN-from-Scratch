import numpy as np
from sklearn.model_selection import train_test_split
from utilities import load_dataset, plot_learning_curve
from model import Classifier, compute_accuracy


def preprocessing():
	df = load_dataset('ionosphere_csv.csv')						# load data
	X = (df.drop(['class'], 1))
	X = (X - X.min()) / (X.max() - X.min())  					# standardize data
	X = X.replace(np.NaN, 0)
	y = df['class'].transform(lambda x: 1 if x is 'g' else 0)			# one hot encode target
	return X, y


def classification(X, y, learning_rate, num_epochs, verbose):

	# split data into training and test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, shuffle=True)

	# create neural network by specifying the number of neurons in each layer
	network = Classifier(X_train.shape[1], 15, 1)

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
