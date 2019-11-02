import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from utilities import load_dataset, plot_learning_curve, plot_predictions
from model import Regretter


def rms_error(predictions, targets):
	return sqrt(1/len(predictions) * np.sum([(predictions[i] - targets[i]) ** 2 for i in range(len(predictions))]))


def one_hot_encode(df, categorical_columns, categories):				# one-hot encode categorical features
	for col in categorical_columns:
		df[col] = df[col].replace(categories)
	return pd.get_dummies(df, categorical_columns)


def correlation_selector(X, y):											# compute the correlation with y for each feature
	correlation = []
	for i in X.columns.tolist():
		cor = np.corrcoef(X[i], y)[0, 1]
		correlation.append(cor)
	return X.iloc[:, np.argsort(np.abs(correlation))].columns.tolist()


def preprocessing():
	categorical_columns = ['Orientation', 'Glazing Area Distribution']
	categories = {1: 'uniform', 2: 'north', 3: 'east', 4: 'south', 5: 'west'}
	target = 'Heating Load'
	df = load_dataset('EnergyEfficiency_data.csv')						# load data
	df = one_hot_encode(df, categorical_columns, categories)			# one hot encode categorical columns
	df = df.drop('Glazing Area Distribution_0', 1)
	df = (df - df.min()) / (df.max() - df.min())						# standardize data
	X = df.drop(target, 1)												# split data into training and test
	y = df[target]
	return X, y


def regression(X, y, learning_rate, num_epochs):

	# split data into training and test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17, shuffle=True)

	# create neural network by specifying the number of neurons in each layer
	network = Regretter(X_train.shape[1], 10, 5, 1)

	# fit neural network and return the loss history
	mse_history = network.regress(np.array(X_train), np.array(y_train), learning_rate, num_epochs, verbose=1)

	# predict training data and compute RMS error for training
	train_predictions = [network.predict(np.array(row)) for _, row in X_train.iterrows()]
	trainingRMS = rms_error(np.array(train_predictions), np.array(y_train))

	# predict test data and compute RMS error for test
	test_predictions = [network.predict(np.array(row)) for _, row in X_test.iterrows()]
	testRMS = rms_error(np.array(test_predictions), np.array(y_test))

	print('Training RMS error:', trainingRMS)
	print('Test RMS error:', testRMS)

	# plot curves
	plot_learning_curve(mse_history, 'mse')
	plot_predictions(train_predictions, y_train, 'training')
	plot_predictions(test_predictions, y_test, 'test')
