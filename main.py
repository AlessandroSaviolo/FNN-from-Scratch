import classification
import regression


if __name__ == '__main__':

	task = 'r'

	learning_rate = 0.0005  												# hyper parameters
	num_epochs = 4000
	num_features = 5

	if task == 'r':
		X, y = regression.preprocessing()									# preprocess data
		regression.regression(X, y, learning_rate, num_epochs)				# regression
		ranked_features = regression.correlation_selector(X, y)  			# sort features based on their correlation
		best_features = ranked_features[-num_features:]  					# select the best features
		X = X.drop(ranked_features[:num_features], 1) 						# drop non-selected features
		regression.regression(X, y, learning_rate, num_epochs)  			# regression with selected features

	elif task == 'c':
		X, y = classification.preprocessing()												# preprocess data
		classification.classification(X, y, learning_rate, num_epochs, verbose=1)  		# classification

	else:
		print('unknown task, please select classification or regression tasks')
