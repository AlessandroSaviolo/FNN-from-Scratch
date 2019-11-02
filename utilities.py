import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_dataset(path):
	return pd.read_csv(path)


def plot_learning_curve(history, title):
	plt.figure(figsize=[8, 6])
	plt.plot(history, 'r', linewidth=3.0)
	plt.xlabel('epoch', fontsize=16)
	plt.ylabel(title, fontsize=16)
	plt.title('training curve', fontsize=16)
	plt.savefig(title + '.png')
	plt.show()


def plot_predictions(predictions, labels, title):
	x_axis = np.arange(len(predictions))
	plt.figure(figsize=[8, 6])
	plt.plot(x_axis, labels, 'b', linewidth=1, label='label')
	plt.plot(x_axis, predictions, 'r', linewidth=1, label='predict')
	plt.legend(loc='upper left')
	plt.xlabel('Instance', fontsize=12)
	plt.ylabel('Heating Load', fontsize=12)
	plt.title('prediction for ' + title + ' data', fontsize=12)
	plt.savefig(title + '.png')
	plt.show()


def plot_latent_features_2D(latent_features, predictions, epoch):
	plt.figure(figsize=[8, 6])

	class1, class2 = [], []
	for i in range(latent_features.shape[0]):
		class1.append(tuple(latent_features[i, :])) if predictions[i] else class2.append(tuple(latent_features[i, :]))
	class1 = np.array(class1)
	class2 = np.array(class2)

	if class1.shape[0]:
		plt.scatter(class1[:, 0], class1[:, 1], c='b', label='Class 1')
	if class2.shape[0]:
		plt.scatter(class2[:, 0], class2[:, 1], c='r', label='Class 2')

	plt.legend(loc='upper right')
	plt.title('latent features at epoch ' + str(epoch), fontsize=12)
	plt.savefig('latentfeatures/latent_features_epoch_' + str(epoch) + '.png')
	plt.show()


def plot_latent_features_3D(latent_features, predictions, epoch):
	fig = plt.figure(figsize=[8, 6])
	ax = Axes3D(fig)

	class1, class2 = [], []
	for i in range(latent_features.shape[0]):
		class1.append(tuple(latent_features[i, :])) if predictions[i] else class2.append(tuple(latent_features[i, :]))

	class1 = np.array(class1)
	class2 = np.array(class2)

	if class1.shape[0]:
		ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='b', label='Class 1')

	if class2.shape[0]:
		ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='r', label='Class 2')

	plt.legend(loc='upper right')
	plt.title('latent features at epoch ' + str(epoch), fontsize=12)
	plt.savefig('latentfeatures/latent_features_epoch_' + str(epoch) + '.png')
	plt.show()
