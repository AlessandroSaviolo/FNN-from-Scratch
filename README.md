# DeepLearning_0845086_HW1

The purpose of this document is to present and analyze the neural network models built from scratch for Homework 1. Furthermore, the results produced by such networks on different datasets are illustrated.
The first model analyzed is the neural network for regression. The model architecture consists of four layers. The two hidden layers are used to obtain better performances, at the cost of more computational time. It uses the ReLU activation function for all layers except for the output layer (no activation function there). The neural network uses Gradient Descent in order to reduce the loss function which is the Mean Squared Error.
The second model analyzed is the neural network for classification. The model architecture consists of 3 layers. This choice is related to the complexity of the dataset. It uses the ReLU activation function for all layers except for the output layer which uses the Sigmoid. The neural network uses Gradient Descent in order to reduce the loss function which is the Cross-Entropy.

The code is organized as follows:

-> main.py : main function, use it to change task ('r' or 'c') and hyperparameters (i.e., learning rate, number of epochs)
-> model.py : contains the regression and classification neural network models
-> regression.py : run regression using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)
-> classification.py : run classification using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)
-> utilities.py : contains plot functions and common functions among the different files (i.e., load dataset which is used both for regression and classification)
-> deep_classification.py : deep classifier used to plot the distribution of latent features at different training stages. It contains also the deep model
