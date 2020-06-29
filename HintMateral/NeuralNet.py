#####################################################################################################################
#   Assignment 2: Neural Network Programming
#   This is a starter code in Python 3.6 for a 1-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   NeuralNet class init method takes file path as parameter and splits it into train and test part
#         - it assumes that the last column will the label (output) column
#   h - number of neurons in the hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   W_hidden - weight matrix connecting input to hidden layer
#   Wb_hidden - bias matrix for the hidden layer
#   W_output - weight matrix connecting hidden layer to output layer
#   Wb_output - bias matrix connecting hidden layer to output layer
#   deltaOut - delta for output unit (see slides for definition)
#   deltaHidden - delta for hidden unit (see slides for definition)
#   other symbols have self-explanatory meaning
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import requests
import io

# read data from public source
#url = "https://raw.githubusercontent.com/hip-hipJorge/ML-Assignment2/master/Assignment-2/ENB2012_data.csv"
url = "https://raw.githubusercontent.com/hip-hipJorge/ml6375-A1/master/car.data"

read_data = requests.get(url).content
# print formatting
np.set_printoptions(suppress=True, precision=3)

class NeuralNet:
    def __init__(self, dataFile, header=True, h=1):
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        raw_input = pd.read_csv(dataFile, delimiter="\t")
        processed_data = self.preprocess(raw_input)
        self.train_dataset, self.test_dataset = train_test_split(processed_data)
        ncols = len(self.train_dataset.columns)
        nrows = len(self.train_dataset.index)
        self.X = self.train_dataset.iloc[:, 0:(ncols-1)].values.reshape(nrows, ncols-1)
        self.y = self.train_dataset.iloc[:, ncols-1:ncols].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[1])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        self.h = h

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(x)
        if activation == "ReLu":
            self.__ReLu(x)

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(x)
        if activation == "ReLu":
            self.__ReLu_derivative(x)

    def __sigmoid(self, x):
        for i in range(x.shape[0]):
            x[i] = 1 / (1 + math.exp(-x[i]))
        return x

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __ReLu(self, x):
        return max(0, x)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)


    def __tanh_derivative(self, x):
        return ((2 * np.exp(2 * x) / (np.exp(2 * x) + 1)) * (1 - self.__tanh(x)))


    def __ReLu_derivative(self, x):
        if x.all() > 0:
            return 1
        elif x.all() < 0:
            return 0
        else:
            return math.nan

    def preprocess(self, X):
        for i in range(len(X.index)):
            X.loc[i] = list_format(list(X.iloc[i, 0:7]))
        return X

    # Below is the training function

    def train(self, max_iterations=1, learning_rate=0.25):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation="sigmoid")
            error = 0.5 * np.power((out - self.y), 2)
            # TODO: I have coded the sigmoid activation, you have to do the rest
            self.backward_pass(out, activation="sigmoid")

            update_weight_output = learning_rate * np.dot(self.X_hidden.T, self.deltaOut)
            print("shape of x_hidden.t: " + str(self.X_hidden.T.shape))
            update_weight_output_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)
            print("shape of delta out: " + str(self.deltaOut.shape))

            update_weight_hidden = learning_rate * np.dot(self.X.T, self.deltaHidden)
            print("update weight hidden:\n" + str(update_weight_hidden))

            update_weight_hidden_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden)
            print("shape of w_hidden: " + str(self.W_hidden.shape))
            print(self.W_hidden)
            self.W_output += float(update_weight_output)
            self.Wb_output += float(update_weight_output_b)
            np.add(self.W_hidden, update_weight_hidden)
            print(self.W_hidden)
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))

    def forward_pass(self, activation="sigmoid"):
        # pass our inputs through our neural network
        in_hidden = np.dot(self.X, self.W_hidden) + self.Wb_hidden
        # TODO: I have coded the sigmoid activation, you have to do the rest
        ## Hidden Node Pass
        if activation == "sigmoid":
            self.X_hidden = self.__sigmoid(in_hidden)
        if activation == 'tanh':
            self.X_hidden = self.__tanh(in_hidden)
        if activation == 'ReLu':
            self.X_hidden = self.__ReLu(in_hidden)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output

        ## Output Node Pass
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        if activation == 'tanh':
            out = self.__tanh(in_output)
        if activation == 'ReLu':
            out = self.__ReLu(in_output)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if activation == "ReLu":
            delta_output = (self.y - out) * (self.__ReLu_derivative(out))
        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden))
        if activation == "tanh":
            delta_hidden_layer = (self.deltaOut.dot(self.Wb_output.T)) * (self.__tanh_derivative(self.X_hidden))
        if activation == "ReLu":
            delta_hidden_layer = (self.deltaOut.dot(self.Wb_output.T)) * (self.__ReLu_derivative(self.X_hidden))
        self.deltaHidden = delta_hidden_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, header = True):
        # TODO: obtain prediction on self.test_dataset

        return 0


attr = {
    # buying/maint/safety
    'vhigh': 4,
    'high': 3,
    'med': 2,
    'low': 1,
    # lug_boot
    'big': 3,
    'small': 1,
    # doors/persons
    '2': 2,
    '3': 3,
    '4': 4,
    '5more': 5,
    # persons
    'more': 5,
    # class values
    'unacc': 1,
    'acc': 2,
    'good': 3,
    'vgood': 4
}


def list_format(lst):
    for i in range(len(lst)):
        lst[i] = float(attr[lst[i]])
    return lst


if __name__ == "__main__":
    df = io.StringIO(read_data.decode('utf-8'))
    '''
    dframe = pd.read_csv(df, delimiter="\t")
    print(dframe.shape)
    print(dframe.head())
    print()
    for i in range(len(dframe.index)):
        dframe.loc[i] = list_format(list(dframe.iloc[i, 0:7]))
    print(dframe.head())

    '''

    neural_network = NeuralNet(df)
    neural_network.train()
    #testError = neural_network.predict()
    #print("Test error = " + str(testError))


