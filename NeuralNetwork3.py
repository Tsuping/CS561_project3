from email import header
import math
from operator import index
import os
import sys
from time import time
import pandas as pd
import numpy as np
import time


batch_size = 45
learning_rate = 0.01
epochs = 80

w1 = np.random.randn(512, 784) * np.sqrt(2.0 / (784 + 512))
b1 = np.random.randn(512,1) * np.sqrt(1.0 / 784)
w2 = np.random.randn(128, 512) * np.sqrt(2.0 / (512 + 128))
b2 = np.random.randn(128,1) * np.sqrt(1.0/ 512)
w3 = np.random.randn(10, 128) * np.sqrt(2.0 / (128 + 10))
b3 = np.random.randn(10,1) * np.sqrt(1.0 / 128)
def sigmoid_activation_function(x):
    sigmoid = np.clip(x, -500, 500)
    return 1.0/(1.0 + np.exp(-sigmoid))

def softmax_activation_function(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def sigmoid_activation_function_backward(val, X):
    sigmoid = sigmoid_activation_function(X)
    sigmoid_backward = val * sigmoid * (1 - sigmoid)
    return sigmoid_backward

def cross_entropy_loss_function(Y, X):
    loss_value = -1/Y.shape[1] * np.sum(np.multiply(Y, np.log(X)) + np.multiply(1 - Y, np.log(1 - X)))
    return loss_value


def minimum_batches(X, y, size):
    mini_batches = []
    batches_amount = math.floor(X.shape[1]/size)
    for w in range(0, batches_amount):
        batch_x_size = X[:, w * size : (w+1) * size]
        batch_y_size = y[:, w * size : (w+1) * size]
        mini_batches.append([batch_x_size, batch_y_size])

    if X.shape[1] % size != 0:
        batch_x_size = X[:, batch_size * math.floor(X.shape[1] / batch_size) : X.shape[1]]
        batch_y_size = y[:, batch_size * math.floor(X.shape[1] / batch_size) : X.shape[1]]
        mini_batches.append([batch_x_size, batch_y_size])

    return mini_batches


def train(X,y):
    global learning_rate
    global w1
    global b1
    global w2
    global b2
    global w3
    global b3
    global epochs
    global batch_size
    for j in range(0, epochs):
        random_number = np.arange(len(X[1]))
        np.random.shuffle(random_number)
        random_x = X[:,random_number]
        random_y = y[:,random_number]
        minimum_batch = minimum_batches(random_x, random_y, batch_size)
        for mins in minimum_batch:
            mbatch_x, mbatch_y = mins
            cache = dict()
            cache['Z1'] = np.dot(w1, mbatch_x) + b1
            cache['A1'] = sigmoid_activation_function(cache['Z1'])
            cache['Z2'] = np.dot(w2, cache['A1']) + b2
            cache['A2'] = sigmoid_activation_function(cache['Z2'])
            cache['Z3'] = np.dot(w3, cache['A2']) + b3
            cache['A3'] = softmax_activation_function(cache['Z3'])
         
            dW3 = np.dot(cache['A3'] - mbatch_y, cache["A2"].T) / mbatch_x.shape[1]
            db3 = np.sum(cache['A3'] - mbatch_y, axis=1, keepdims=True) / mbatch_x.shape[1]
            
            dZ2 = sigmoid_activation_function_backward(np.dot(w3.T, cache['A3'] - mbatch_y), cache['Z2'])

            dW2 = np.dot(dZ2, cache['A1'].T) / mbatch_x.shape[1]
            db2 = np.sum(dZ2, axis=1, keepdims=True) / mbatch_x.shape[1]
            
            dZ1 = sigmoid_activation_function_backward(np.dot(w2.T, dZ2), cache['Z1'])
            
            dW1 = np.dot(dZ1, mbatch_x.T) / mbatch_x.shape[1]
            db1 = np.sum(dZ1, axis=1, keepdims=True) / mbatch_x.shape[1]
            key = ['dW3', 'db3', 'dW2', 'db2', 'dW1', 'db1']
            value = [dW3, db3, dW2, db2, dW1, db1]
            w1 = w1 - (learning_rate * value[key.index('dW1')])
            b1 = b1 - (learning_rate * value[key.index('db1')])
            w2 = w2 - (learning_rate * value[key.index('dW2')])
            b2 = b2 - (learning_rate * value[key.index('db2')])
            w3 = w3 - (learning_rate * value[key.index('dW3')])
            b3 = b3 - (learning_rate * value[key.index('db3')])

            





train_image = pd.read_csv(sys.argv[1], header=None)
train_label = pd.read_csv(sys.argv[2], header=None)
test_image = pd.read_csv(sys.argv[3], header=None)
train_x = train_image.T
train_y = train_label.T
test_x = test_image.T
train_x = train_x.values
train_y = train_y.values
test_x = test_x.values


onehotY = np.zeros((train_y.size, 10))
onehotY[np.arange(train_y.size), train_y] = 1
onehotY = onehotY.T
starts = time.time()
print("beggining trainging: ")
train(train_x, onehotY)
print("start prediction: ")
cache = dict()
cache['Z1'] = np.dot(w1, test_x) + b1
cache['A1'] = sigmoid_activation_function(cache['Z1'])
cache['Z2'] = np.dot(w2, cache['A1']) + b2
cache['A2'] = sigmoid_activation_function(cache['Z2'])
cache['Z3'] = np.dot(w3, cache['A2']) + b3
cache['A3'] = softmax_activation_function(cache['Z3'])

output = cache['A3']
pred = np.argmax(output, axis=0)
pd.DataFrame(pred).to_csv('test_predictions.csv', header=None, index=None)
end = time.time()
