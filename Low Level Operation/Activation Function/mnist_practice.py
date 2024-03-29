import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras import datasets

(x_train, y_train), (x_val, y_val) = datasets.mnist.load_data()

def init_net():
    with open('/Users/dlwoals/Documents/Code/Deep Learning/Low Level Operation/Activation Function/sample_weight.pkl', 'rb') as f:
        net = pickle.load(f)
    
    return net

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x1):
    c = np.max(x1)
    return np.exp(x1 - c) / np.sum(np.exp(x1 - c))

def predict(net, x):
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

net = init_net()
accuracy_cnt = 0

for i in range(len(x_val)):
    y = predict(net, x_val[i])
    p = np.argmax(y)
    if p == y_val[i]:
        accuracy_cnt += 1

print('Accuracy: ' + str(float(accuracy_cnt) / len(x_val)))