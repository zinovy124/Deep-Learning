import numpy as np

def softmax(x1):
    return np.exp(x1) / np.sum(np.exp(x1))

def softmax_modidied(x1):
    c = np.max(x1)
    return np.exp(x1 - c) / np.sum(np.exp(x1 - c))

print(softmax(np.array([1, 2, 3])))
print(softmax_modidied(np.array([1, 2, 3])))
print(softmax_modidied(np.array([1010, 1000, 990])))