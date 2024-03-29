import numpy as np

def step(x):
    y = x > 0
    return y.astype(int)

print(step(np.array([0.7, -0.1, 0.5])))