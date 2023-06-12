import numpy as np
import math


class FeedForwardNN():
    def __init__(self, layers=(100, 10, 1)):
        self.parameters = {}

        for i in range(1, len(layers)):
            w = (np.random.randn(layers[i], layers[i-1])
                 * math.sqrt(1 / layers[i-1]))
            b = np.zeros((layers[i], 1))

            self.parameters['w'+str(i)] = w
            self.parameters['b'+str(i)] = b
            self.cache = {}

    def forward_step(self, prev_A, w, b, activation):
        z = np.dot(w, prev_A) + b

        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            a = z * (z > 0)
        else:
            a = z

        return z, a

    def all_forward(self, a0):
        layers_num = len(self.parameters) // 2
        a = a0

        for i in range(1, layers_num):
            z, a = self.forward_step(a,
                                     self.parameters['w'+str(i)],
                                     self.parameters['b'+str(i)],
                                     'relu')
            self.cache['z'+str(i)] = z
            self.cache['a'+str(i)] = a

        z, a = self.forward_step(a,
                                 self.parameters['w'+str(layers_num)],
                                 self.parameters['b'+str(layers_num)],
                                 'sigmoid')
        self.cache['z'+str(layers_num)] = z
        self.cache['a'+str(layers_num)] = a


with np.load('mnist_data.npz') as data:
    X = data['X']
    Y = data['Y']

net = FeedForwardNN((784, 32, 16, 8, 4, 1))
net.all_forward(X)

for key in net.parameters:
    print(f"{key} :: {net.parameters[key].shape}")
for key in net.cache:
    print(f"{key} :: {net.cache[key].shape}")
