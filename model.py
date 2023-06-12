import numpy as np
import math


class FeedForwardNN():
    def __init__(self, X=None, Y=None, layers=(100, 10, 1)):
        self.X = X
        self.Y = Y
        self.m = X.shape[1]
        self.layers_num = len(layers)
        self.parameters = {}

        for i in range(1, self.layers_num):
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
        a = a0

        for i in range(1, self.layers_num - 1):
            z, a = self.forward_step(a,
                                     self.parameters['w'+str(i)],
                                     self.parameters['b'+str(i)],
                                     'relu')
            self.cache['z'+str(i)] = z
            self.cache['a'+str(i)] = a

        z, a = self.forward_step(a,
                                 self.parameters['w'+str(self.layers_num-1)],
                                 self.parameters['b'+str(self.layers_num-1)],
                                 'sigmoid')
        self.cache['z'+str(self.layers_num-1)] = z
        self.cache['a'+str(self.layers_num-1)] = a

    def compute_cost(self):
        y_hat = self.cache['a'+str(self.layers_num-1)]
        cost = ((-1 / self.m) * (np.dot(self.Y, np.log(y_hat).T)
                                 + np.dot((1 - Y), np.log(1 - y_hat).T)))
        print(cost)


with np.load('mnist_data.npz') as data:
    X = data['X']
    Y = data['Y']

INPUT_SIZE, M = X.shape

net = FeedForwardNN(X=X, Y=Y, layers=(INPUT_SIZE, 20, 10, 1))
net.all_forward(X)
net.compute_cost()

for key in net.parameters:
    print(f"{key} :: {net.parameters[key].shape}")
for key in net.cache:
    print(f"{key} :: {net.cache[key].shape}")
