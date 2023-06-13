import numpy as np
import math


class FeedForwardNN():

    def __init__(self, X, Y, layers=(100, 10, 1), bath=1000):
        self.X = X
        self.Y = Y
        self.m = X.shape[1]
        self.bath = bath
        self.layers = (X.shape[0], *layers)
        self.layers_num = len(self.layers)
        self.parameters = {}
        self.cache = {'a0': self.X}
        self.grads = {}

        np.random.seed(3)

        # initializing and print parameters ----------------------------------
        for i in range(1, self.layers_num):
            w = np.random.randn(self.layers[i], self.layers[i-1]) \
                                    * math.sqrt(1 / self.layers[i-1])
            b = np.random.randn(self.layers[i], 1)

            print(f"w{i}_shape = {w.shape}")
            print(f"b{i}_shape = {b.shape}")

            self.parameters['w'+str(i)] = w
            self.parameters['b'+str(i)] = b
        # --------------------------------------------------------------------

    def forward_step(self, prev_A, w, b, activation):
        z = np.dot(w, prev_A) + b

        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            a = z * (z > 0)
        elif activation == 'tanh':
            a = np.tanh(z)
        else:
            a = z

        return z, a

    def all_forward(self):
        a = self.cache['a0']

        # Forward prop for all layers except the last ------------------------
        for i in range(1, self.layers_num - 1):
            z, a = self.forward_step(a,
                                     self.parameters['w'+str(i)],
                                     self.parameters['b'+str(i)],
                                     'tanh')
            self.cache['z'+str(i)] = z
            self.cache['a'+str(i)] = a
        # --------------------------------------------------------------------

        # Last layer forward prop --------------------------------------------
        z, a = self.forward_step(a,
                                 self.parameters['w'+str(self.layers_num-1)],
                                 self.parameters['b'+str(self.layers_num-1)],
                                 'sigmoid')
        self.cache['z'+str(self.layers_num-1)] = z
        self.cache['a'+str(self.layers_num-1)] = a
        # --------------------------------------------------------------------

    def compute_cost(self):
        y_hat = self.cache['a'+str(self.layers_num-1)]
        cost = (-1 / self.m) * (np.dot(self.Y, np.log(y_hat).T)
                                + np.dot((1 - Y), np.log(1 - y_hat).T))
        print(cost)

    def backward_step(self, da, activation, layer_i):
        z = self.cache['z'+str(layer_i)]
        a = self.cache['a'+str(layer_i)]
        a_prev = self.cache['a'+str(layer_i-1)]

        if activation == 'relu':
            dz = np.array(da, copy=True)
            dz[z <= 0] = 0
        elif activation == 'tanh':
            dz = da * (1 - np.power(a, 2))
        elif activation == 'sigmoid':
            dz = da * a * (1 - a)

        self.grads['dw'+str(layer_i)] = (1 / self.m) * np.dot(dz, a_prev.T)
        self.grads['db'+str(layer_i)] = (1 / self.m) * np.sum(dz, axis=1,
                                                              keepdims=True)

        # don't need to calculate da for the first layer (for a0 = X)
        if layer_i != 1:
            da_prev = np.dot(self.parameters['w'+str(layer_i)].T, dz)
        else:
            da_prev = None

        return da_prev

    def all_backward(self):
        y_hat = self.cache['a'+str(self.layers_num-1)]
        da_l = - (np.divide(self.Y, y_hat)) + np.divide(1 - self.Y, 1 - y_hat)

        # First and the others steps of back propagation ---------------------
        da_prev = self.backward_step(da_l, 'sigmoid', self.layers_num-1)
        for i in reversed(range(1, self.layers_num-1)):
            da_prev = self.backward_step(da_prev, 'tanh', i)
        # --------------------------------------------------------------------

    def update_parameters(self, ALPHA=0.01):
        for i in range(1, self.layers_num):
            self.parameters['w'+str(i)] -= ALPHA * self.grads['dw'+str(i)]
            self.parameters['b'+str(i)] -= ALPHA * self.grads['db'+str(i)]


# load data
# assert that X.shape = (nX, m)
with np.load('mnist_data.npz') as data:
    X = data['X']
    Y = data['Y']

# initialize an instance of neural network
net = FeedForwardNN(X, Y, layers=(20, 10, 1))

# run iterations
for i in range(20):
    net.all_forward()
    net.compute_cost()
    net.all_backward()
    net.update_parameters(ALPHA=0.05)
