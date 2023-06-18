import numpy as np
import math
from PIL import Image


class FeedForwardNN():
    # params: dict[str, ndarray] = dict['w1': w1, 'w2': w2, ...]
    # layers: tuple of int
    def __init__(self, X, Y, layers=(10, 5, 1), bath=None, params=None):
        self.cache = {'a0': X}
        self.Y = Y
        self.m = X.shape[1]
        self.bath = bath
        self.layers = (X.shape[0], *layers)
        self.layers_num = len(self.layers)
        self.parameters = {}
        self.grads = {}
        self.dropout_layers = {}
        self.activations = ('relu', 'sigmoid')

        np.random.seed(3)

        info_str = "Model settings"
        print(f"----{info_str:-<36}")
        if not params:
            # initializing and print parameters ------------------------------
            for i in range(1, self.layers_num):
                w = np.random.randn(self.layers[i], self.layers[i-1]) \
                                        * math.sqrt(1 / self.layers[i-1])
                b = np.random.randn(self.layers[i], 1)

                print(f"w{i}_shape = {w.shape}")

                self.parameters['w'+str(i)] = w
                self.parameters['b'+str(i)] = b
            # ----------------------------------------------------------------
        else:
            # get parameters from user
            self.parameters = params
        print(self.layers)
        print('--'*20)

    # set-up of drop-out regularization
    def set_dropout(self, layer_i, keep_prob):
        self.dropout_layers[layer_i] = keep_prob
        print('--'*20)
        print('Drop-out is set')
        print('--'*20)

    def set_activations(self, func_name_1, func_name_2):
        self.activations = (func_name_1, func_name_2)
        print('--'*20)
        print('Activations is set')
        print('--'*20)

    def forward_step(self, prev_A, w, b, current_layer, activation):
        z = np.dot(w, prev_A) + b

        # calculating activation ---------------------------------------------
        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            a = z * (z > 0)
        elif activation == 'tanh':
            a = np.tanh(z)
        else:
            a = z
        # --------------------------------------------------------------------

        # drop-out regularization --------------------------------------------
        keep_prob = self.dropout_layers.get(current_layer)
        if keep_prob:
            drop_mask = np.random.rand(*a.shape) < keep_prob
            a *= drop_mask
            a /= keep_prob

            self.cache['drop_mask_'+str(current_layer)] = drop_mask
        # --------------------------------------------------------------------

        return z, a

    def all_forward(self):
        a = self.cache['a0']

        # Forward prop for all layers except the last ------------------------
        for i in range(1, self.layers_num - 1):
            z, a = self.forward_step(a,
                                     self.parameters['w'+str(i)],
                                     self.parameters['b'+str(i)],
                                     i,
                                     self.activations[0])
            self.cache['z'+str(i)] = z
            self.cache['a'+str(i)] = a
        # --------------------------------------------------------------------

        # Last layer forward prop --------------------------------------------
        z, a = self.forward_step(a,
                                 self.parameters['w'+str(self.layers_num-1)],
                                 self.parameters['b'+str(self.layers_num-1)],
                                 self.layers_num - 1,
                                 self.activations[1])
        self.cache['z'+str(self.layers_num-1)] = z
        self.cache['a'+str(self.layers_num-1)] = a
        # --------------------------------------------------------------------

    def compute_cost(self):
        y_hat = self.cache['a'+str(self.layers_num-1)]
        cost = (-1 / self.m) * (np.dot(self.Y, np.log(y_hat).T)
                                + np.dot((1 - self.Y), np.log(1 - y_hat).T))
        print(cost)

    def backward_step(self, da, activation, current_layer):
        z = self.cache['z'+str(current_layer)]
        a = self.cache['a'+str(current_layer)]
        a_prev = self.cache['a'+str(current_layer-1)]

        if activation == 'relu':
            dz = np.array(da, copy=True)
            dz[z <= 0] = 0
        elif activation == 'tanh':
            dz = da * (1 - np.power(a, 2))
        elif activation == 'sigmoid':
            dz = da * a * (1 - a)

        self.grads['dw'+str(current_layer)] = ((1 / self.m)
                                               * np.dot(dz, a_prev.T))

        self.grads['db'+str(current_layer)] = ((1 / self.m)
                                               * np.sum(dz, axis=1,
                                                        keepdims=True))

        # don't need to calculate da for the first layer (for a0 = X)
        if current_layer != 1:
            # this calculates da of layer[current_layer-1]
            da_prev = np.dot(self.parameters['w'+str(current_layer)].T, dz)

            # backward prop drop-out -----------------------------------------
            keep_prob = self.dropout_layers.get(current_layer - 1)
            if keep_prob:
                da_prev *= self.cache['drop_mask_'+str(current_layer - 1)]
                da_prev /= keep_prob
            # ----------------------------------------------------------------
        else:
            da_prev = None

        return da_prev

    def all_backward(self):
        y_hat = self.cache['a'+str(self.layers_num-1)]
        da_l = - (np.divide(self.Y, y_hat)) + np.divide(1 - self.Y, 1 - y_hat)

        # First and the others steps of back propagation ---------------------
        da_prev = self.backward_step(da_l,
                                     self.activations[1],
                                     self.layers_num-1)

        for i in reversed(range(1, self.layers_num-1)):
            da_prev = self.backward_step(da_prev, self.activations[0], i)
        # --------------------------------------------------------------------

    def update_parameters(self, ALPHA=0.01):
        for i in range(1, self.layers_num):
            self.parameters['w'+str(i)] -= ALPHA * self.grads['dw'+str(i)]
            self.parameters['b'+str(i)] -= ALPHA * self.grads['db'+str(i)]

    def predict(self, X_test, Y_test):
        print('--'*20)
        self.cache['a0'] = X_test
        self.Y = Y_test
        self.m = X_test.shape[1]
        self.all_forward()
        print('Test cost: ', end='')
        self.compute_cost()

        predictions = self.cache['a'+str(self.layers_num-1)] > 0.5
        correct = np.sum(predictions == self.Y)
        accuracy = correct / self.Y.shape[1] * 100
        print('Model accuracy: ', '%.2f' % accuracy, ' %')
        print('--'*20)

    def visualize_params(self, path):
        w = self.parameters['w1']
        mn = np.amin(w)
        w = w - mn
        mx = np.amax(w)  # 0 .. mx
        w = w * (255 / mx)  # 0 .. 255

        for i in range(w.shape[0]):
            arr = w[i, :].reshape(28, 28)
            arr = np.array(arr, dtype='uint8')
            img = Image.fromarray(arr)
            img = img.resize((196, 196))
            img.save(f"{path}{i}.png")

    def save_params(self, fname):
        np.savez(fname, **self.parameters)
