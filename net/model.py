import numpy as np
import math
from PIL import Image


class FeedForwardNN():
    # params: dict[str, ndarray] = dict['w1': w1, 'w2': w2, ...]
    # layers: tuple of int
    def __init__(self, X, Y, layers=(10, 5, 1), batch=None, params=None):
        self.batch = X.shape[1] if batch is None else batch
        self.t = 0

        self.cache = {'a0': X}
        self.Y = Y

        self.m = X.shape[1]
        self.layers = (X.shape[0], *layers)
        self.layers_num = len(self.layers)
        self.parameters = {}
        self.grads = {}
        self.dropout_layers = {}
        self.activations = ('relu', 'sigmoid')
        self.current_m = None

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
        print('Drop-out is set')
        print('--'*20)

    def set_activations(self, func_name_1, func_name_2):
        self.activations = (func_name_1, func_name_2)
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

    def batch_forward(self):
        '''Getting a batch and calculating its length for the case when it's
        the last batch and hence its length may be smaller the self.batch'''
        a = self.cache['a0'][:, self.t:self.t+self.batch]
        self.current_m = len(a[0])

        # Forward prop for all layers except the last --------------------
        for i in range(1, self.layers_num - 1):
            z, a = self.forward_step(a,
                                     self.parameters['w'+str(i)],
                                     self.parameters['b'+str(i)],
                                     i,
                                     self.activations[0])
            self.cache['z'+str(i)] = z
            self.cache['a'+str(i)] = a
        # ----------------------------------------------------------------

        # Last layer forward prop ----------------------------------------
        z, a = self.forward_step(a,
                                 self.parameters['w'+str(self.layers_num-1)],
                                 self.parameters['b'+str(self.layers_num-1)],
                                 self.layers_num - 1,
                                 self.activations[1])
        self.cache['z'+str(self.layers_num-1)] = z
        self.cache['a'+str(self.layers_num-1)] = a
        # ----------------------------------------------------------------

    def compute_batch_cost(self):
        m = self.current_m
        st = self.t
        ed = self.t + m

        y_hat = self.cache['a'+str(self.layers_num-1)]

        cost = (-1 / m) * (np.dot(self.Y[:, st:ed], np.log(y_hat).T)
                           + np.dot((1 - self.Y[:, st:ed]),
                                    np.log(1 - y_hat).T))
        return cost

    def backward_step(self, da, activation, current_layer):
        z = self.cache['z'+str(current_layer)]
        a = self.cache['a'+str(current_layer)]

        if current_layer != 1:
            a_prev = self.cache['a'+str(current_layer-1)]
        else:
            a_prev = self.cache['a0'][:, self.t:self.t+self.current_m]

        if activation == 'relu':
            dz = np.array(da, copy=True)
            dz[z <= 0] = 0
        elif activation == 'tanh':
            dz = da * (1 - np.power(a, 2))
        elif activation == 'sigmoid':
            dz = da * a * (1 - a)

        self.grads['dw'+str(current_layer)] = ((1 / self.current_m)
                                               * np.dot(dz, a_prev.T))

        self.grads['db'+str(current_layer)] = ((1 / self.current_m)
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

    def batch_backward(self):
        st = self.t
        ed = self.t + self.current_m

        y_hat = self.cache['a'+str(self.layers_num-1)]
        da_l = (- (np.divide(self.Y[:, st:ed], y_hat)) +
                np.divide(1 - self.Y[:, st:ed], 1 - y_hat))

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

    def full_circle(self, ALPHA=0.01):

        batches_num = 0
        cost_average = 0
        while self.t < self.cache['a0'].shape[1]:
            self.batch_forward()
            batch_cost = self.compute_batch_cost()[0, 0]
            self.batch_backward()
            self.update_parameters(ALPHA=ALPHA)
            self.t += self.batch
            batches_num += 1
            cost_average += batch_cost
        self.t = 0

        print(f"    Circle cost: {'%.5f' % (cost_average / batches_num)}")

    def predict(self, X_test, Y_test):
        self.cache['a0'] = X_test
        self.Y = Y_test
        self.m = X_test.shape[1]

        batches_num = 0
        cost_average = 0
        self.t = 0
        predictions = []

        while self.t < self.cache['a0'].shape[1]:
            self.batch_forward()
            batch_cost = self.compute_batch_cost()
            prediction = self.cache['a'+str(self.layers_num-1)] > 0.5
            prediction = prediction.tolist()[0]
            predictions.extend(prediction)

            self.t += self.batch
            batches_num += 1
            cost_average += batch_cost


        # print('Preditions length: ', len(predictions))
        correct = np.sum(predictions == self.Y)
        accuracy = correct / self.Y.shape[1] * 100

        print('Model accuracy: ', '%.2f' % accuracy, ' %')
        print('--'*20)

    def save_params(self, fname):
        np.savez(fname, **self.parameters)
