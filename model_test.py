# -*- coding: utf-8 -*-
from model import FeedForwardNN
import numpy as np
from PIL import Image

with np.load('parameters_cd.npz') as data:
    w1 = data['w1']
    b1 = data['b1']
    w2 = data['w2']
    b2 = data['b2']
    w3 = data['w3']
    b3 = data['b3']
    w4 = data['w4']
    b4 = data['b4']
    w5 = data['w5']
    b5 = data['b5']

params_dict = {'w1': w1, 'b1': b1,
               'w2': w2, 'b2': b2,
               'w3': w3, 'b3': b3,
               'w4': w4, 'b4': b4,
               'w5': w5, 'b5': b5}

img = Image.open('cat2.jpeg').resize((400, 400))
arr = np.array(img)
X = arr.reshape(arr.shape[0] * arr.shape[1] * arr.shape[2], 1)
Y = np.array([[1.0]])

catNet = FeedForwardNN(X, Y, layers=(128, 90, 60, 32, 1),
                       params=params_dict)


catNet.all_forward()
catNet.compute_cost()
catNet.print_prediction()
