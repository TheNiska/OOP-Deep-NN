from PIL import Image
import numpy as np
import PIL.ImageOps as imOp
from net.model import FeedForwardNN

fname = "test_image.png"
parameters_path = 'parameters/'

name, ext = fname.split('.')

img = Image.open(fname).convert('L').resize((28, 28))
img = imOp.invert(img)
img.save(name + "_grey." + ext)

X = np.array(img.getdata()).reshape(784, 1)
X = (X - 127.5) / 127.5
Y = np.array([[1]])

digit = 0

full_path = f"{parameters_path}mnist_params_{digit}.npz"
dict_params = np.load(full_path)

net = FeedForwardNN(X, Y, layers=(8, 4, 1), params=dict_params)
probabl = net.show_probability()
print("Probability of 0 is ", probabl)
