from net.model import FeedForwardNN
import numpy as np

data_path = 'examples/'
parameters_path = 'parameters/'
digit = 1

with np.load(f"{data_path}mnist_digit_{digit}.npz") as data:
    X = data['X']
    Y = data['Y']


np.random.seed(1)
np.random.shuffle(X.T)
np.random.seed(1)
np.random.shuffle(Y.T)


m_train = X.shape[1] - round(0.97 * X.shape[1])

X_train = X[:, :m_train]
Y_train = Y[:, :m_train]

X_test = X[:, m_train:]
Y_test = Y[:, m_train:]

print("Train shape: ", X_train.shape)
print("Test shape: ", X_test.shape)

# initialize an instance of neural network
net = FeedForwardNN(X_train, Y_train, layers=(32, 16, 1), batch=2000)
net.set_activations('tanh', 'sigmoid')
# net.set_dropout(1, 0.90)

print(net.cache['a0'].shape)

# run iterations
for i in range(1, 121):
    print(f"epoch {i: <4}: ", end='')
    net.full_circle(ALPHA=0.02)

# test neural network
net.predict(X_train, Y_train)
net.predict(X_test, Y_test)

# net.save_params(f"{parameters_path}mnist_params_{digit}")
