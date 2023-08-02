from net.model import FeedForwardNN
import numpy as np

data_path = 'examples/'
parameters_path = 'parameters/'
digit = 3

with np.load(f"{data_path}mnist_digit_{digit}.npz") as data:
    X = data['X']
    Y = data['Y']


np.random.seed(1)
np.random.shuffle(X.T)
np.random.seed(1)
np.random.shuffle(Y.T)

# percent of data for testing
TEST_SIZE = 0.10

m_train = X.shape[1] - round(TEST_SIZE * X.shape[1])

X_train = X[:, :m_train]
Y_train = Y[:, :m_train]

X_test = X[:, m_train:]
Y_test = Y[:, m_train:]

print("Train shape: ", X_train.shape)
print("Test shape: ", X_test.shape)

# initialize an instance of neural network
net = FeedForwardNN(X_train, Y_train, layers=(64, 32, 16, 16, 8, 1), batch=2000)
net.set_activations('tanh', 'sigmoid')
# net.set_dropout(1, 0.90)

print(net.cache['a0'].shape)

# run iterations
for i in range(1, 221):
    print(f"epoch {i: <4}: ", end='')
    net.full_circle(ALPHA=0.04)

# test neural network
net.predict(X_train, Y_train)
net.predict(X_test, Y_test)

# net.save_params(f"{parameters_path}mnist_params_{digit}")
