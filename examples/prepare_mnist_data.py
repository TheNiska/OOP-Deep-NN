import numpy as np
import pandas as pd


# function to make labels data binary
def y_to_one_number(y, num):
    isNum = y[:, :] == num
    new_y = (y + 1) * isNum // (num + 1)
    return new_y


def from_csv_to_npz(number=0, X=None, Y=None):
    '''
    # reading and changing the shape of the data
    df = pd.read_csv('train.csv')
    data = df.to_numpy().T

    # making correct X and Y for training
    X = data[1:, :]
    Y = data[0, :].reshape(1, 42000)

    # normalizing X
    X = (X - 127.5) / 127.5
    '''

    # binary Y only for one number
    Y = y_to_one_number(Y, number)

    np.savez(f'mnist_data_for_{number}', X=X, Y=Y)


def make_one_class_data(number=0):
    with np.load(f'mnist_data_for_{number}.npz') as data:
        X_data = data['X']
        Y_data = data['Y']

    '''We want the number of 1's to be equal to the number of 0's in true label
    Y, but currently there are 90 % of ones and 10% of zeros because we are
    doing binary classification on mnist data that has not 2 classes but ten.
    '''
    m_size = np.sum(Y_data) * 2
    X = np.zeros((X_data.shape[0], m_size))
    Y = np.zeros((1, m_size))

    k = 0

    for i in range(X_data.shape[1]):
        if Y_data[:, i] == 1:
            X[:, k] = X_data[:, i]
            Y[:, k] = 1
            k += 1
            if k == (m_size // 2):
                break

    for i in range(X_data.shape[1]):
        if Y_data[:, i] == 0:
            X[:, k] = X_data[:, i]
            Y[:, k] = 0
            k += 1
            if k == m_size:
                break

    np.savez(f'mnist_digit_{number}', X=X, Y=Y)


if __name__ == "__main__":
    # reading and changing the shape of the data
    df = pd.read_csv('train.csv')
    data = df.to_numpy().T

    # making correct X and Y for training
    X = data[1:, :]
    Y = data[0, :].reshape(1, 42000)

    # normalizing X
    X = (X - 127.5) / 127.5

    for i in range(10):
        from_csv_to_npz(number=i, X=X, Y=Y)
        print('-------------------')
        make_one_class_data(number=i)
        print(f"Done {(i+1) / 10 * 100} %")
