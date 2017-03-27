import sklearn.preprocessing
import numpy as np
from scipy import linalg

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

"""
When grading, We only run this function, So don\'t make any global variable.
Actually you can use global variable, but do not use that in unrelated function to the execution of this function
You can implement other function and use it.
And of course you can\'t change the function name
Just know that we execute only this function(profile_log_likelihood) when grading!!
"""


def main():
    random_data = np.ndarray((50, 100))
    for i in range(50):
        random_data[i] = np.random.uniform(0, 50, 100)
    scaler = sklearn.preprocessing.MinMaxScaler()
    data = scaler.fit_transform(random_data)

    armax = profile_log_likelihood(data)
    print(armax)


def get_eigenvalue(data):
    mu = np.mean(data, axis=0)
    U, s, V = linalg.svd(data - mu)
    return s * s


def profile_log_likelihood(X):
    # implement profile likelihood function
    # input description:
    # input: Data X
    # X.shape = (N, D).
    # N is number of data, D is dimension of data
    # return optimal number of component value in integer.
    N = X.shape[0]
    e_val = get_eigenvalue(X)

    l_val = []
    for L in range(N):
        u1 = np.sum(e_val[:L])/L
        u2 = np.sum(e_val[L:])/(N-L)
        o2 = (np.sum((e_val[:L] - u1)**2) + np.sum((e_val[L:] - u2)**2))/N
        la = np.sum(np.log(1 / (np.sqrt(2*np.pi*o2)) * np.exp(-(e_val[:L]-u1)**2/(2*o2))))
        lb = np.sum(np.log(1 / (np.sqrt(2 * np.pi * o2)) * np.exp(-(e_val[L:] - u2) ** 2 / (2 * o2))))
        l_val.append(la+lb)
    np.array(l_val)
    return np.argmax(l_val)


if __name__ == "__main__":
    main()
