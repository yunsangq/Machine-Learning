import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from matplotlib import pylab as plt

coef = 1e-9
d = 3

# *******************************
# Please fill in the blanks below
# *******************************
def poly_kernel(x1, x2):
    return (coef*np.dot(x1, x2) + 1)**d # Use the coef and d variable defined above!

# Functions nonlinear_func1, nonlinear_func2, and generate_data is used to generate random datapoints. You do not need to change this function
def nonlinear_func1(x):
    l = len(x)
    return (0.1 * pow(x, 3) + 0.2 * pow(x, 2) + 0.3 * x + 1000 + 3000 * np.random.random(l))

def nonlinear_func2(x):
    l = len(x)
    return (0.1 * pow(x, 3) + 0.2 * pow(x, 2) - 0.3 * x - 1000 - 3000 * np.random.random(l))

def generate_data(n):
    np.random.seed(32840091)

    x1_1 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_1 = nonlinear_func1(x1_1)
    x1_2 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_2 = nonlinear_func2(x1_2)
    y_1 = np.ones(int(0.5 * n))
    y_2 = -1 * np.ones(int(0.5 * n))

    x1 = np.concatenate((x1_1, x1_2))
    x2 = np.concatenate((x2_1, x2_2))
    y = np.concatenate((y_1, y_2))
    X = np.array(list(zip(x1, x2)))

    return (X, y)

# *******************************
# Please fill in the blanks below
# *******************************
def svm(X, y):
    n = len(y)

    # N x N Gram matrix
    K = np.zeros((n, n))
    # inner products
    for i in range(n):
        for j in range(n):
            K[i, j] = poly_kernel(X[i], X[j])

    P = matrix(np.outer(y, y) * K, tc='d')
    q = matrix(np.ones(n) * -1, tc='d')

    G = matrix(np.diag(np.ones(n) * -1), tc='d')
    h = matrix(np.zeros(n), tc='d')

    A = matrix(y, (1, n), tc='d')
    b = matrix(0.0, tc='d')

    opts={'show_progress' : False}
    solution = solvers.qp(P, q, G, h, A, b, options=opts)

    # Lagrange multipliers
    a = np.ravel(solution['x'])
    # non-zero alphas
    ssv = a > 1e-5
    # get non-zero alphas support vectors
    support_vec_idx = np.arange(len(a))[ssv]
    # select alphas, support vectors, class labels
    a = a[ssv]
    _x = X[ssv]
    _y = y[ssv]

    # b using any SV
    b = _y[0]
    for n in range(len(a)):
        b -= a[n] * _y[n] * K[support_vec_idx[n], support_vec_idx[0]]

    return (solution, b, support_vec_idx)

def draw_datasetonly(X, y):
    filename = "dataset.png"

    ###These are to make the figure clearer. You don't need to change this part.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555'); plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')
    plt.scatter(X[:, 0], X[:, 1], c = ['#444C5C' if yy == 1 else '#78A5A3' for yy in y], edgecolor='none', s=30)

    plt.savefig(filename)
    #elice_utils.send_image(filename)

    plt.close()

def draw(X, y, sol, b, support_vec_idx):
    filename = "svm.png"

    ###These are to make the figure clearer. You don't need to change this part.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555'); plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')
    plt.scatter(X[:, 0], X[:, 1], c = ['#444C5C' if yy == 1 else '#78A5A3' for yy in y], edgecolor='none', s=30)

    for x1s in np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100):
        # Note that we cannot directly calculate W when using polynomical or RBF kernels. Thus, we look at all datapoints X and if g(X) is similar to 0, we include such datapoint into the decision boundary.
        for x2s in np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100):
            returnthis = b
            for n in range(100):
                returnthis += sol['x'][n] * y[n] * poly_kernel(X[n], np.array([x1s, x2s]))
            if abs(returnthis) < 0.1:
                plt.scatter(x1s, x2s, c = '#CE5A57', edgecolor='none', s = 30)

    plt.scatter(X[:, 0][support_vec_idx], X[:, 1][support_vec_idx], s=200, c='#CE5A57', edgecolor='none', marker='*')

    plt.savefig(filename)
    #elice_utils.send_image(filename)

    plt.close()

if __name__ == '__main__':
    X, y = generate_data(100)
    sol, b, support_vec_idx = svm(X, y)
    # Comment out the function below after you optimize all the parameters and find the support vectors!
    #draw_datasetonly(X, y)
    # Use the function below after you optimize all the parameters and find the support vectors!
    draw(X, y, sol, b, support_vec_idx)