import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

# Functions linear_func1, linear_func2, and generate_data is used to generate random datapoints. You do not need to change this function
def linear_func1(x):
    l = len(x)
    return (3 * x + 100 + 30 * np.random.randn(l))


def linear_func2(x):
    l = len(x)
    return (3 * x - 100 + 30 * np.random.randn(l))


def generate_data(n):
    np.random.seed(32840091)

    x1_1 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_1 = linear_func1(x1_1)
    x1_2 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_2 = linear_func2(x1_2)
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
            K[i, j] = np.dot(X[i], X[j])

    P = matrix(np.outer(y, y) * K, tc='d')
    q = matrix(np.ones(n)*-1, tc='d')

    G = matrix(np.diag(np.ones(n)*-1), tc='d')
    h = matrix(np.zeros(n), tc='d')

    A = matrix(y, (1, n), tc='d')
    b = matrix(0.0, tc='d')

    opts = {'show_progress': False}
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

    # W = (w1, w2)
    W = np.zeros(X.shape[1])
    for n in range(len(a)):
        W += a[n] * _y[n] * _x[n]

    # b using any SV
    b = y[0] - np.dot(W.T, _x[0])

    return (W, b, support_vec_idx)


def draw_datasetonly(X, y):
    filename = "dataset.png"

    ###These are to make the figure clearer. You don't need to change this part.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555');
    plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')
    plt.scatter(X[:, 0], X[:, 1], c=['#444C5C' if yy == 1 else '#78A5A3' for yy in y], edgecolor='none', s=30)

    plt.savefig(filename)
    # elice_utils.send_image(filename)

    plt.close()


def draw(X, y, W, b, support_vec_idx):
    filename = "svm.png"

    ###These are to make the figure clearer. You don't need to change this part.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555');
    plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')
    plt.scatter(X[:, 0], X[:, 1], c=['#444C5C' if yy == 1 else '#78A5A3' for yy in y], edgecolor='none', s=30)

    ls = np.linspace(-50, 50, 100)
    plt.plot(ls, [-1 * (W[0] * v + b) / W[1] for v in ls], lw=1.5, color='#CE5A57')
    plt.scatter(X[:, 0][support_vec_idx], X[:, 1][support_vec_idx], s=200, c='#CE5A57', edgecolor='none', marker='*')

    plt.savefig(filename)
    # elice_utils.send_image(filename)

    plt.close()


if __name__ == '__main__':
    X, y = generate_data(100)
    W, b, support_vec_idx = svm(X, y)
    # Comment out the function below after you optimize all the parameters and find the support vectors!
    # draw_datasetonly(X, y)
    # Use the function below after you optimize all the parameters and find the support vectors!
    draw(X, y, W, b, support_vec_idx)