import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import r2_score


def main():
    data = pd.read_csv("./Advertising.csv", header=0)
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    for independent_var in range(1, 4):
        try:
            x_vec = data.ix[:, independent_var].as_matrix().reshape(-1, 1)
            x_label = data.columns[independent_var]

            one_var_advertising(x_vec, y_vec, x_label, y_label)
        except ValueError:
            pass

    '''
    f = open("./001.txt", 'r')

    x_vec = []
    x_label = 'x'
    y_vec = []
    y_label = 'y'

    while True:
        line = f.readline()
        if not line: break
        x = []
        x.append(int(line.split(',')[0]))
        y = []
        y.append(int(line.split(',')[1][:1]))
        x_vec.append(x)
        y_vec.append(y)

    x_vec = np.array(x_vec)
    y_vec = np.array(y_vec)

    one_var_advertising(x_vec, y_vec, x_label, y_label)
    # print(x_vec)
    # print(y_vec)
    '''

def one_var_advertising(x_vec, y_vec, x_label, y_label):
    filename = "advertising_fig_simple_{}.png".format(x_label)

    regr = linear_model.LinearRegression()

    regr.fit(x_vec, y_vec)
    predicted_y = regr.predict(x_vec)
    print(filename)
    print("Independent variable: {}".format(x_label))
    print("Coefficients: {}".format(regr.coef_))
    print("Intercept: {}".format(regr.intercept_))
    print("RSS: {}".format(np.sum((predicted_y - y_vec) ** 2)))
    print("R^2: {}".format(r2_score(y_vec, predicted_y)))
    print()

    plt.scatter(x_vec, y_vec, color='black')
    plt.plot(x_vec, predicted_y, color='blue', linewidth=3)

    plt.xlim((0, int(max(x_vec) * 1.1)))
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    plt.savefig(filename)
    # elice_utils.send_image(filename)

    plt.close()


if __name__ == "__main__":
    main()
