"""
Linear Regression with Regularization - Cross Validation
"""
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold


def main(rs=108):
    data = pd.read_csv("./data/Hitters.csv", header=0)
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    x_label = ", ".join(data.columns[1:-1])
    x_mat = data.ix[:, 1:-1].as_matrix()
    x_mat = x_mat.reshape(-1, x_mat.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(x_mat, y_vec, test_size=0.2, random_state=rs)

    # Linear Regression
    rss, r2, mse = multi_var_hitter(x_train, x_test, y_train, y_test, x_label)
    print("Linear Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print()

    # Ridge Regression
    best_lambda_ridge, best_lambda_lasso = get_best_lambda_value_ridge_lasso(data)
    rss, r2, mse = multi_var_hitter_ridge(x_train, x_test, y_train, y_test, x_label, best_lambda_ridge)
    print("Ridge Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_ridge))
    print()

    # lasso
    rss, r2, mse = multi_var_hitter_lasso(x_train, x_test, y_train, y_test, x_label, best_lambda_lasso)
    print("lasso Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_lasso))
    print()


def get_best_lambda_value_ridge_lasso(data):
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    x_label = ", ".join(data.columns[1:-1])
    x_mat = data.ix[:, 1:-1].as_matrix()
    x_mat = x_mat.reshape(-1, x_mat.shape[1])

    rid_alpha_space = np.linspace(0, 100000, 50)
    rid_score_list = []

    for alpha in rid_alpha_space:
        r2_list = []
        kf = KFold(n_splits=10)
        for train_idx, test_idx in kf.split(x_mat):
            x_train, x_test = x_mat[train_idx], x_mat[test_idx]
            y_train, y_test = y_vec[train_idx], y_vec[test_idx]
            rid = linear_model.Ridge(alpha=alpha)
            rid.fit(x_train, y_train)
            predicted_y_test = rid.predict(x_test)
            r2 = r2_score(y_test, predicted_y_test)
            r2_list.append(r2)
        rid_score_list.append(np.array(r2_list).sum()/10.0)

    las_alpha_space = np.linspace(0.1, 1000, 50)
    las_score_list = []

    for alpha in las_alpha_space:
        r2_list = []
        kf = KFold(n_splits=10)
        for train_idx, test_idx in kf.split(x_mat):
            x_train, x_test = x_mat[train_idx], x_mat[test_idx]
            y_train, y_test = y_vec[train_idx], y_vec[test_idx]
            las = linear_model.Lasso(alpha=alpha, max_iter=100000)
            las.fit(x_train, y_train)
            predicted_y_test = las.predict(x_test)
            r2 = r2_score(y_test, predicted_y_test)
            r2_list.append(r2)
        las_score_list.append(np.array(r2_list).sum() / 10.0)


    best_lambda_ridge = rid_alpha_space[np.array(rid_score_list).argmax()]
    best_lambda_lasso = las_alpha_space[np.array(las_score_list).argmax()]

    return best_lambda_ridge, best_lambda_lasso


def multi_var_hitter(x_train, x_test, y_train, y_test, x_label):
    regr = linear_model.LinearRegression()

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)
    return rss, r2, mse


def multi_var_hitter_ridge(x_train, x_test, y_train, y_test, x_label, best_lambda):
    regr = linear_model.Ridge(alpha=best_lambda)

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)

    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)

    return rss, r2, mse


def multi_var_hitter_lasso(x_train, x_test, y_train, y_test, x_label, best_lambda):
    regr = linear_model.Lasso(alpha=best_lambda)

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)

    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)

    return rss, r2, mse


if __name__ == "__main__":
    main()
