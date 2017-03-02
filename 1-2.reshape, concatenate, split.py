import numpy as np


def main():
    print(matrix_tutorial())


def matrix_tutorial():
    A = np.array([[1, 4, 5, 8], [2, 1, 7, 3], [5, 4, 5, 9]])
    # 1
    B = A.reshape((6, 2))
    # 2
    tmp = np.array([[2, 2],
                    [5, 3]])
    B = np.concatenate((B, tmp))
    # 3
    tmp = np.split(B, 2, axis=0)
    C = tmp[0]
    D = tmp[1]
    # 4
    E = np.concatenate((C, D), axis=1)
    # 5
    return E


if __name__ == "__main__":
    main()
