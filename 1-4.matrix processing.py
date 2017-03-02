import numpy


def main():
    A = get_matrix()
    print(matrix_tutorial(A))


def get_matrix():
    mat = []
    [n, m] = [int(x) for x in input().strip().split(" ")]
    for i in range(n):
        row = [int(x) for x in input().strip().split(" ")]
        mat.append(row)
    return numpy.array(mat)


def matrix_tutorial(A):
    # 2
    B = A.transpose()
    # 3
    try:
        C = numpy.linalg.inv(B)
        # 4
        return C[C > 0].size
    except:
        return "not invertible"


if __name__ == "__main__":
    main()

"""
3 5
1 2 6 3 8
11 0 -1 3 1
9 0 7 -3 4

2 2
1 1
1 2
"""