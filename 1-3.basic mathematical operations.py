import numpy


def main():
    print(matrix_tutorial())


def matrix_tutorial():
    A = numpy.array([[1, 4, 5, 8], [2, 1, 7, 3], [5, 4, 5, 9]])

    B = A.reshape((6, 2))
    B = numpy.concatenate((B, numpy.array([[2, 2], [5, 3]])), axis=0)
    (C, D) = numpy.split(B, 2, axis=0)
    E = numpy.concatenate((C, D), axis=1)

    _sum = numpy.sum(E)
    tmp = E / _sum
    # print (tmp)
    return numpy.var(tmp)


if __name__ == "__main__":
    main()
