from random import randint
from svm import *
from svmutil import *

__author__ = 'Artanis'


if __name__ == '__main__':
    # Create 200 random points
    X = [[randint(-20, 20), randint(-20, 20)] for i in range(20000)]

    # Classify them as 1 if they are in the circle and 0 if not
    Y = [(x1**2 + x2**2) < 144 and 1 or 0 for (x1, x2) in X]

    problem = svm_problem(Y, X)
    param = svm_parameter('-t 2 -b 1')
    model = svm_train(problem, param)
    print svm_predict([1, 0, 0], [[2, 2], [14, 13], [-18, 0]], model, '-b 1')