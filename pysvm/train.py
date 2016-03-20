from subprocess import Popen, PIPE
import time
from svmutil import *
import os
import sys
import random

__author__ = 'Artanis'

svmScaleExe = r'.\bin\svm-scale.exe'


def scaleProblem(trainFilePath, scaledFileName, rangeFileName):
    print('Scaling training data...')

    cmd = '%s -s "%s" "%s" > "%s"' % (svmScaleExe, rangeFileName, trainFilePath, scaledFileName)
    Popen(cmd, shell=True, stdout=PIPE).communicate()


def readProblem(filePath):
    assert os.path.exists(filePath), "%s not found." % filePath

    Y = []
    X = []

    for line in file(filePath):
        spline = line.strip().split()
        Y.append(int(spline[0]))
        X.append([float(token.split(':')[1]) for token in spline[1:]])

    return Y, X


def buildModel(label, trainY, trainX, c, g):
    # build a binary classification model
    tmpY = []
    for y in trainY:
        tmpY.append(y == label and 1 or -1)

    problem = svm_problem(tmpY, trainX)
    param = svm_parameter('-c %f -g %f -b 1 -q' % (c, g))
    return svm_train(problem, param)


def genPerm(length):
    perm = [i for i in range(length)]

    for i in range(length):
        j = i + random.randint(0, length - i - 1)

        tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp

    return perm


def divideDataset(Y, X, perm, begin, end):
    length = len(Y)
    selectedY = []
    selectedX = []
    remainedY = []
    remainedX = []

    for j in range(length):
        idx = perm[j]

        if begin <= j < end:
            selectedY.append(Y[idx])
            selectedX.append(X[idx])
        else:
            remainedY.append(Y[idx])
            remainedX.append(X[idx])

    return selectedY, selectedX, remainedY, remainedX


def crossValidation(Y, X, fold, c, g):
    length = len(X)
    uniLabels = sorted(set(Y))

    perm = genPerm(length)
    foldStart = []

    for i in range(fold + 1):
        foldStart.append(i * length / fold)

    nCorrect = 0

    for i in range(fold):
        testY, testX, trainY, trainX = divideDataset(Y, X, perm, foldStart[i], foldStart[i + 1])

        models = []
        for label in uniLabels:
            models.append(buildModel(label, trainY, trainX, c, g))
            print '.',

        predictY = [-1] * len(testY)
        predictProb = [-1.0] * len(testY)

        for j in range(len(models)):
            results = svm_predict(testY, testX, models[j], '-b 1 -q')

            for k in range(len(testX)):
                if int(results[0][k] + 0.5) == 1:
                    prob = max(results[2][k])
                else:
                    prob = min(results[2][k])

                if prob > predictProb[k]:
                    predictProb[k] = prob
                    predictY[k] = uniLabels[j]

        for j in range(len(testX)):
            assert predictY[j] != -1
            if predictY[j] == testY[j]:
                nCorrect += 1

    return float(nCorrect) / length


def range_f(begin, end, step):
    # like range, but works on non-integer too
    seq = []
    while True:
        if step > 0 and begin > end:
            break
        if step < 0 and begin < end:
            break
        seq.append(begin)
        begin = begin + step
    return seq


if __name__ == "__main__":
    assert os.path.exists(svmScaleExe), "svm-scale executable not found"

    if len(sys.argv) >= 2:
        trainFilePath = sys.argv[1]
    else:
        trainFilePath = 'sub.txt'
    assert os.path.exists(trainFilePath), "training file not found"
    trainFileName = os.path.split(trainFilePath)[1]

    scaledFileName = trainFileName + ".scale"
    rangeFileName = trainFileName + ".range"

    scaleProblem(trainFilePath, scaledFileName, rangeFileName)
    labels, features = readProblem(scaledFileName)

    fold = 3
    c_begin, c_end, c_step = 1, 5, 2
    g_begin, g_end, g_step = 1, -12, -2

    c_seq = range_f(c_begin, c_end, c_step)
    g_seq = range_f(g_begin, g_end, g_step)

    fd = open("cross.txt", "w")
    begin = time.time()

    for c in c_seq:
        for g in g_seq:
            acc = crossValidation(labels, features, fold, 2**c, 2**g)
            fd.write(str(2**c) + ' ' + str(2**g) + ' ' + str(acc) + '\n')
            print 2**c, 2**g, acc

    print 'It costs ' + str(time.time() - begin) + ' sec.'
    fd.close()

