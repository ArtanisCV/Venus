from subprocess import Popen, PIPE
from svmutil import *
import os
import random

__author__ = 'Artanis'

svmScaleExe = r'.\bin\svm-scale.exe'
svmTrainExe = r'.\bin\svm-train.exe'
tmpTrainFile = "tmp_train"


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


# Give me a label
def buildProblem(label, Y, X, fileName):
    # build a binary classification problem
    fd = open(fileName, "w")

    for i in range(len(Y)):
        if label == Y[i]:
            fd.write("+1")
        else:
            fd.write("-1")
        for j in range(len(X[i])):
            fd.write(" " + str(j + 1) + ":" + str(X[i][j]))
        fd.write("\n")

    fd.close()


def trainProblem(c, g, trainFileName):
    modelFileName = "tmp_model"

    #Y, X = readProblem(trainFileName)
    #problem = svm_problem(Y, X)
    #param = svm_parameter('-c %f -g %f -b 1 -q' % (c, g))
    #return svm_train(problem, param)

    cmd = "%s -c %f -g %f -b 1 -q %s %s" % (svmTrainExe, c, g, trainFileName, modelFileName)
    Popen(cmd, shell=True, stdout=PIPE).communicate()

    model = svm_load_model(modelFileName)
    os.remove(modelFileName)
    return model


def crossValidation(Y, X, fold, c, g):
    length = len(X)
    uniLabels = sorted(set(Y))
    perm = [i for i in range(length)]

    for i in range(length):
        j = i + random.randint(0, length - i - 1)

        tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp

    foldStart = []

    for i in range(fold + 1):
        foldStart.append(i * length / fold)

    nCorrect = 0

    for i in range(fold):
        begin = foldStart[i]
        end = foldStart[i + 1]

        trainY = []
        trainX = []
        testY = []
        testX = []

        for j in range(length):
            idx = perm[j]

            if begin <= j < end:
                testY.append(Y[idx])
                testX.append(X[idx])
            else:
                trainY.append(Y[idx])
                trainX.append(X[idx])

        models = []

        for label in uniLabels:
            tmpY = []
            for y in trainY:
                tmpY.append(y == label and 1 or -1)

            problem = svm_problem(tmpY, trainX)
            param = svm_parameter('-c %f -g %f -b 1 -q' % (c, g))
            models.append(svm_train(problem, param))

            #buildProblem(label, trainY, trainX, tmpTrainFile)
            #models.append(trainProblem(c, g, tmpTrainFile))

        predictYs = [-1] * len(testY)
        predictProb = [-1.0] * len(testY)

        for j in range(len(models)):
            results = svm_predict(testY, testX, models[j], '-b 1')

            for k in range(len(testX)):
                if int(results[0][k] + 0.5) == 1:
                    prob = max(results[2][k])
                else:
                    prob = min(results[2][k])

                if prob > predictProb[k]:
                    predictProb[k] = prob
                    predictYs[k] = uniLabels[j]

        for j in range(len(testX)):
            assert predictYs[j] != -1
            if predictYs[j] == testY[j]:
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
    assert os.path.exists(svmTrainExe), "svm-train executable not found"

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

    for c in c_seq:
        for g in g_seq:
            acc = crossValidation(labels, features, fold, 2**c, 2**g)
            fd.write(str(2**c) + ' ' + str(2**g) + ' ' + str(acc) + '\n')
            print 2**c, 2**g, acc

    fd.close()

    #trainFile = 'sub.txt.scale'
    #labels, features = readProblem(trainFile)
    #length = len(features)
    #
    #perm = [i for i in range(length)]
    #for i in range(length):
    #    j = i + random.randint(0, length - i - 1)
    #
    #    tmp = perm[i]
    #    perm[i] = perm[j]
    #    perm[j] = tmp
    #
    #testLabels = []
    #testFeatures = []
    #trainLabels = []
    #trainFeatures = []
    #
    #for i in range(length):
    #    labels[i] = labels[i] == 1 and 1 or -1

