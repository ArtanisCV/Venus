from math import exp

__author__ = 'Artanis'

import numpy
from scipy import optimize


def normTwoSqr(p1, p2):
    return float((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def rasterization(gaussMixParas, imgParas):
    rows, cols = imgParas
    img = numpy.zeros(shape=(rows, cols), dtype=numpy.float64)

    nGauss = gaussMixParas.shape[0] / 3

    for x in range(rows):
        for y in range(cols):
            for i in range(nGauss):
                x0, y0, v0 = gaussMixParas[3 * i: 3 * i + 3]

                d = normTwoSqr((x, y), (x0, y0))
                img[x][y] += exp(-d / v0) / nGauss

    return img


def dR_dcx(i, x, y, gaussMixParas):
    nGauss = gaussMixParas.shape[0] / 3
    cx, cy, v = gaussMixParas[3 * i: 3 * i + 3]

    d = normTwoSqr((x, y), (cx, cy))
    return exp(-d / v) * 2.0 * float(x - cx) / float(v) / nGauss


def dR_dcy(i, x, y, gaussMixParas):
    nGauss = gaussMixParas.shape[0] / 3
    cx, cy, v = gaussMixParas[3 * i: 3 * i + 3]

    d = normTwoSqr((x, y), (cx, cy))
    return exp(-d / v) * 2.0 * float(y - cy) / float(v) / nGauss


def dR_dv(i, x, y, gaussMixParas):
    nGauss = gaussMixParas.shape[0] / 3
    cx, cy, v = gaussMixParas[3 * i: 3 * i + 3]

    d = normTwoSqr((x, y), (cx, cy))
    return exp(-d / v) * d / float(v) ** 2 / nGauss


def grads(optGaussMixParas, imgParas, oriImg):
    optImg = rasterization(optGaussMixParas, imgParas)
    grads_sum = [0] * optGaussMixParas.shape[0]
    nGauss = optGaussMixParas.shape[0] / 3
    rows, cols = imgParas

    for x in range(rows):
        for y in range(cols):
            diff = 2.0 * (optImg[x][y] - oriImg[x][y])
            for i in range(nGauss):
                grads_sum[3 * i] += dR_dcx(i, x, y, optGaussMixParas) * diff
                grads_sum[3 * i + 1] += dR_dcy(i, x, y, optGaussMixParas) * diff
                grads_sum[3 * i + 2] += dR_dv(i, x, y, optGaussMixParas) * diff

    return grads_sum


def like(img1, img2):
    rows, cols = img1.shape
    like_sum = 0

    for i in range(rows):
        for j in range(cols):
            like_sum += (img1[i][j] - img2[i][j]) ** 2

    return like_sum


if __name__ == "__main__":
    imgParas = (64, 64)
    optGaussMixParas = numpy.asarray((42, 23, 150, 30, 50, 100))
    oriGaussMixParas = numpy.asarray((20, 20, 100, 40, 40, 225))

    oriImg = rasterization(oriGaussMixParas, imgParas)
    # import cv2
    # cv2.imwrite("test.png", numpy.array(oriImg * 255.0, dtype=numpy.uint8))

    def f(gaussMixParas, printLike=True):
        optImg = rasterization(gaussMixParas, imgParas)
        result = like(oriImg, optImg)

        if printLike:
            print result
        return result

    def fPrime(gaussMixParas):
        return numpy.asarray(grads(gaussMixParas, imgParas, oriImg))

    print fPrime(optGaussMixParas)
    print grads(optGaussMixParas, imgParas, oriImg)
    print optimize.fmin_cg(f, optGaussMixParas)
    print optimize.fmin_cg(f, optGaussMixParas, fPrime)