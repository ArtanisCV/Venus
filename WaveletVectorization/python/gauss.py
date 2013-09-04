from math import exp

__author__ = 'Artanis'

import numpy
from scipy import optimize


def normTwoSqr(p1, p2):
    return float((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def rasterization(gaussParas, imgParas):
    cx, cy, v = gaussParas
    rows, cols = imgParas

    img = numpy.ndarray(shape=(rows, cols), dtype=numpy.float64)

    for x in range(rows):
        for y in range(cols):
            d = normTwoSqr((x, y), (cx, cy))
            img[x][y] = exp(-d / v)

    return img


def dR_dcx(x, y, gaussParas):
    cx, cy, v = gaussParas

    d = normTwoSqr((x, y), (cx, cy))
    return exp(-d / v) * 2.0 * float(x - cx) / float(v)


def dR_dcy(x, y, gaussParas):
    cx, cy, v = gaussParas

    d = normTwoSqr((x, y), (cx, cy))
    return exp(-d / v) * 2.0 * float(y - cy) / float(v)


def dR_dv(x, y, gaussParas):
    cx, cy, v = gaussParas

    d = normTwoSqr((x, y), (cx, cy))
    return exp(-d / v) * d / float(v) ** 2


def grads(optGaussParas, imgParas, oriImg):
    optImg = rasterization(optGaussParas, imgParas)
    grads_sum = [0] * 3
    rows, cols = imgParas

    for x in range(rows):
        for y in range(cols):
            diff = 2.0 * (optImg[x][y] - oriImg[x][y])
            grads_sum[0] += dR_dcx(x, y, optGaussParas) * diff
            grads_sum[1] += dR_dcy(x, y, optGaussParas) * diff
            grads_sum[2] += dR_dv(x, y, optGaussParas) * diff

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
    optGaussParas = numpy.asarray((42, 23, 150))
    oriGaussParas = numpy.asarray((32, 32, 100))

    oriImg = rasterization(oriGaussParas, imgParas)
    # import cv2
    # cv2.imwrite("test.png", numpy.array(oriImg * 255.0, dtype=numpy.uint8))

    def f(gaussParas, printLike=True):
        optImg = rasterization(gaussParas, imgParas)
        result = like(oriImg, optImg)

        if printLike:
            print result
        return result

    def fPrime(gaussParas):
        return numpy.asarray(grads(gaussParas, imgParas, oriImg))

    print fPrime(optGaussParas)
    print grads(optGaussParas, imgParas, oriImg)
    print optimize.fmin_cg(f, optGaussParas)
    print optimize.fmin_cg(f, optGaussParas, fPrime)