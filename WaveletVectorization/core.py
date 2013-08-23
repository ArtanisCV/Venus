import math
import numpy
from scipy.integrate import quad
from scipy.misc import derivative

__author__ = 'Artanis'


def phi_1d(x):
    if 0 <= x < 1:
        return 1.0
    else:
        return 0.0


def psi_1d(x):
    if 0 <= x < 0.5:
        return 1.0
    elif 0.5 <= x < 1:
        return -1.0
    else:
        return 0.0


phi_1d_jk = lambda x, j, k: phi_1d(2 ** j * x - k)
psi_1d_jk = lambda x, j, k: psi_1d(2 ** j * x - k)


def psi_2d_jke(x, y, j, kx, ky, e):
    if e == 0:
        return 2 ** j * phi_1d_jk(x, j, kx) * phi_1d_jk(y, j, ky)
    elif e == 1:
        return 2 ** j * phi_1d_jk(x, j, kx) * psi_1d_jk(y, j, ky)
    elif e == 2:
        return 2 ** j * psi_1d_jk(x, j, kx) * phi_1d_jk(y, j, ky)
    else:
        return 2 ** j * psi_1d_jk(x, j, kx) * psi_1d_jk(y, j, ky)


def cphi_1d(x):
    if x < 0:
        return 0.0
    elif 0 <= x < 1:
        return float(x)
    else:
        return 1.0


def cpsi_1d(x):
    if x < 0:
        return 0.0
    elif 0 <= x < 0.5:
        return float(x)
    elif 0.5 <= x < 1:
        return 1 - float(x)
    else:
        return 0.0


cphi_1d_jk = lambda x, j, k: 2 ** (-j) * cphi_1d(2 ** j * x - k)
cpsi_1d_jk = lambda x, j, k: 2 ** (-j) * cpsi_1d(2 ** j * x - k)


def dF_dp(i, t):
    if i == 0:
        return (1.0 - t) ** 3
    elif i == 1:
        return 3.0 * (1.0 - t) ** 2 * t
    elif i == 2:
        return 3.0 * (1.0 - t) * t ** 2
    else:
        return t ** 3


def dFp_dp(i, t):
    if i == 0:
        return -3.0 * (1.0 - t) ** 2
    elif i == 1:
        return 3.0 * (-2.0 * (1.0 - t) * t + (1.0 - t) ** 2)
    elif i == 2:
        return 3.0 * (-t ** 2 + (1.0 - t) * 2 * t)
    else:
        return 3.0 * t ** 2


F = lambda t, para: dF_dp(0, t) * para[0] + dF_dp(1, t) * para[1] + dF_dp(2, t) * para[2] + dF_dp(3, t) * para[3]
Fp = lambda t, para: dFp_dp(0, t) * para[0] + dFp_dp(1, t) * para[1] + dFp_dp(2, t) * para[2] + dFp_dp(3, t) * para[3]


MAX_E = 3


class RasterizeCubicBezier:
    def __init__(self, xPara, yPara, imgShape):
        self.xPara = xPara
        self.yPara = yPara

        self.rows = imgShape[0]
        self.cols = imgShape[1]

        self.MAX_J = int(math.ceil(math.log(min(self.rows, self.cols), 2))) - 1
        self.MAX_K = 2 ** self.MAX_J - 1

        self.X = lambda t: F(t, xPara)
        self.Y = lambda t: F(t, yPara)
        self.Xp = lambda t: Fp(t, xPara)
        self.Yp = lambda t: Fp(t, yPara)

        self.c_mat = numpy.zeros(shape=(self.MAX_J + 1, self.MAX_K + 1, self.MAX_K + 1, MAX_E + 1),
                                 dtype=numpy.float64)
        for j in range(self.MAX_J + 1):
            for kx in range(2 ** j):
                for ky in range(2 ** j):
                    for e in range(MAX_E + 1):
                        self.c_mat[j][kx][ky][e] = self.c_jk(j, kx, ky, e)

    def inner_c00_jk(self, t):
        return cphi_1d(self.X(t)) * phi_1d(self.Y(t)) * self.Yp(t)

    def inner_c01_jk(self, t, j, kx, ky):
        return -(2 ** j) * cpsi_1d_jk(self.Y(t), j, ky) * phi_1d_jk(self.X(t), j, kx) * self.Xp(t)

    def inner_c10_jk(self, t, j, kx, ky):
        return 2 ** j * cpsi_1d_jk(self.X(t), j, kx) * phi_1d_jk(self.Y(t), j, ky) * self.Yp(t)

    def inner_c11_jk(self, t, j, kx, ky):
        return 2 ** j * cpsi_1d_jk(self.X(t), j, kx) * psi_1d_jk(self.Y(t), j, ky) * self.Yp(t)

    def c_jk(self, j, kx, ky, e):
        if e == 0:
            return quad(self.inner_c00_jk, 0, 1)[0]
        elif e == 1:
            return quad(lambda t: self.inner_c01_jk(t, j, kx, ky), 0, 1)[0]
        elif e == 2:
            return quad(lambda t: self.inner_c10_jk(t, j, kx, ky), 0, 1)[0]
        else:
            return quad(lambda t: self.inner_c11_jk(t, j, kx, ky), 0, 1)[0]

    def R(self, x, y):
        result = self.c_mat[0][0][0][0] * psi_2d_jke(x, y, 0, 0, 0, 0)

        for j in range(self.MAX_J + 1):
            for kx in range(2 ** j):
                for ky in range(2 ** j):
                    for e in range(1, MAX_E + 1):
                        result += self.c_mat[j][kx][ky][e] * psi_2d_jke(x, y, j, kx, ky, e)

        return result

    def getImg(self):
        img = numpy.zeros(shape=(self.rows, self.cols), dtype=numpy.float64)

        for x in range(self.rows):
            for y in range(self.cols):
                img[x][y] += self.R(x / float(self.rows), y / float(self.cols))

        return img


class Rasterization:
    def __init__(self, contour, imgShape):
        self.contour = contour
        self.rows = imgShape[0]
        self.cols = imgShape[1]

    def getImg(self):
        img = numpy.zeros(shape=(self.rows, self.cols), dtype=numpy.float64)

        for i in range(0, len(self.contour), 6):
            xPara = [self.contour[(i + j * 2) % len(self.contour)] / float(self.rows) for j in range(4)]
            yPara = [self.contour[(i + 1 + j * 2) % len(self.contour)] / float(self.rows) for j in range(4)]

            img += RasterizeCubicBezier(xPara, yPara, (self.rows, self.cols)).getImg()

        minimum = numpy.min(img)
        maximum = numpy.max(img)
        normalizedImg = (img - minimum) / float(maximum - minimum + 1e-8)

        return img, normalizedImg


class VectorizeCubicBezier:
    def __init__(self, xPara, yPara, diff):
        self.xPara = xPara
        self.yPara = yPara

        self.rows = diff.shape[0]
        self.cols = diff.shape[1]
        self.diff = diff

        self.MAX_J = int(math.ceil(math.log(min(self.rows, self.cols), 2))) - 1
        self.MAX_K = 2 ** self.MAX_J - 1

        self.X = lambda t: F(t, xPara)
        self.Y = lambda t: F(t, yPara)
        self.Xp = lambda t: Fp(t, xPara)
        self.Yp = lambda t: Fp(t, yPara)

    def inner_dc00_dx(self, i, t):
        return phi_1d(self.X(t)) * phi_1d(self.Y(t)) * dF_dp(i, t) * self.Yp(t)

    def inner_dc01_dx(self, i, t, j, kx, ky):
        return -(2 ** j) * cpsi_1d_jk(self.Y(t), j, ky) * dFp_dp(i, t) * phi_1d_jk(self.X(t), j, kx)

    def inner_dc10_dx(self, i, t, j, kx, ky):
        return 2 ** j * psi_1d_jk(self.X(t), j, kx) * dF_dp(i, t) * phi_1d_jk(self.Y(t), j, ky) * self.Yp(t)

    def inner_dc11_dx(self, i, t, j, kx, ky):
        return 2 ** j * psi_1d_jk(self.X(t), j, kx) * dF_dp(i, t) * psi_1d_jk(self.Y(t), j, ky) * self.Yp(t)

    def dc_dx(self, i, j, kx, ky, e):
        if e == 0:
            return quad(lambda t: self.inner_dc00_dx(i, t), 0, 1)[0]
        elif e == 1:
            return quad(lambda t: self.inner_dc01_dx(i, t, j, kx, ky), 0, 1)[0]
        elif e == 2:
            return quad(lambda t: self.inner_dc10_dx(i, t, j, kx, ky), 0, 1)[0]
        else:
            return quad(lambda t: self.inner_dc11_dx(i, t, j, kx, ky), 0, 1)[0]

    def inner_dc00_dy(self, i, t):
        return cphi_1d(self.X(t)) * phi_1d(self.Y(t)) * dFp_dp(i, t)

    def inner_dc01_dy(self, i, t, j, kx, ky):
        return -(2 ** j) * psi_1d_jk(self.Y(t), j, ky) * dF_dp(i, t) * phi_1d_jk(self.X(t), j, kx) * self.Xp(t)

    def inner_dc10_dy(self, i, t, j, kx, ky):
        return 2 ** j * cpsi_1d_jk(self.X(t), j, kx) * dFp_dp(i, t) * phi_1d_jk(self.Y(t), j, ky)

    def inner_dc11_dy(self, i, t, j, kx, ky):
        return 2 ** j * cpsi_1d_jk(self.X(t), j, kx) * dFp_dp(i, t) * psi_1d_jk(self.Y(t), j, ky)

    def dc_dy(self, i, j, kx, ky, e):
        if e == 0:
            return quad(lambda t: self.inner_dc00_dy(i, t), 0, 1)[0]
        elif e == 1:
            return quad(lambda t: self.inner_dc01_dy(i, t, j, kx, ky), 0, 1)[0]
        elif e == 2:
            return quad(lambda t: self.inner_dc10_dy(i, t, j, kx, ky), 0, 1)[0]
        else:
            return quad(lambda t: self.inner_dc11_dy(i, t, j, kx, ky), 0, 1)[0]

    def getGrads(self):
        grads = [0.0] * 8
        grads_core = [0.0] * 8

        for i in range(4):
            dc_dxi_mat = numpy.zeros(shape=(self.MAX_J + 1, self.MAX_K + 1, self.MAX_K + 1, MAX_E + 1),
                                     dtype=numpy.float64)
            for j in range(self.MAX_J + 1):
                for kx in range(2 ** j):
                    for ky in range(2 ** j):
                        for e in range(MAX_E + 1):
                            dc_dxi_mat[j][kx][ky][e] = self.dc_dx(i, j, kx, ky, e)

            for x in range(self.rows):
                for y in range(self.cols):
                    p = (x / float(self.rows), y / float(self.cols))
                    g_xi = dc_dxi_mat[0][0][0][0] * psi_2d_jke(p[0], p[1], 0, 0, 0, 0)

                    for j in range(self.MAX_J + 1):
                        for kx in range(2 ** j):
                            for ky in range(2 ** j):
                                for e in range(1, MAX_E + 1):
                                    g_xi += dc_dxi_mat[j][kx][ky][e] * psi_2d_jke(p[0], p[1], j, kx, ky, e)

                    grads[i * 2] += g_xi * self.diff[x][y] * 2.0
                    grads_core[i * 2] += g_xi

            dc_dyi_mat = numpy.zeros(shape=(self.MAX_J + 1, self.MAX_K + 1, self.MAX_K + 1, MAX_E + 1),
                                     dtype=numpy.float64)
            for j in range(self.MAX_J + 1):
                for kx in range(2 ** j):
                    for ky in range(2 ** j):
                        for e in range(MAX_E + 1):
                            dc_dyi_mat[j][kx][ky][e] = self.dc_dy(i, j, kx, ky, e)

            for x in range(self.rows):
                for y in range(self.cols):
                    p = (x / float(self.rows), y / float(self.cols))
                    g_yi = dc_dyi_mat[0][0][0][0] * psi_2d_jke(p[0], p[1], 0, 0, 0, 0)

                    for j in range(self.MAX_J + 1):
                        for kx in range(2 ** j):
                            for ky in range(2 ** j):
                                for e in range(1, MAX_E + 1):
                                    g_yi += dc_dyi_mat[j][kx][ky][e] * psi_2d_jke(p[0], p[1], j, kx, ky, e)

                    grads[i * 2 + 1] += g_yi * self.diff[x][y] * 2.0
                    grads_core[i * 2 + 1] += g_yi

        return grads, grads_core


class VectorizeTest:
    def __init__(self, xPara, yPara, diff):
        self.xPara = xPara
        self.yPara = yPara

        self.rows = diff.shape[0]
        self.cols = diff.shape[1]
        self.diff = diff

        self.MAX_J = int(math.ceil(math.log(min(self.rows, self.cols), 2))) - 1
        self.MAX_K = 2 ** self.MAX_J - 1

        self.Y = lambda t: F(t, yPara)
        self.Yp = lambda t: Fp(t, yPara)

    def Xi(self, i, xi, t):
        result = 0

        for k in range(4):
            if k != i:
                result += dF_dp(k, t) * self.xPara[k]
            else:
                result += dF_dp(k, t) * xi

        return result

    def Xpi(self, i, xi, t):
        result = 0

        for k in range(4):
            if k != i:
                result += dFp_dp(k, t) * self.xPara[k]
            else:
                result += dFp_dp(k, t) * xi

        return result

    def inner_c00_jk_x(self, i, xi, t):
        return cphi_1d(self.Xi(i, xi, t)) * phi_1d(self.Y(t)) * self.Yp(t)

    def inner_c01_jk_x(self, i, xi, t, j, kx, ky):
        return -(2 ** j) * cpsi_1d_jk(self.Y(t), j, ky) * phi_1d_jk(self.Xi(i, xi, t), j, kx) * self.Xpi(i, xi, t)

    def inner_c10_jk_x(self, i, xi, t, j, kx, ky):
        return 2 ** j * cpsi_1d_jk(self.Xi(i, xi, t), j, kx) * phi_1d_jk(self.Y(t), j, ky) * self.Yp(t)

    def inner_c11_jk_x(self, i, xi, t, j, kx, ky):
        return 2 ** j * cpsi_1d_jk(self.Xi(i, xi, t), j, kx) * psi_1d_jk(self.Y(t), j, ky) * self.Yp(t)

    def inner_dc00_dx(self, i, xi):
        return quad(lambda t: self.inner_c00_jk_x(i, xi, t), 0, 1)[0]

    def inner_dc01_dx(self, i, xi, j, kx, ky):
        return quad(lambda t: self.inner_c01_jk_x(i, xi, t, j, kx, ky), 0, 1)[0]

    def inner_dc10_dx(self, i, xi, j, kx, ky):
        return quad(lambda t: self.inner_c10_jk_x(i, xi, t, j, kx, ky), 0, 1)[0]

    def inner_dc11_dx(self, i, xi, j, kx, ky):
        return quad(lambda t: self.inner_c11_jk_x(i, xi, t, j, kx, ky), 0, 1)[0]

    def dc_dx(self, i, j, kx, ky, e):
        if e == 0:
            return derivative(lambda xi: self.inner_dc00_dx(i, xi), xPara[i], dx=1e-4)
        elif e == 1:
            return derivative(lambda xi: self.inner_dc01_dx(i, xi, j, kx, ky), xPara[i], dx=1e-4)
        elif e == 2:
            return derivative(lambda xi: self.inner_dc10_dx(i, xi, j, kx, ky), xPara[i], dx=1e-4)
        else:
            return derivative(lambda xi: self.inner_dc11_dx(i, xi, j, kx, ky), xPara[i], dx=1e-4)

    def getGrads(self):
        grads = [0.0] * 8
        grads_core = [0.0] * 8

        for i in range(4):
            dc_dxi_mat = numpy.zeros(shape=(self.MAX_J + 1, self.MAX_K + 1, self.MAX_K + 1, MAX_E + 1),
                                     dtype=numpy.float64)
            for j in range(self.MAX_J + 1):
                for kx in range(2 ** j):
                    for ky in range(2 ** j):
                        for e in range(MAX_E + 1):
                            dc_dxi_mat[j][kx][ky][e] = self.dc_dx(i, j, kx, ky, e)

            for x in range(self.rows):
                for y in range(self.cols):
                    p = (x / float(self.rows), y / float(self.cols))
                    g_xi = dc_dxi_mat[0][0][0][0] * psi_2d_jke(p[0], p[1], 0, 0, 0, 0)

                    for j in range(self.MAX_J + 1):
                        for kx in range(2 ** j):
                            for ky in range(2 ** j):
                                for e in range(1, MAX_E + 1):
                                    g_xi += dc_dxi_mat[j][kx][ky][e] * psi_2d_jke(p[0], p[1], j, kx, ky, e)

                    grads[i * 2] += g_xi * self.diff[x][y] * 2.0
                    grads_core[i * 2] += g_xi

        return grads, grads_core


class Vectorization:
    def __init__(self, contour, oriImg):
        self.contour = contour
        self.rows = oriImg.shape[0]
        self.cols = oriImg.shape[1]

        self.diff = Rasterization(self.contour, (self.rows, self.cols)).getImg()[0] - oriImg

    def getGrads(self):
        t_grads = [0.0] * len(self.contour)
        t_grads_core = [0.0] * len(self.contour)

        for i in range(0, len(self.contour), 6):
            xPara = [self.contour[(i + j * 2) % len(self.contour)] / float(self.rows) for j in range(4)]
            yPara = [self.contour[(i + 1 + j * 2) % len(self.contour)] / float(self.rows) for j in range(4)]

            grads, grads_core = VectorizeTest(xPara, yPara, self.diff).getGrads()
            for j in range(len(grads)):
                t_grads[(i + j) % len(self.contour)] += grads[j]
                t_grads_core[(i + j) % len(self.contour)] += grads_core[j]

        return t_grads, t_grads_core


if __name__ == "__main__":
    oriPath = [1, 1, 3, 1, 7, 3, 7, 7, 3, 7, 1, 3]
    optPath = [1, 1, 4, 1, 7, 3, 7, 7, 3, 7, 1, 3]

    import numpy
    numpy.warnings.simplefilter("ignore", Warning)

    imgShape = (8, 8)
    # img, normalizedImg = Rasterization(oriPath, imgShape).getImg()
    #
    # file = open("test.txt", "w")
    # for i in range(8):
    #     for j in range(8):
    #         file.write(str(img[i][j]) + " ")
    #     file.write("\n")
    # file.close()
    #
    # import cv2
    # cv2.imwrite("test.png", numpy.array(normalizedImg * 255.0, dtype=numpy.uint8))

    oriRaster = Rasterization(oriPath, imgShape).getImg()[0]
    optRaster = Rasterization(optPath, imgShape).getImg()[0]

    xPara = [item / float(imgShape[0]) for item in [1, 4, 7, 7]]
    yPara = [item / float(imgShape[1]) for item in [1, 1, 3, 7]]

    diff = numpy.ndarray(shape=imgShape, dtype=numpy.float64)
    fd = open("raster_diff.txt", "r")
    for i in range(imgShape[0]):
        tokens = fd.readline().strip().split()
        for j in range(imgShape[1]):
            diff[i][j] = float(tokens[j])
    fd.close()

    grads, grads_core = VectorizeCubicBezier(xPara, yPara, diff).getGrads()
    print grads_core
    print grads
    print

    grads, grads_core = VectorizeTest(xPara, yPara, diff).getGrads()
    print grads_core
    print grads
    print

    grads, grads_core = Vectorization(optPath, oriRaster).getGrads()
    print grads