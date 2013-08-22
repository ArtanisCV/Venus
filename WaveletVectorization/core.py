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


def psi_jke(x, y, j, kx, ky, e):
    if e == 0:
        return 2 ** j * phi_1d_jk(x, j, kx) * phi_1d_jk(y, j, ky)
    elif e == 1:
        return 2 ** j * phi_1d_jk(x, j, kx) * psi_1d_jk(y, j, ky)
    elif e == 2:
        return 2 ** j * psi_1d_jk(x, j, kx) * phi_1d_jk(y, j, ky)
    else:
        return 2 ** j * psi_1d_jk(x, j, kx) * psi_1d_jk(y, j, ky)


def dF_dp(i, t):
    if i == 0:
        return (1 - t) ** 3
    elif i == 1:
        return 3 * (1 - t) ** 2 * t
    elif i == 2:
        return 3 * (1 - t) * t ** 2
    else:
        return t ** 3


def dFp_dp(i, t):
    if i == 0:
        return -3 * (1 - t) ** 2
    elif i == 1:
        return 3 * (-2 * (1 - t) * t + (1 - t) ** 2)
    elif i == 2:
        return 3 * (-t ** 2 + (1 - t) * 2 * t)
    else:
        return 3 * t ** 2


F = lambda t, para: dF_dp(0, t) * para[0] + dF_dp(1, t) * para[1] + dF_dp(2, t) * para[2] + dF_dp(3, t) * para[3]
Fp = lambda t, para: dFp_dp(0, t) * para[0] + dFp_dp(1, t) * para[1] + dFp_dp(2, t) * para[2] + dFp_dp(3, t) * para[3]


# Take x = [1, 4, 7, 7]
xPara = numpy.asarray([1, 4, 7, 7]) / 8.0
X = lambda t: F(t, xPara)
Xp = lambda t: Fp(t, xPara)
# Take x = [7, 3, 1, 1]

# Take y = [1, 1, 3, 7]
yPara = numpy.asarray([1, 1, 3, 7]) / 8.0
Y = lambda t: F(t, yPara)
Yp = lambda t: Fp(t, yPara)
# Take y = [7, 7, 3, 1]


inner_dc00_dx = lambda i, t: phi_1d(X(t)) * phi_1d(Y(t)) * dF_dp(i, t) * Yp(t)
inner_dc01_dx = lambda i, t, j, kx, ky: -(2 ** j) * cpsi_1d_jk(Y(t), j, ky) * dFp_dp(i, t) * phi_1d_jk(X(t), j, kx)
inner_dc10_dx = lambda i, t, j, kx, ky: 2 ** j * psi_1d_jk(X(t), j, kx) * dF_dp(i, t) * phi_1d_jk(Y(t), j, ky) * Yp(t)
inner_dc11_dx = lambda i, t, j, kx, ky: 2 ** j * psi_1d_jk(X(t), j, kx) * dF_dp(i, t) * psi_1d_jk(Y(t), j, ky) * Yp(t)


def dc_dx(i, j, kx, ky, e):
    if e == 0:
        return quad(lambda t: inner_dc00_dx(i, t), 0, 1)[0]
    elif e == 1:
        return quad(lambda t: inner_dc01_dx(i, t, j, kx, ky), 0, 1)[0]
    elif e == 2:
        return quad(lambda t: inner_dc10_dx(i, t, j, kx, ky), 0, 1)[0]
    else:
        return quad(lambda t: inner_dc11_dx(i, t, j, kx, ky), 0, 1)[0]


inner_dc00_dy = lambda i, t: cphi_1d(X(t)) * phi_1d(Y(t)) * dFp_dp(i, t)
inner_dc01_dy = lambda i, t, j, kx, ky: -(2 ** j) * psi_1d_jk(Y(t), j, ky) * dF_dp(i, t) * phi_1d_jk(X(t), j, kx) * Xp(
    t)
inner_dc10_dy = lambda i, t, j, kx, ky: 2 ** j * cpsi_1d_jk(X(t), j, kx) * dFp_dp(i, t) * phi_1d_jk(Y(t), j, ky)
inner_dc11_dy = lambda i, t, j, kx, ky: 2 ** j * cpsi_1d_jk(X(t), j, kx) * dFp_dp(i, t) * psi_1d_jk(Y(t), j, ky)


def dc_dy(i, j, kx, ky, e):
    if e == 0:
        return quad(lambda t: inner_dc00_dy(i, t), 0, 1)[0]
    elif e == 1:
        return quad(lambda t: inner_dc01_dy(i, t, j, kx, ky), 0, 1)[0]
    elif e == 2:
        return quad(lambda t: inner_dc10_dy(i, t, j, kx, ky), 0, 1)[0]
    else:
        return quad(lambda t: inner_dc11_dy(i, t, j, kx, ky), 0, 1)[0]


def Xi(t, i, xi):
    result = 0

    for k in range(4):
        if k != i:
            result += dF_dp(k, t) * xPara[k]
        else:
            result += dF_dp(k, t) * xi

    return result


def Xpi(t, i, xi):
    result = 0

    for k in range(4):
        if k != i:
            result += dFp_dp(k, t) * xPara[k]
        else:
            result += dFp_dp(k, t) * xi

    return result


inner_dc00_dx_test = lambda i, xi: \
    quad(lambda t: cphi_1d(Xi(t, i, xi)) * phi_1d(Y(t)) * Yp(t), 0, 1)[0]
inner_dc01_dx_test = lambda i, xi, j, kx, ky: \
    quad(lambda t: -(2 ** j) * cpsi_1d_jk(Y(t), j, ky) * phi_1d_jk(Xi(t, i, xi), j, kx) * Xpi(t, i, xi), 0, 1)[0]
inner_dc10_dx_test = lambda i, xi, j, kx, ky: \
    quad(lambda t: 2 ** j * cpsi_1d_jk(Xi(t, i, xi), j, kx) * phi_1d_jk(Y(t), j, ky) * Yp(t), 0, 1)[0]
inner_dc11_dx_test = lambda i, xi, j, kx, ky: \
    quad(lambda t: 2 ** j * cpsi_1d_jk(Xi(t, i, xi), j, kx) * psi_1d_jk(Y(t), j, ky) * Yp(t), 0, 1)[0]


def dc_dx_test(i, j, kx, ky, e):
    if e == 0:
        return derivative(lambda xi: inner_dc00_dx_test(i, xi), xPara[i], dx=1e-2)
    elif e == 1:
        return derivative(lambda xi: inner_dc01_dx_test(i, xi, j, kx, ky), xPara[i], dx=1e-2)
    elif e == 2:
        return derivative(lambda xi: inner_dc10_dx_test(i, xi, j, kx, ky), xPara[i], dx=1e-2)
    else:
        return derivative(lambda xi: inner_dc11_dx_test(i, xi, j, kx, ky), xPara[i], dx=1e-2)


inner_c00_jk = lambda t: cphi_1d(X(t)) * phi_1d(Y(t)) * Yp(t)
inner_c01_jk = lambda t, j, kx, ky: -(2**j) * cpsi_1d_jk(Y(t), j, ky) * phi_1d_jk(X(t), j, kx) * Xp(t)
inner_c10_jk = lambda t, j, kx, ky: 2**j * cpsi_1d_jk(X(t), j, kx) * phi_1d_jk(Y(t), j, ky) * Yp(t)
inner_c11_jk = lambda t, j, kx, ky: 2**j * cpsi_1d_jk(X(t), j, kx) * psi_1d_jk(Y(t), j, ky) * Yp(t)


def c_jk(j, kx, ky, e):
    if e == 0:
        return quad(inner_c00_jk, 0, 1)[0]
    elif e == 1:
        return quad(lambda t: inner_c01_jk(t, j, kx, ky), 0, 1)[0]
    elif e == 2:
        return quad(lambda t: inner_c10_jk(t, j, kx, ky), 0, 1)[0]
    else:
        return quad(lambda t: inner_c11_jk(t, j, kx, ky), 0, 1)[0]


MAX_I = 3
MAX_J = 2
MAX_K = 2 ** MAX_J - 1
MAX_E = 3


class Rasterization:
    def __init__(self):
        self.computeC()

    def computeC(self):
        self.m_c = numpy.zeros(shape=(MAX_J + 1, MAX_K + 1, MAX_K + 1, MAX_E + 1), dtype=numpy.float64)

        for j in range(MAX_J + 1):
            for kx in range(2**j):
                for ky in range(2**j):
                    for e in range(MAX_E + 1):
                        self.m_c[j][kx][ky][e] = c_jk(j, kx, ky, e)

    def R(self, x, y):
        result = self.m_c[0][0][0][0] * psi_jke(x, y, 0, 0, 0, 0)

        for j in range(MAX_J + 1):
            for kx in range(2**j):
                for ky in range(2**j):
                    for e in range(1, MAX_E + 1):
                        result += self.m_c[j][kx][ky][e] * psi_jke(x, y, j, kx, ky, e)

        return result


if __name__ == "__main__":
    import numpy
    numpy.warnings.simplefilter("ignore", Warning)

    img = numpy.zeros(shape=(8, 8), dtype=numpy.float64)

    xPara = numpy.asarray([1, 3, 7, 7]) / 8.0
    yPara = numpy.asarray([1, 1, 3, 7]) / 8.0
    r = Rasterization()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x][y] += r.R(x / 8.0, y / 8.0)

    xPara = numpy.asarray([7, 3, 1, 1]) / 8.0
    yPara = numpy.asarray([7, 7, 3, 1]) / 8.0
    r.computeC()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x][y] += r.R(x / 8.0, y / 8.0)

    file = open("test.txt", "w")
    for i in range(8):
        for j in range(8):
            file.write(str(img[i][j]) + " ")
        file.write("\n")
    file.close()

    minimum = numpy.min(img)
    maximum = numpy.max(img)
    img = (img - minimum) / float(maximum - minimum)

    import cv2
    cv2.imwrite("test.png", numpy.array(img * 255.0, dtype=numpy.uint8))

    # diff = numpy.ndarray(shape=(8, 8), dtype=numpy.float64)
    # i = 0
    # for line in file("raster_diff.txt", "r"):
    #     tokens = line.strip().split()
    #     for j in range(8):
    #         diff[i][j] = float(tokens[j])
    #     i += 1
    #
    # for i in range(MAX_I + 1):
    #     m_dc = numpy.zeros(shape=(MAX_J + 1, MAX_K + 1, MAX_K + 1, MAX_E + 1), dtype=numpy.float64)
    #
    #     for j in range(MAX_J + 1):
    #         for kx in range(2**j):
    #             for ky in range(2**j):
    #                 for e in range(MAX_E + 1):
    #                     m_dc[j][kx][ky][e] = dc_dx_test(i, j, kx, ky, e)
    #
    #     c_sum = 0
    #     g_sum = 0
    #
    #     for x in range(8):
    #         for y in range(8):
    #             tmp = m_dc[0][0][0][0] * psi_jke(x / 8.0, y / 8.0, 0, 0, 0, 0)
    #
    #             for j in range(MAX_J + 1):
    #                 for kx in range(2**j):
    #                     for ky in range(2**j):
    #                         for e in range(1, MAX_E + 1):
    #                             tmp += m_dc[j][kx][ky][e] * psi_jke(x / 8.0, y / 8.0, j, kx, ky, e)
    #
    #             g_sum += tmp
    #             c_sum += tmp * 2.0 * diff[x][y]
    #
    #     print g_sum, c_sum