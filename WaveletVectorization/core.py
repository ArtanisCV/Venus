import numpy
from scipy.integrate import quad, quadrature
from scipy.misc import derivative
from WaveletVectorization.contour import CubicBezier
from WaveletVectorization.vectorizer import Rasterizer

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


phi_1d_jk = lambda x, j, k: phi_1d(2**j * x - k)
psi_1d_jk = lambda x, j, k: psi_1d(2**j * x - k)


cphi_1d = lambda x: quad(phi_1d, 0, x)[0]
cpsi_1d = lambda x: quad(psi_1d, 0, x)[0]
cphi_1d_jk = lambda x, j, k: quad(lambda t: phi_1d_jk(t, j, k), 0, x)[0]
cpsi_1d_jk = lambda x, j, k: quad(lambda t: psi_1d_jk(t, j, k), 0, x)[0]


psi_00_jk = lambda x, y, j, kx, ky: 2**j * phi_1d_jk(x, j, kx) * phi_1d_jk(y, j, ky)
psi_01_jk = lambda x, y, j, kx, ky: 2**j * phi_1d_jk(x, j, kx) * psi_1d_jk(y, j, ky)
psi_10_jk = lambda x, y, j, kx, ky: 2**j * psi_1d_jk(x, j, kx) * phi_1d_jk(y, j, ky)
psi_11_jk = lambda x, y, j, kx, ky: 2**j * psi_1d_jk(x, j, kx) * psi_1d_jk(y, j, ky)


def psi_jke(x, y, j, kx, ky, e):
    if e == 0:
        return psi_00_jk(x, y, j, kx, ky)
    elif e == 1:
        return psi_01_jk(x, y, j, kx, ky)
    elif e == 2:
        return psi_10_jk(x, y, j, kx, ky)
    else:
        return psi_11_jk(x, y, j, kx, ky)


def dF_p(i, t):
    if i == 0:
        return (1 - t)**3
    elif i == 1:
        return 3 * (1 - t)**2 * t
    elif i == 2:
        return 3 * (1 - t) * t**2
    else:
        return t**3


def dFp_p(i, t):
    if i == 0:
        return -3 * (1 - t)**2
    elif i == 1:
        return 3 * (-2 * (1 - t) * t + (1 - t)**2)
    elif i == 2:
        return 3 * (-t**2 + (1 - t) * 2 * t)
    else:
        return 3 * t**2


F = lambda t, para: dF_p(0, t) * para[0] + dF_p(1, t) * para[1] + dF_p(2, t) * para[2] + dF_p(3, t) * para[3]
Fp = lambda t, para: dFp_p(0, t) * para[0] + dFp_p(1, t) * para[1] + dFp_p(2, t) * para[2] + dFp_p(3, t) * para[3]


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


inner_dc00_dx = lambda i, t: phi_1d(X(t)) * phi_1d(Y(t)) * dF_p(i, t) * Yp(t)
inner_dc01_dx = lambda i, t, j, kx, ky: -(2**j) * cpsi_1d_jk(Y(t), j, ky) * dFp_p(i, t) * phi_1d_jk(X(t), j, kx)
inner_dc10_dx = lambda i, t, j, kx, ky: 2**j * psi_1d_jk(X(t), j, kx) * dF_p(i, t) * phi_1d_jk(Y(t), j, ky) * Yp(t)
inner_dc11_dx = lambda i, t, j, kx, ky: 2**j * psi_1d_jk(X(t), j, kx) * dF_p(i, t) * psi_1d_jk(Y(t), j, ky) * Yp(t)


def dc_dx(i, j, kx, ky, e):
    if e == 0:
        return quad(lambda t: inner_dc00_dx(i, t), 0, 1)[0]
    elif e == 1:
        return quadrature(lambda t: inner_dc01_dx(i, t, j, kx, ky), 0, 1, maxiter=125, vec_func=False)[0]
    elif e == 2:
        return quad(lambda t: inner_dc10_dx(i, t, j, kx, ky), 0, 1)[0]
    else:
        return quad(lambda t: inner_dc11_dx(i, t, j, kx, ky), 0, 1)[0]


inner_dc00_dy = lambda i, t: cphi_1d(X(t)) * phi_1d(Y(t)) * dFp_p(i, t)
inner_dc01_dy = lambda i, t, j, kx, ky: -(2**j) * psi_1d_jk(Y(t), j, ky) * dF_p(i, t) * phi_1d_jk(X(t), j, kx) * Xp(t)
inner_dc10_dy = lambda i, t, j, kx, ky: 2**j * cpsi_1d_jk(X(t), j, kx) * dFp_p(i, t) * phi_1d_jk(Y(t), j, ky)
inner_dc11_dy = lambda i, t, j, kx, ky: 2**j * cpsi_1d_jk(X(t), j, kx) * dFp_p(i, t) * psi_1d_jk(Y(t), j, ky)


def dc_dy(i, j, kx, ky, e):
    if e == 0:
        return quadrature(lambda t: inner_dc00_dy(i, t), 0, 1, maxiter=125, vec_func=False)[0]
    elif e == 1:
        return quad(lambda t: inner_dc01_dy(i, t, j, kx, ky), 0, 1)[0]
    elif e == 2:
        return quadrature(lambda t: inner_dc10_dy(i, t, j, kx, ky), 0, 1, maxiter=125, vec_func=False)[0]
    else:
        return quadrature(lambda t: inner_dc11_dy(i, t, j, kx, ky), 0, 1, maxiter=125, vec_func=False)[0]


def Xi(t, i, xi):
    result = 0

    for k in range(4):
        if k != i:
            result += dF_p(k, t) * xPara[k]
        else:
            result += dF_p(k, t) * xi

    return result


def Xpi(t, i, xi):
    result = 0

    for k in range(4):
        if k != i:
            result += dFp_p(k, t) * xPara[k]
        else:
            result += dFp_p(k, t) * xi

    return result


inner_dc00_dx_test = lambda i, xi: \
    quad(lambda t: cphi_1d(Xi(t, i, xi)) * phi_1d(Y(t)) * Yp(t), 0, 1)[0]
inner_dc01_dx_test = lambda i, xi, j, kx, ky: \
    quad(lambda t: -(2**j) * cpsi_1d_jk(Y(t), j, ky) * phi_1d_jk(Xi(t, i, xi), j, kx) * Xpi(t, i, xi), 0, 1)[0]
inner_dc10_dx_test = lambda i, xi, j, kx, ky: \
    quad(lambda t: 2**j * cpsi_1d_jk(Xi(t, i, xi), j, kx) * phi_1d_jk(Y(t), j, ky) * Yp(t), 0, 1)[0]
inner_dc11_dx_test = lambda i, xi, j, kx, ky: \
    quad(lambda t: 2**j * cpsi_1d_jk(Xi(t, i, xi), j, kx) * psi_1d_jk(Y(t), j, ky) * Yp(t), 0, 1)[0]


def dc_dx_test(i, j, kx, ky, e):
    if e == 0:
        return derivative(lambda xi: inner_dc00_dx_test(i, xi), xPara[i])
    elif e == 1:
        return derivative(lambda xi: inner_dc01_dx_test(i, xi, j, kx, ky), xPara[i])
    elif e == 2:
        return derivative(lambda xi: inner_dc10_dx_test(i, xi, j, kx, ky), xPara[i])
    else:
        return derivative(lambda xi: inner_dc11_dx_test(i, xi, j, kx, ky), xPara[i])


if __name__ == "__main__":
    import numpy
    numpy.warnings.simplefilter("ignore", Warning)

    MAX_I = 3
    MAX_J = 2
    MAX_K = 2**MAX_J - 1
    MAX_E = 3

    oriPath = [1, 1, 3, 1, 7, 3, 7, 7, 3, 7, 1, 3]
    optPath = [1, 1, 4, 1, 7, 3, 7, 7, 3, 7, 1, 3]

    def convertPathToContour(path):
        pts = [(path[i], path[i + 1]) for i in xrange(0, len(path), 2)]
        return CubicBezier.Contour(pts)

    ori_raster = Rasterizer(convertPathToContour(oriPath), 8, 8).get_fast()
    ori_raster = numpy.array(ori_raster)
    opt_raster = Rasterizer(convertPathToContour(optPath), 8, 8).get_fast()
    opt_raster = numpy.array(opt_raster)

    for i in range(MAX_I + 1):
        m_dc = numpy.zeros(shape=(MAX_J + 1, MAX_K + 1, MAX_K + 1, MAX_E + 1), dtype=numpy.float64)

        for j in range(MAX_J + 1):
            for kx in range(2**j):
                for ky in range(2**j):
                    for e in range(MAX_E + 1):
                        m_dc[j][kx][ky][e] = dc_dx(i, j, kx, ky, e)

        result = 0
        for x in range(8):
            for y in range(8):
                tmp = m_dc[0][0][0][0] * psi_jke(x / 8.0, y / 8.0, 0, 0, 0, 0)

                for j in range(MAX_J + 1):
                    for kx in range(2**j):
                        for ky in range(2**j):
                            for e in range(1, MAX_E + 1):
                                if i == 0 and x == 2 and y == 0:
                                    print m_dc[j][kx][ky][e], j, kx, ky, e
                                tmp += m_dc[j][kx][ky][e] * psi_jke(x / 8.0, y / 8.0, j, kx, ky, e)

                result += tmp * 2.0 * (opt_raster[x][y] - ori_raster[x][y])

        print result