import math
import copy
from collections import namedtuple

import numpy as np


# -----------------------------------------------------------------------------
Point = namedtuple('Point', 'x y')
# -----------------------------------------------------------------------------

class Rasterizer:
    def __init__(self, contour, w, h):
        self.w = w
        self.h = h
        self.max_j = int(math.ceil(math.log(max(w, h), 2))) - 1
        self.wh = 2 ** (self.max_j + 1)

        def normalize(p):
            return p[0] / float(self.wh), p[1] / float(self.wh)

        self.contour = copy.deepcopy(contour)
        self.contour.process(normalize)
        self.area = self.contour.area()
        self.lattice = [Point(*normalize((x, y)))
                        for x in xrange(h) for y in xrange(w)]
        # prepare all c
        self.all_c = {}
        for j in xrange(self.max_j + 1):
            for kx in xrange(2 ** j):
                for ky in xrange(2 ** j):
                    self.all_c[(j, kx, ky)] = self.c(j, (kx, ky))

    def psi(self, p, e, j, k):
        def psi_1d(p, e):
            if e == 0:
                return 1 if 0 <= p < 1 else 0
            else:
                return (1 if 0 <= p < 0.5 else -1) if 0 <= p < 1 else 0

        return 2 ** j * psi_1d(2 ** j * p.x - k.x, e.x) * psi_1d(2 ** j * p.y - k.y, e.y)

    def c(self, j, k):
        def transform(section, Q):
            return (2 ** (j + 1) * p[i] - k[i] * 2 - Q[i]
                    for p in section for i in xrange(2))

        Q_00, Q_01 = Point(0, 0), Point(0, 1)
        Q_10, Q_11 = Point(1, 0), Point(1, 1)
        c10, c01, c11 = 0, 0, 0
        for section in self.contour.each():
            KQ00, LQ00 = self.contour.get_KL(transform(section, Q_00))
            KQ01, LQ01 = self.contour.get_KL(transform(section, Q_01))
            KQ10, LQ10 = self.contour.get_KL(transform(section, Q_10))
            KQ11, LQ11 = self.contour.get_KL(transform(section, Q_11))
            c10 += LQ00.x + LQ01.x + KQ10.x \
                   - LQ10.x + KQ11.x - LQ11.x
            c01 += LQ00.y + LQ10.y + KQ01.y \
                   - LQ01.y + KQ11.y - LQ11.y
            c11 += LQ00.x - LQ01.x + KQ10.x \
                   - LQ10.x - KQ11.x + LQ11.x
        return c01, c10, c11

    def g(self, p):
        s = self.area
        E = [Point(0, 1), Point(1, 0), Point(1, 1)]
        for j in xrange(self.max_j + 1):
            for kx in xrange(2 ** j):
                for ky in xrange(2 ** j):
                    k = Point(kx, ky)
                    cs = self.all_c[(j, kx, ky)]
                    for i, e in enumerate(E):
                        psi = self.psi(p, e, j, k)
                        if psi > 0:
                            s += cs[i]
                        elif psi < 0:
                            s -= cs[i]
        return s

    def get(self):
        px_arr = [self.g(p) for p in self.lattice]
        px_mat = [px_arr[i * self.w: (i + 1) * self.w] for i in xrange(self.h)]
        return px_mat

    def get_fast(self):  # 100x faster than get()
        from util.getpx import get_px as get_cpp

        px_arr = get_cpp(self.area, self.max_j, self.all_c, self.lattice)
        px_mat = [px_arr[i * self.w: (i + 1) * self.w] for i in xrange(self.h)]
        return px_mat

# -----------------------------------------------------------------------------

def addmul(A, B, mul=None):
    if mul is None:
        for i, b in enumerate(B):
            A[i] += b
    else:
        for i, b in enumerate(B):
            A[i] += mul * b


class Vectorizer:
    def __init__(self, contour, org_img):
        h, w = org_img.shape[:2]
        self.w = w
        self.h = h
        self.org_img = org_img
        self.max_j = int(math.ceil(math.log(max(w, h), 2))) - 1
        self.wh = 2 ** (self.max_j + 1)
        self.contour = contour
        self.lattice = [Point(x, y) for x in xrange(h) for y in xrange(w)]
        self.num = len(self.contour.contour) * 2
        self.lattice = [(x, y) for x in xrange(h) for y in xrange(w)]
        # prepare all dc_dX
        self.all_dc = {}
        for j in xrange(self.max_j + 1):
            for kx in xrange(2 ** j):
                for ky in xrange(2 ** j):
                    self.all_dc[(j, kx, ky)] = [self.dc_dX(j, Point(kx, ky), ei)
                                                for ei in xrange(1, 4)]

    def psi(self, p, e, j, k):
        def psi_1d(p, e):
            if e == 0:
                return 1 if 0 <= p < 1 else 0
            else:
                return (1 if 0 <= p < 0.5 else -1) if 0 <= p < 1 else 0

        return 2 ** j * psi_1d(2 ** j * p.x - k.x, e.x) * psi_1d(2 ** j * p.y - k.y, e.y)

    def dc_dX(self, j, k, ei):
        def normalize(p):
            return p / float(self.wh)

        Q = [Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1)]
        sign = ((1, 1, 1, 1), (1, -1, 1, -1), (+1, +1, -1, -1), (+1, -1, -1, +1))
        grads = [0] * self.num

        for section, indice in self.contour.each_with_indice():
            section = list(normalize(pt[ip]) for pt in section for ip in xrange(2))

            for qi, q in enumerate(Q):
                left = (k.x + q.x * 0.5) / 2 ** j
                right = (k.x + q.x * 0.5 + 0.5) / 2 ** j
                bottom = (k.y + q.y * 0.5 + 0.5) / 2 ** j
                top = (k.y + q.y * 0.5) / 2 ** j

                sec_grads = self.contour.get_grads(section, ei, k, j, q,
                                                   sign[ei][qi], left, right, bottom, top)
                for i, idx in enumerate(indice):
                    grads[idx * 2] += sec_grads[i * 2] / 8.0
                    grads[idx * 2 + 1] += sec_grads[i * 2 + 1] / 8.0
        return grads

    def dLike_dX(self):
        E = [Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1)]
        # raster = Rasterizer(self.contour, self.w, self.h).get_fast()
        # raster = np.asarray(raster)
        diff = np.ndarray(shape=(8, 8), dtype=np.float64)
        i = 0
        for line in file("raster_diff.txt", "r"):
            tokens = line.strip().split()
            for j in range(8):
                diff[i][j] = float(tokens[j])
            i += 1

        grads = [0] * self.num
        grads_sum = [0] * self.num

        for x, y in self.lattice:
            p = Point(x / float(self.wh), y / float(self.wh))
            # dR/dX
            grads_R = [0] * self.num
            k0 = Point(0, 0)
            addmul(grads_R, self.dc_dX(0, k0, 0), self.psi(p, E[0], 0, k0))
            for j in xrange(self.max_j + 1):
                for kx in xrange(2 ** j):
                    for ky in xrange(2 ** j):
                        k = Point(kx, ky)
                        dcs = self.all_dc[(j, kx, ky)]
                        for ei in xrange(1, 4):
                            addmul(grads_R, dcs[ei - 1], self.psi(p, E[ei], j, k))
            addmul(grads_sum, grads_R)
            addmul(grads, grads_R, 2 * diff[x, y])
                   #2 * (float(raster[x, y]) - float(self.org_img[x, y])))
        print grads_sum
        return grads

    def like(self):
        raster = Rasterizer(self.contour, self.w, self.h).get_fast()
        raster = np.asarray(raster)
        s = 0
        for x, y in self.lattice:
            s += (float(raster[y, x]) - float(self.org_img[y, x])) ** 2
        return s

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from contour import *

    def f(X, printLike=True):
        contour = convertPathToContour(X)
        like = Vectorizer(contour, raster).like()
        if printLike:
            print like
        return like

    def fPrime(X, eps=1e-4):
        X = X.tolist()
        grads = [0] * len(X)

        for i in range(len(X)):
            appPathLarge = X[:]
            appPathLarge[i] += eps

            appPathSmall = X[:]
            appPathSmall[i] -= eps

            grads[i] = (f(appPathLarge, False) - f(appPathSmall, False)) / (2 * eps)

        return np.asarray(grads)

    oriPath = [1, 1, 3, 1, 7, 3, 7, 7, 3, 7, 1, 3]
    optPath = [1, 1, 4, 1, 7, 3, 7, 7, 3, 7, 1, 3]

    def convertPathToContour(path):
        pts = [(path[i], path[i + 1]) for i in xrange(0, len(path), 2)]
        return CubicBezier.Contour(pts)

    # ts = time.time()
    raster = Rasterizer(convertPathToContour(oriPath), 8, 8).get_fast()
    raster = np.array(raster)
    # print time.time() - ts, ' secs'

    # ts = time.time()
    contour = convertPathToContour(optPath)
    grads = Vectorizer(contour, raster).dLike_dX()
    print grads
    # print time.time() - ts, ' secs'

    print fPrime(np.asarray(optPath), 1e-4)

    #
    # from scipy import optimize
    # X0 = np.asarray(optPath)
    # print optimize.fmin_cg(f, X0, fPrime)