import math, copy, numpy as np
from collections import namedtuple

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
        self.lattice = [Point(*normalize((x, y))) \
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
            return (2 ** (j + 1) * p[i] - k[i] * 2 - Q[i] \
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

    def get_fast(self): # 100x faster than get()
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

    def psi(self, p, e, j, k):
        def psi_1d(p, e):
            if e == 0:
                return 1 if 0 <= p < 1 else 0
            else:
                return (1 if 0 <= p < 0.5 else -1) if 0 <= p < 1 else 0

        return 2 ** j * psi_1d(2 ** j * p.x - k.x, e.x) * psi_1d(2 ** j * p.y - k.y, e.y)

    def dc_dX(self, p, j, k, ei):
        def normalize(p):
            return p / float(self.wh)

        Q = [Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1)]
        E = [Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1)]
        sign = ((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1))
        grads = [0] * self.num
        pre = 2 ** (self.max_j + 1 - j)
        for qi, q in enumerate(Q):
            left = pre * (k.x + q.x * 0.5)
            right = pre * (k.x + q.x * 0.5 + 0.5)
            bottom = pre * (k.y + q.y * 0.5 + 0.5)
            top = pre * (k.y + q.y * 0.5)
            fac = self.psi(Point(normalize(p.x), normalize(p.y)), E[ei], j, k)
            for section, indice in self.contour.each_with_indice():
                section = (normalize(pt[ip]) \
                           for pt in section for ip in xrange(2))
                left, right = normalize(left), normalize(right)
                bottom, top = normalize(bottom), normalize(top)
                sec_grads = self.contour.get_grads(section, ei, k, j, q, sign[ei][qi],
                                                   left, right, bottom, top)
                for i, idx in enumerate(indice):
                    grads[idx * 2] += fac * sec_grads[i * 2]
                    grads[idx * 2 + 1] += fac * sec_grads[i * 2 + 1]
        return grads

    def dLike_dX(self):
        raster = Rasterizer(self.contour, self.w, self.h).get_fast()
        raster = np.array(np.asarray(raster) * 255 + 0.5, np.uint8)
        grads = [0] * self.num
        # for x, y in self.lattice:
        #     p = Point(x, y)
        #
        #     # dR/dX
        #     grads_R = self.dc_dX(p, 0, Point(0, 0), 0)
        #     for j in xrange(self.max_j + 1):
        #         for kx in xrange(2 ** j):
        #             for ky in xrange(2 ** j):
        #                 k = Point(kx, ky)
        #                 for ei in xrange(1, 4):
        #                     addmul(grads_R, self.dc_dX(p, j, k, ei))
        #     addmul(grads, grads_R, 2 * (float(raster[y, x]) - float(self.org_img[y, x])))

        p = Point(1, 2)

        # dR/dX
        grads_R = self.dc_dX(p, 0, Point(0, 0), 0)
        for j in xrange(self.max_j + 1):
            for kx in xrange(2 ** j):
                for ky in xrange(2 ** j):
                    k = Point(kx, ky)
                    for ei in xrange(1, 4):
                        addmul(grads_R, self.dc_dX(p, j, k, ei))

        addmul(grads, grads_R, 2 * (float(raster[2, 1]) - float(self.org_img[2, 1])))
        return grads

    def like(self):
        raster = Rasterizer(self.contour, self.w, self.h).get_fast()
        raster = np.array(np.asarray(raster) * 255 + 0.5, np.uint8)
        s = 0
        for x, y in self.lattice:
            s += (float(raster[y, x]) - float(self.org_img[y, x])) ** 2
        return s


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import cv2, time
    from contour import *

    oriPath = [1, 1, 3, 1, 7, 3, 7, 7, 3, 7, 1, 3]
    optPath = [1, 1, 3.5, 1, 7, 3, 7, 7, 3, 7, 1, 3]

    appPaths = [[1, 1, 3.5002, 1, 7, 3, 7, 7, 3, 7, 1, 3],
               [1, 1, 3.4998, 1, 7, 3, 7, 7, 3, 7, 1, 3],
               [1, 1, 3.51, 1, 7, 3, 7, 7, 3, 7, 1, 3],
               [1, 1, 3.49, 1, 7, 3, 7, 7, 3, 7, 1, 3],
               [1, 1, 3.6, 1, 7, 3, 7, 7, 3, 7, 1, 3],
               [1, 1, 3.4, 1, 7, 3, 7, 7, 3, 7, 1, 3]]

    def convertPathToContour(path):
        pts = [(path[i], path[i+1]) for i in xrange(0, len(path), 2)]
        return CubicBezier.Contour(pts)

    # ts = time.time()
    raster = Rasterizer(convertPathToContour(oriPath), 8, 8).get_fast()
    raster = np.array(np.asarray(raster) * 255 + 0.5, np.uint8)
    cv2.imwrite('CubicBezier.png', raster)
    # print time.time() - ts, ' secs'

    # ts = time.time()
    contour = convertPathToContour(optPath)
    grads = Vectorizer(contour, raster).dLike_dX()
    print grads
    like = Vectorizer(contour, raster).like()
    print like
    # print time.time() - ts, ' secs'

    for appPath in appPaths:
        print appPath

        optC = convertPathToContour(optPath)
        appC = convertPathToContour(appPath)

        like1 = Vectorizer(optC, raster).like()
        like2 = Vectorizer(appC, raster).like()
        print "Like:", like2
        print "Der:", (like1 - like2) / (optPath[2] - appPath[2])

    # def f(X):
    #     contour = convertPathToContour(X)
    #     s = Vectorizer(contour, raster).like()
    #     print s
    #     return s
    #
    # from scipy import optimize
    # X0 = np.asarray(optPath)
    # print optimize.fmin_cg(f, X0, epsilon=0.03)