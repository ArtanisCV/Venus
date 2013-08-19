import numpy as np
from collections import namedtuple
from WaveletVectorization.util.solver import cubic

# -----------------------------------------------------------------------------
Point = namedtuple('Point', 'x y')
# -----------------------------------------------------------------------------

class CubicBezier:

    def __init__(self, x0, y0, x1, y1, x2, y2, x3, y3):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.x2, self.y2, self.x3, self.y3 = x2, y2, x3, y3

    def evaluate(self, t):
        return (self.x0*(1-t)**3 + 3*self.x1*(1-t)**2*t \
                + 3*self.x2*(1-t)*t**2 + self.x3*t**3,
                self.y0*(1-t)**3 + 3*self.y1*(1-t)**2*t \
                + 3*self.y2*(1-t)*t**2 + self.y3*t**3)

    def subsection(self, t0, t1):
        u0 = 1.0 - t0
        u1 = 1.0 - t1

        qxa = self.x0*u0*u0 + self.x1*2*t0*u0 + self.x2*t0*t0
        qxb = self.x0*u1*u1 + self.x1*2*t1*u1 + self.x2*t1*t1
        qxc = self.x1*u0*u0 + self.x2*2*t0*u0 + self.x3*t0*t0
        qxd = self.x1*u1*u1 + self.x2*2*t1*u1 + self.x3*t1*t1

        qya = self.y0*u0*u0 + self.y1*2*t0*u0 + self.y2*t0*t0
        qyb = self.y0*u1*u1 + self.y1*2*t1*u1 + self.y2*t1*t1
        qyc = self.y1*u0*u0 + self.y2*2*t0*u0 + self.y3*t0*t0
        qyd = self.y1*u1*u1 + self.y2*2*t1*u1 + self.y3*t1*t1

        sec = CubicBezier(  qxa*u0 + qxc*t0, qya*u0 + qyc*t0,
                            qxa*u1 + qxc*t1, qya*u1 + qyc*t1,
                            qxb*u0 + qxd*t0, qyb*u0 + qyd*t0,
                            qxb*u1 + qxd*t1, qyb*u1 + qyd*t1 )
        return sec

    def clip(self, left, right, bottom, top):
        def is_t_in(t, eps = 1e-5):
            pt = self.evaluate(t)
            return left-eps<=pt[0]<=right+eps and top-eps<=pt[1]<=bottom+eps

        ax = -self.x0 + 3*self.x1 - 3*self.x2 + self.x3
        bx = 3*self.x0 - 6*self.x1 + 3*self.x2
        cx, _dx = 3*self.x1 - 3*self.x0, self.x0
        ay = -self.y0 + 3*self.y1 - 3*self.y2 + self.y3
        by = 3*self.y0 - 6*self.y1 + 3*self.y2
        cy, _dy = 3*self.y1 - 3*self.y0, self.y0
        ts = [0]
        ts += cubic(ax, bx, cx, _dx-left)
        ts += cubic(ax, bx, cx, _dx-right)
        ts += cubic(ay, by, cy, _dy-bottom)
        ts += cubic(ay, by, cy, _dy-top)
        ts.append(1)
        ts = [t for t in ts if 0 <= t <= 1 and is_t_in(t)]
        ts = sorted(ts)
        ts = [t for i, t in enumerate(ts) if t != ts[i-1]]
        pairs = [(ts[i-1], t) for i, t in enumerate(ts) \
                    if i > 0 and is_t_in((t + ts[i-1]) * 0.5)]
        sections = [self.subsection(a, b) for a, b in pairs]
        return sections

    def get_KL(self, eps = 1e-5):
        Kx, Ky, Lx, Ly = 0, 0, 0, 0
        for sec in self.clip(0, 1, 1, 0):
            v3 = Point(sec.x0, sec.y0)
            v2 = Point(sec.x1, sec.y1)
            v1 = Point(sec.x2, sec.y2)
            v0 = Point(sec.x3, sec.y3)
            if abs(v0.x - 1) < eps and abs(v1.x - 1) < eps \
                and abs(v2.x - 1) < eps and abs(v3.x - 1) < eps\
            or abs(v0.y - 1) < eps and abs(v1.y - 1) < eps \
                and abs(v2.y - 1) < eps and abs(v3.y - 1) < eps:
                continue

            Kx += 1./4 * (v0.y - v3.y)
            Ky += 1./4 * (v3.x - v0.x)
            Lx += 1./80* (6 * v2.y*v3.x + 3 * v1.y*(v2.x+v3.x) \
                        + v0.y * (6*v1.x+3*v2.x+v3.x) \
                        - 6 * v2.x*v3.y - 3 * v1.x*(v2.y+v3.y) \
                        - 10 * v3.x*v3.y \
                        + v0.x * (10*v0.y-6*v1.y-3*v2.y-v3.y) )
            Ly += 1./80* (6 * v2.y*v3.x + 3 * v1.y*(v2.x+v3.x) \
                        + v0.y * (6*v1.x+3*v2.x+v3.x) \
                        - 6 * v2.x*v3.y - 3 * v1.x*(v2.y+v3.y) \
                        + 10 * v3.x*v3.y \
                        - v0.x * (10*v0.y+6*v1.y+3*v2.y+v3.y) )
        return Point(Kx, Ky), Point(Lx, Ly)

    # -------------------------------------------------------------------------

    def clip_t(self, left, right, bottom, top):
        def is_t_in(t, eps=1e-5):
            pt = self.evaluate(t)
            return left - eps <= pt[0] <= right + eps and top - eps <= pt[1] <= bottom + eps

        ax = -self.x0 + 3*self.x1 - 3*self.x2 + self.x3
        bx = 3*self.x0 - 6*self.x1 + 3*self.x2
        cx, _dx = 3*self.x1 - 3*self.x0, self.x0
        ay = -self.y0 + 3*self.y1 - 3*self.y2 + self.y3
        by = 3*self.y0 - 6*self.y1 + 3*self.y2
        cy, _dy = 3*self.y1 - 3*self.y0, self.y0
        ts = [0]
        ts += cubic(ax, bx, cx, _dx-left)
        ts += cubic(ax, bx, cx, _dx-right)
        ts += cubic(ay, by, cy, _dy-bottom)
        ts += cubic(ay, by, cy, _dy-top)
        ts.append(1)
        ts = [t for t in ts if 0 <= t <= 1 and is_t_in(t)]
        ts = sorted(ts)
        ts = [t for i, t in enumerate(ts) if t != ts[i-1]]
        pairs = [(ts[i-1], t) for i, t in enumerate(ts) \
                    if i > 0 and is_t_in((t + ts[i-1]) * 0.5)]
        return pairs

    def F_c10_x0(self, t):
        return (self.y0/2. - (3*self.y1)/2. + (3*self.y2)/2. - self.y3/2.)*t**6 + ((39*self.y1)/5. - 3*self.y0 - (33*self.y2)/5. + (9*self.y3)/5.)*t**5 + ((15*self.y0)/2. - (33*self.y1)/2. + (45*self.y2)/4. - (9*self.y3)/4.)*t**4 + (18*self.y1 - 10*self.y0 - 9*self.y2 + self.y3)*t**3 + ((15*self.y0)/2. - (21*self.y1)/2. + 3*self.y2)*t**2 + (3*self.y1 - 3*self.y0)*t
    def F_c10_x1(self, t):
        return ((9*self.y1)/2. - (3*self.y0)/2. - (9*self.y2)/2. + (3*self.y3)/2.)*t**6 + ((36*self.y0)/5. - 18*self.y1 + (72*self.y2)/5. - (18*self.y3)/5.)*t**5 + (27*self.y1 - (27*self.y0)/2. - (63*self.y2)/4. + (9*self.y3)/4.)*t**4 + (12*self.y0 - 18*self.y1 + 6*self.y2)*t**3 + ((9*self.y1)/2. - (9*self.y0)/2.)*t**2
    def F_c10_x2(self, t):
        return ((3*self.y0)/2. - (9*self.y1)/2. + (9*self.y2)/2. - (3*self.y3)/2.)*t**6 + ((63*self.y1)/5. - (27*self.y0)/5. - 9*self.y2 + (9*self.y3)/5.)*t**5 + ((27*self.y0)/4. - (45*self.y1)/4. + (9*self.y2)/2.)*t**4 + (3*self.y1 - 3*self.y0)*t**3
    def F_c10_x3(self, t):
        return ((3*self.y1)/2. - self.y0/2. - (3*self.y2)/2. + self.y3/2.)*t**6 + ((6*self.y0)/5. - (12*self.y1)/5. + (6*self.y2)/5.)*t**5 + ((3*self.y1)/4. - (3*self.y0)/4.)*t**4
    def K_c10_y0(self, t):
        return (self.x0/2. - (3*self.x1)/2. + (3*self.x2)/2. - self.x3/2.)*t**6 + ((36*self.x1)/5. - 3*self.x0 - (27*self.x2)/5. + (6*self.x3)/5.)*t**5 + ((15*self.x0)/2. - (27*self.x1)/2. + (27*self.x2)/4. - (3*self.x3)/4.)*t**4 + (12*self.x1 - 10*self.x0 - 3*self.x2)*t**3 + ((15*self.x0)/2. - (9*self.x1)/2.)*t**2 - 3*self.x0*t
    def L_c10_y0(self, t):
        return -(t - 1)**3
    def K_c10_y1(self, t):
        return ((9*self.x1)/2. - (3*self.x0)/2. - (9*self.x2)/2. + (3*self.x3)/2.)*t**6 + ((39*self.x0)/5. - 18*self.x1 + (63*self.x2)/5. - (12*self.x3)/5.)*t**5 + (27*self.x1 - (33*self.x0)/2. - (45*self.x2)/4. + (3*self.x3)/4.)*t**4 + (18*self.x0 - 18*self.x1 + 3*self.x2)*t**3 + ((9*self.x1)/2. - (21*self.x0)/2.)*t**2 + 3*self.x0*t
    def L_c10_y1(self, t):
        return 3*t*(t - 1)**2
    def K_c10_y2(self, t):
        return ((3*self.x0)/2. - (9*self.x1)/2. + (9*self.x2)/2. - (3*self.x3)/2.)*t**6 + ((72*self.x1)/5. - (33*self.x0)/5. - 9*self.x2 + (6*self.x3)/5.)*t**5 + ((45*self.x0)/4. - (63*self.x1)/4. + (9*self.x2)/2.)*t**4 + (6*self.x1 - 9*self.x0)*t**3 + 3*self.x0*t**2
    def L_c10_y2(self, t):
        return -3*t**2*(t - 1)
    def K_c10_y3(self, t):
        return ((3*self.x1)/2. - self.x0/2. - (3*self.x2)/2. + self.x3/2.)*t**6 + ((9*self.x0)/5. - (18*self.x1)/5. + (9*self.x2)/5.)*t**5 + ((9*self.x1)/4. - (9*self.x0)/4.)*t**4 + self.x0*t**3
    def L_c10_y3(self, t):
        return t**3
    def K_c01_x0(self, t):
        return (self.y0/2. - (3*self.y1)/2. + (3*self.y2)/2. - self.y3/2.)*t**6 + ((36*self.y1)/5. - 3*self.y0 - (27*self.y2)/5. + (6*self.y3)/5.)*t**5 + ((15*self.y0)/2. - (27*self.y1)/2. + (27*self.y2)/4. - (3*self.y3)/4.)*t**4 + (12*self.y1 - 10*self.y0 - 3*self.y2)*t**3 + ((15*self.y0)/2. - (9*self.y1)/2.)*t**2 - 3*self.y0*t
    def L_c01_x0(self, t):
        return -(t - 1)**3
    def K_c01_x1(self, t):
        return ((9*self.y1)/2. - (3*self.y0)/2. - (9*self.y2)/2. + (3*self.y3)/2.)*t**6 + ((39*self.y0)/5. - 18*self.y1 + (63*self.y2)/5. - (12*self.y3)/5.)*t**5 + (27*self.y1 - (33*self.y0)/2. - (45*self.y2)/4. + (3*self.y3)/4.)*t**4 + (18*self.y0 - 18*self.y1 + 3*self.y2)*t**3 + ((9*self.y1)/2. - (21*self.y0)/2.)*t**2 + 3*self.y0*t
    def L_c01_x1(self, t):
        return 3*t*(t - 1)**2
    def K_c01_x2(self, t):
        return ((3*self.y0)/2. - (9*self.y1)/2. + (9*self.y2)/2. - (3*self.y3)/2.)*t**6 + ((72*self.y1)/5. - (33*self.y0)/5. - 9*self.y2 + (6*self.y3)/5.)*t**5 + ((45*self.y0)/4. - (63*self.y1)/4. + (9*self.y2)/2.)*t**4 + (6*self.y1 - 9*self.y0)*t**3 + 3*self.y0*t**2
    def L_c01_x2(self, t):
        return -3*t**2*(t - 1)
    def K_c01_x3(self, t):
        return ((3*self.y1)/2. - self.y0/2. - (3*self.y2)/2. + self.y3/2.)*t**6 + ((9*self.y0)/5. - (18*self.y1)/5. + (9*self.y2)/5.)*t**5 + ((9*self.y1)/4. - (9*self.y0)/4.)*t**4 + self.y0*t**3
    def L_c01_x3(self, t):
        return t**3
    def F_c01_y0(self, t):
        return (self.x0/2. - (3*self.x1)/2. + (3*self.x2)/2. - self.x3/2.)*t**6 + ((39*self.x1)/5. - 3*self.x0 - (33*self.x2)/5. + (9*self.x3)/5.)*t**5 + ((15*self.x0)/2. - (33*self.x1)/2. + (45*self.x2)/4. - (9*self.x3)/4.)*t**4 + (18*self.x1 - 10*self.x0 - 9*self.x2 + self.x3)*t**3 + ((15*self.x0)/2. - (21*self.x1)/2. + 3*self.x2)*t**2 + (3*self.x1 - 3*self.x0)*t
    def F_c01_y1(self, t):
        return ((9*self.x1)/2. - (3*self.x0)/2. - (9*self.x2)/2. + (3*self.x3)/2.)*t**6 + ((36*self.x0)/5. - 18*self.x1 + (72*self.x2)/5. - (18*self.x3)/5.)*t**5 + (27*self.x1 - (27*self.x0)/2. - (63*self.x2)/4. + (9*self.x3)/4.)*t**4 + (12*self.x0 - 18*self.x1 + 6*self.x2)*t**3 + ((9*self.x1)/2. - (9*self.x0)/2.)*t**2
    def F_c01_y2(self, t):
        return ((3*self.x0)/2. - (9*self.x1)/2. + (9*self.x2)/2. - (3*self.x3)/2.)*t**6 + ((63*self.x1)/5. - (27*self.x0)/5. - 9*self.x2 + (9*self.x3)/5.)*t**5 + ((27*self.x0)/4. - (45*self.x1)/4. + (9*self.x2)/2.)*t**4 + (3*self.x1 - 3*self.x0)*t**3
    def F_c01_y3(self, t):
        return ((3*self.x1)/2. - self.x0/2. - (3*self.x2)/2. + self.x3/2.)*t**6 + ((6*self.x0)/5. - (12*self.x1)/5. + (6*self.x2)/5.)*t**5 + ((3*self.x1)/4. - (3*self.x0)/4.)*t**4
    def F_c11_x0(self, t):
        return (self.y0/2. - (3*self.y1)/2. + (3*self.y2)/2. - self.y3/2.)*t**6 + ((39*self.y1)/5. - 3*self.y0 - (33*self.y2)/5. + (9*self.y3)/5.)*t**5 + ((15*self.y0)/2. - (33*self.y1)/2. + (45*self.y2)/4. - (9*self.y3)/4.)*t**4 + (18*self.y1 - 10*self.y0 - 9*self.y2 + self.y3)*t**3 + ((15*self.y0)/2. - (21*self.y1)/2. + 3*self.y2)*t**2 + (3*self.y1 - 3*self.y0)*t
    def F_c11_x1(self, t):
        return ((9*self.y1)/2. - (3*self.y0)/2. - (9*self.y2)/2. + (3*self.y3)/2.)*t**6 + ((36*self.y0)/5. - 18*self.y1 + (72*self.y2)/5. - (18*self.y3)/5.)*t**5 + (27*self.y1 - (27*self.y0)/2. - (63*self.y2)/4. + (9*self.y3)/4.)*t**4 + (12*self.y0 - 18*self.y1 + 6*self.y2)*t**3 + ((9*self.y1)/2. - (9*self.y0)/2.)*t**2
    def F_c11_x2(self, t):
        return ((3*self.y0)/2. - (9*self.y1)/2. + (9*self.y2)/2. - (3*self.y3)/2.)*t**6 + ((63*self.y1)/5. - (27*self.y0)/5. - 9*self.y2 + (9*self.y3)/5.)*t**5 + ((27*self.y0)/4. - (45*self.y1)/4. + (9*self.y2)/2.)*t**4 + (3*self.y1 - 3*self.y0)*t**3
    def F_c11_x3(self, t):
        return ((3*self.y1)/2. - self.y0/2. - (3*self.y2)/2. + self.y3/2.)*t**6 + ((6*self.y0)/5. - (12*self.y1)/5. + (6*self.y2)/5.)*t**5 + ((3*self.y1)/4. - (3*self.y0)/4.)*t**4
    def K_c11_y0(self, t):
        return (self.x0/2. - (3*self.x1)/2. + (3*self.x2)/2. - self.x3/2.)*t**6 + ((36*self.x1)/5. - 3*self.x0 - (27*self.x2)/5. + (6*self.x3)/5.)*t**5 + ((15*self.x0)/2. - (27*self.x1)/2. + (27*self.x2)/4. - (3*self.x3)/4.)*t**4 + (12*self.x1 - 10*self.x0 - 3*self.x2)*t**3 + ((15*self.x0)/2. - (9*self.x1)/2.)*t**2 - 3*self.x0*t
    def L_c11_y0(self, t):
        return -(t - 1)**3
    def K_c11_y1(self, t):
        return ((9*self.x1)/2. - (3*self.x0)/2. - (9*self.x2)/2. + (3*self.x3)/2.)*t**6 + ((39*self.x0)/5. - 18*self.x1 + (63*self.x2)/5. - (12*self.x3)/5.)*t**5 + (27*self.x1 - (33*self.x0)/2. - (45*self.x2)/4. + (3*self.x3)/4.)*t**4 + (18*self.x0 - 18*self.x1 + 3*self.x2)*t**3 + ((9*self.x1)/2. - (21*self.x0)/2.)*t**2 + 3*self.x0*t
    def L_c11_y1(self, t):
        return 3*t*(t - 1)**2
    def K_c11_y2(self, t):
        return ((3*self.x0)/2. - (9*self.x1)/2. + (9*self.x2)/2. - (3*self.x3)/2.)*t**6 + ((72*self.x1)/5. - (33*self.x0)/5. - 9*self.x2 + (6*self.x3)/5.)*t**5 + ((45*self.x0)/4. - (63*self.x1)/4. + (9*self.x2)/2.)*t**4 + (6*self.x1 - 9*self.x0)*t**3 + 3*self.x0*t**2
    def L_c11_y2(self, t):
        return -3*t**2*(t - 1)
    def K_c11_y3(self, t):
        return ((3*self.x1)/2. - self.x0/2. - (3*self.x2)/2. + self.x3/2.)*t**6 + ((9*self.x0)/5. - (18*self.x1)/5. + (9*self.x2)/5.)*t**5 + ((9*self.x1)/4. - (9*self.x0)/4.)*t**4 + self.x0*t**3
    def L_c11_y3(self, t):
        return t**3
    def F_c00_x0(self, t):
        return (self.y0/2. - (3*self.y1)/2. + (3*self.y2)/2. - self.y3/2.)*t**6 + ((39*self.y1)/5. - 3*self.y0 - (33*self.y2)/5. + (9*self.y3)/5.)*t**5 + ((15*self.y0)/2. - (33*self.y1)/2. + (45*self.y2)/4. - (9*self.y3)/4.)*t**4 + (18*self.y1 - 10*self.y0 - 9*self.y2 + self.y3)*t**3 + ((15*self.y0)/2. - (21*self.y1)/2. + 3*self.y2)*t**2 + (3*self.y1 - 3*self.y0)*t
    def F_c00_x1(self, t):
        return ((9*self.y1)/2. - (3*self.y0)/2. - (9*self.y2)/2. + (3*self.y3)/2.)*t**6 + ((36*self.y0)/5. - 18*self.y1 + (72*self.y2)/5. - (18*self.y3)/5.)*t**5 + (27*self.y1 - (27*self.y0)/2. - (63*self.y2)/4. + (9*self.y3)/4.)*t**4 + (12*self.y0 - 18*self.y1 + 6*self.y2)*t**3 + ((9*self.y1)/2. - (9*self.y0)/2.)*t**2
    def F_c00_x2(self, t):
        return ((3*self.y0)/2. - (9*self.y1)/2. + (9*self.y2)/2. - (3*self.y3)/2.)*t**6 + ((63*self.y1)/5. - (27*self.y0)/5. - 9*self.y2 + (9*self.y3)/5.)*t**5 + ((27*self.y0)/4. - (45*self.y1)/4. + (9*self.y2)/2.)*t**4 + (3*self.y1 - 3*self.y0)*t**3
    def F_c00_x3(self, t):
        return ((3*self.y1)/2. - self.y0/2. - (3*self.y2)/2. + self.y3/2.)*t**6 + ((6*self.y0)/5. - (12*self.y1)/5. + (6*self.y2)/5.)*t**5 + ((3*self.y1)/4. - (3*self.y0)/4.)*t**4
    def F_c00_y0(self, t):
        return (self.x0/2. - (3*self.x1)/2. + (3*self.x2)/2. - self.x3/2.)*t**6 + ((36*self.x1)/5. - 3*self.x0 - (27*self.x2)/5. + (6*self.x3)/5.)*t**5 + ((15*self.x0)/2. - (27*self.x1)/2. + (27*self.x2)/4. - (3*self.x3)/4.)*t**4 + (12*self.x1 - 10*self.x0 - 3*self.x2)*t**3 + ((15*self.x0)/2. - (9*self.x1)/2.)*t**2 - 3*self.x0*t
    def F_c00_y1(self, t):
        return ((9*self.x1)/2. - (3*self.x0)/2. - (9*self.x2)/2. + (3*self.x3)/2.)*t**6 + ((39*self.x0)/5. - 18*self.x1 + (63*self.x2)/5. - (12*self.x3)/5.)*t**5 + (27*self.x1 - (33*self.x0)/2. - (45*self.x2)/4. + (3*self.x3)/4.)*t**4 + (18*self.x0 - 18*self.x1 + 3*self.x2)*t**3 + ((9*self.x1)/2. - (21*self.x0)/2.)*t**2 + 3*self.x0*t
    def F_c00_y2(self, t):
        return ((3*self.x0)/2. - (9*self.x1)/2. + (9*self.x2)/2. - (3*self.x3)/2.)*t**6 + ((72*self.x1)/5. - (33*self.x0)/5. - 9*self.x2 + (6*self.x3)/5.)*t**5 + ((45*self.x0)/4. - (63*self.x1)/4. + (9*self.x2)/2.)*t**4 + (6*self.x1 - 9*self.x0)*t**3 + 3*self.x0*t**2
    def F_c00_y3(self, t):
        return ((3*self.x1)/2. - self.x0/2. - (3*self.x2)/2. + self.x3/2.)*t**6 + ((9*self.x0)/5. - (18*self.x1)/5. + (9*self.x2)/5.)*t**5 + ((9*self.x1)/4. - (9*self.x0)/4.)*t**4 + self.x0*t**3

    def partial_c00_x0(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_x0(t1) - self.F_c00_x0(t0))
    def partial_c00_x1(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_x1(t1) - self.F_c00_x1(t0))
    def partial_c00_x2(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_x2(t1) - self.F_c00_x2(t0))
    def partial_c00_x3(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_x3(t1) - self.F_c00_x3(t0))
    def partial_c00_y0(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_y0(t1) - self.F_c00_y0(t0))
    def partial_c00_y1(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_y1(t1) - self.F_c00_y1(t0))
    def partial_c00_y2(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_y2(t1) - self.F_c00_y2(t0))
    def partial_c00_y3(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c00_y3(t1) - self.F_c00_y3(t0))
    def partial_c01_x0(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return -( 2**j * (self.K_c01_x0(t1) - self.K_c01_x0(t0)) +  (-k.y) * (self.L_c01_x0(t1) - self.L_c01_x0(t0)))
        elif q.x == 0 and q.y == 1: return -(-2**j * (self.K_c01_x0(t1) - self.K_c01_x0(t0)) + (k.y+1) * (self.L_c01_x0(t1) - self.L_c01_x0(t0)))
        elif q.x == 1 and q.y == 0: return -( 2**j * (self.K_c01_x0(t1) - self.K_c01_x0(t0)) +  (-k.y) * (self.L_c01_x0(t1) - self.L_c01_x0(t0)))
        else:                       return -(-2**j * (self.K_c01_x0(t1) - self.K_c01_x0(t0)) + (k.y+1) * (self.L_c01_x0(t1) - self.L_c01_x0(t0)))
    def partial_c01_x1(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return -( 2**j * (self.K_c01_x1(t1) - self.K_c01_x1(t0)) +  (-k.y) * (self.L_c01_x1(t1) - self.L_c01_x1(t0)))
        elif q.x == 0 and q.y == 1: return -(-2**j * (self.K_c01_x1(t1) - self.K_c01_x1(t0)) + (k.y+1) * (self.L_c01_x1(t1) - self.L_c01_x1(t0)))
        elif q.x == 1 and q.y == 0: return -( 2**j * (self.K_c01_x1(t1) - self.K_c01_x1(t0)) +  (-k.y) * (self.L_c01_x1(t1) - self.L_c01_x1(t0)))
        else:                       return -(-2**j * (self.K_c01_x1(t1) - self.K_c01_x1(t0)) + (k.y+1) * (self.L_c01_x1(t1) - self.L_c01_x1(t0)))
    def partial_c01_x2(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return -( 2**j * (self.K_c01_x2(t1) - self.K_c01_x2(t0)) +  (-k.y) * (self.L_c01_x2(t1) - self.L_c01_x2(t0)))
        elif q.x == 0 and q.y == 1: return -(-2**j * (self.K_c01_x2(t1) - self.K_c01_x2(t0)) + (k.y+1) * (self.L_c01_x2(t1) - self.L_c01_x2(t0)))
        elif q.x == 1 and q.y == 0: return -( 2**j * (self.K_c01_x2(t1) - self.K_c01_x2(t0)) +  (-k.y) * (self.L_c01_x2(t1) - self.L_c01_x2(t0)))
        else:                       return -(-2**j * (self.K_c01_x2(t1) - self.K_c01_x2(t0)) + (k.y+1) * (self.L_c01_x2(t1) - self.L_c01_x2(t0)))
    def partial_c01_x3(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return -( 2**j * (self.K_c01_x3(t1) - self.K_c01_x3(t0)) +  (-k.y) * (self.L_c01_x3(t1) - self.L_c01_x3(t0)))
        elif q.x == 0 and q.y == 1: return -(-2**j * (self.K_c01_x3(t1) - self.K_c01_x3(t0)) + (k.y+1) * (self.L_c01_x3(t1) - self.L_c01_x3(t0)))
        elif q.x == 1 and q.y == 0: return -( 2**j * (self.K_c01_x3(t1) - self.K_c01_x3(t0)) +  (-k.y) * (self.L_c01_x3(t1) - self.L_c01_x3(t0)))
        else:                       return -(-2**j * (self.K_c01_x3(t1) - self.K_c01_x3(t0)) + (k.y+1) * (self.L_c01_x3(t1) - self.L_c01_x3(t0)))
    def partial_c01_y0(self, k, j, t0, t1, q, sign):
        return -(sign * 2**j * (self.F_c01_y0(t1) - self.F_c01_y0(t0)))
    def partial_c01_y1(self, k, j, t0, t1, q, sign):
        return -(sign * 2**j * (self.F_c01_y1(t1) - self.F_c01_y1(t0)))
    def partial_c01_y2(self, k, j, t0, t1, q, sign):
        return -(sign * 2**j * (self.F_c01_y2(t1) - self.F_c01_y2(t0)))
    def partial_c01_y3(self, k, j, t0, t1, q, sign):
        return -(sign * 2**j * (self.F_c01_y3(t1) - self.F_c01_y3(t0)))
    def partial_c10_x0(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c10_x0(t1) - self.F_c10_x0(t0))
    def partial_c10_x1(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c10_x1(t1) - self.F_c10_x1(t0))
    def partial_c10_x2(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c10_x2(t1) - self.F_c10_x2(t0))
    def partial_c10_x3(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c10_x3(t1) - self.F_c10_x3(t0))
    def partial_c10_y0(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return  2**j * (self.K_c10_y0(t1) - self.K_c10_y0(t0)) +  (-k.x) * (self.L_c10_y0(t1) - self.L_c10_y0(t0))
        elif q.x == 0 and q.y == 1: return  2**j * (self.K_c10_y0(t1) - self.K_c10_y0(t0)) +  (-k.x) * (self.L_c10_y0(t1) - self.L_c10_y0(t0))
        elif q.x == 1 and q.y == 0: return -2**j * (self.K_c10_y0(t1) - self.K_c10_y0(t0)) + (k.x+1) * (self.L_c10_y0(t1) - self.L_c10_y0(t0))
        else:                       return -2**j * (self.K_c10_y0(t1) - self.K_c10_y0(t0)) + (k.x+1) * (self.L_c10_y0(t1) - self.L_c10_y0(t0))
    def partial_c10_y1(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return  2**j * (self.K_c10_y1(t1) - self.K_c10_y1(t0)) +  (-k.x) * (self.L_c10_y1(t1) - self.L_c10_y1(t0))
        elif q.x == 0 and q.y == 1: return  2**j * (self.K_c10_y1(t1) - self.K_c10_y1(t0)) +  (-k.x) * (self.L_c10_y1(t1) - self.L_c10_y1(t0))
        elif q.x == 1 and q.y == 0: return -2**j * (self.K_c10_y1(t1) - self.K_c10_y1(t0)) + (k.x+1) * (self.L_c10_y1(t1) - self.L_c10_y1(t0))
        else:                       return -2**j * (self.K_c10_y1(t1) - self.K_c10_y1(t0)) + (k.x+1) * (self.L_c10_y1(t1) - self.L_c10_y1(t0))
    def partial_c10_y2(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return  2**j * (self.K_c10_y2(t1) - self.K_c10_y2(t0)) +  (-k.x) * (self.L_c10_y2(t1) - self.L_c10_y2(t0))
        elif q.x == 0 and q.y == 1: return  2**j * (self.K_c10_y2(t1) - self.K_c10_y2(t0)) +  (-k.x) * (self.L_c10_y2(t1) - self.L_c10_y2(t0))
        elif q.x == 1 and q.y == 0: return -2**j * (self.K_c10_y2(t1) - self.K_c10_y2(t0)) + (k.x+1) * (self.L_c10_y2(t1) - self.L_c10_y2(t0))
        else:                       return -2**j * (self.K_c10_y2(t1) - self.K_c10_y2(t0)) + (k.x+1) * (self.L_c10_y2(t1) - self.L_c10_y2(t0))
    def partial_c10_y3(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return  2**j * (self.K_c10_y3(t1) - self.K_c10_y3(t0)) +  (-k.x) * (self.L_c10_y3(t1) - self.L_c10_y3(t0))
        elif q.x == 0 and q.y == 1: return  2**j * (self.K_c10_y3(t1) - self.K_c10_y3(t0)) +  (-k.x) * (self.L_c10_y3(t1) - self.L_c10_y3(t0))
        elif q.x == 1 and q.y == 0: return -2**j * (self.K_c10_y3(t1) - self.K_c10_y3(t0)) + (k.x+1) * (self.L_c10_y3(t1) - self.L_c10_y3(t0))
        else:                       return -2**j * (self.K_c10_y3(t1) - self.K_c10_y3(t0)) + (k.x+1) * (self.L_c10_y3(t1) - self.L_c10_y3(t0))
    def partial_c11_x0(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c11_x0(t1) - self.F_c11_x0(t0))
    def partial_c11_x1(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c11_x1(t1) - self.F_c11_x1(t0))
    def partial_c11_x2(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c11_x2(t1) - self.F_c11_x2(t0))
    def partial_c11_x3(self, k, j, t0, t1, q, sign):
        return sign * 2**j * (self.F_c11_x3(t1) - self.F_c11_x3(t0))
    def partial_c11_y0(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return       2**j * (self.K_c11_y0(t1) - self.K_c11_y0(t0)) +  (-k.x) * (self.L_c11_y0(t1) - self.L_c11_y0(t0))
        elif q.x == 0 and q.y == 1: return   - ( 2**j * (self.K_c11_y0(t1) - self.K_c11_y0(t0)) +  (-k.x) * (self.L_c11_y0(t1) - self.L_c11_y0(t0)))
        elif q.x == 1 and q.y == 0: return       2**j * (self.K_c11_y0(t1) - self.K_c11_y0(t0)) + (k.x+1) * (self.L_c11_y0(t1) - self.L_c11_y0(t0))
        else:                       return       2**j * (self.K_c11_y0(t1) - self.K_c11_y0(t0)) + (k.x+1) * (self.L_c11_y0(t1) - self.L_c11_y0(t0))
    def partial_c11_y1(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return       2**j * (self.K_c11_y1(t1) - self.K_c11_y1(t0)) +  (-k.x) * (self.L_c11_y1(t1) - self.L_c11_y1(t0))
        elif q.x == 0 and q.y == 1: return   - ( 2**j * (self.K_c11_y1(t1) - self.K_c11_y1(t0)) +  (-k.x) * (self.L_c11_y1(t1) - self.L_c11_y1(t0)))
        elif q.x == 1 and q.y == 0: return       2**j * (self.K_c11_y1(t1) - self.K_c11_y1(t0)) + (k.x+1) * (self.L_c11_y1(t1) - self.L_c11_y1(t0))
        else:                       return       2**j * (self.K_c11_y1(t1) - self.K_c11_y1(t0)) + (k.x+1) * (self.L_c11_y1(t1) - self.L_c11_y1(t0))
    def partial_c11_y2(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return       2**j * (self.K_c11_y2(t1) - self.K_c11_y2(t0)) +  (-k.x) * (self.L_c11_y2(t1) - self.L_c11_y2(t0))
        elif q.x == 0 and q.y == 1: return   - ( 2**j * (self.K_c11_y2(t1) - self.K_c11_y2(t0)) +  (-k.x) * (self.L_c11_y2(t1) - self.L_c11_y2(t0)))
        elif q.x == 1 and q.y == 0: return       2**j * (self.K_c11_y2(t1) - self.K_c11_y2(t0)) + (k.x+1) * (self.L_c11_y2(t1) - self.L_c11_y2(t0))
        else:                       return       2**j * (self.K_c11_y2(t1) - self.K_c11_y2(t0)) + (k.x+1) * (self.L_c11_y2(t1) - self.L_c11_y2(t0))
    def partial_c11_y3(self, k, j, t0, t1, q, sign):
        if   q.x == 0 and q.y == 0: return       2**j * (self.K_c11_y3(t1) - self.K_c11_y3(t0)) +  (-k.x) * (self.L_c11_y3(t1) - self.L_c11_y3(t0))
        elif q.x == 0 and q.y == 1: return   - ( 2**j * (self.K_c11_y3(t1) - self.K_c11_y3(t0)) +  (-k.x) * (self.L_c11_y3(t1) - self.L_c11_y3(t0)))
        elif q.x == 1 and q.y == 0: return       2**j * (self.K_c11_y3(t1) - self.K_c11_y3(t0)) + (k.x+1) * (self.L_c11_y3(t1) - self.L_c11_y3(t0))
        else:                       return       2**j * (self.K_c11_y3(t1) - self.K_c11_y3(t0)) + (k.x+1) * (self.L_c11_y3(t1) - self.L_c11_y3(t0))

    def get_grads(self, section, ei, k, j, q, sign, left, right, bottom, top):
        partial_matrix = (
           (self.partial_c00_x0,  self.partial_c01_x0,  self.partial_c10_x0,  self.partial_c11_x0),
           (self.partial_c00_y0,  self.partial_c01_y0,  self.partial_c10_y0,  self.partial_c11_y0),
           (self.partial_c00_x1,  self.partial_c01_x1,  self.partial_c10_x1,  self.partial_c11_x1),
           (self.partial_c00_y1,  self.partial_c01_y1,  self.partial_c10_y1,  self.partial_c11_y1),
           (self.partial_c00_x2,  self.partial_c01_x2,  self.partial_c10_x2,  self.partial_c11_x2),
           (self.partial_c00_y2,  self.partial_c01_y2,  self.partial_c10_y2,  self.partial_c11_y2),
           (self.partial_c00_x3,  self.partial_c01_x3,  self.partial_c10_x3,  self.partial_c11_x3),
           (self.partial_c00_y3,  self.partial_c01_y3,  self.partial_c10_y3,  self.partial_c11_y3)
        )
        grads = [0] * 8
        for t0, t1 in self.clip_t(left, right, bottom, top):
            for i in xrange(8):
                method = partial_matrix[i][ei]
                if method is not None:
                    grads[i] += method(k, j, t0, t1, q, sign)
        return grads


# -----------------------------------------------------------------------------

class Contour:

    def __init__(self, contour):
        self.contour = contour

    def __str__(self):
        info = ' '.join('(%2.1f, %2.1f)' % (s[0], s[1]) for s in self.contour)
        return ' :\t'.join(['CubicBezier', info])

    def process(self, method):
        self.contour = [method(p) for p in self.contour]

    def area(self):
        def det(a, b):  return a[0] * b[1] - a[1] * b[0]
        s = 0
        for v0, v1, v2, v3 in self.each():
            s += 3./10 * det(v0,v1) + 3./20 * det(v1,v2) + 3./10 * det(v2,v3) \
               + 3./20 * det(v0,v2) + 3./20 * det(v1,v3) + 1./20 * det(v0,v3)
        return s

    def each(self):
        for i in xrange(0, len(self.contour), 3):
            v3 = self.contour[i]
            v2 = self.contour[i-1]
            v1 = self.contour[i-2]
            v0 = self.contour[i-3]
            yield v0, v1, v2, v3

    def to_lines(self):
        tts = np.linspace(0,1, num=50)
        for section in self.each():
            section = (s[i] for s in section for i in xrange(2))
            bezier = CubicBezier(*section)
            for start, end in zip(tts[:-1], tts[1:]):
                sx, sy = bezier.evaluate(start)
                ex, ey = bezier.evaluate(end)
                yield sx, sy, ex, ey

    def get_KL(self, section):
        bezier = CubicBezier(*section)
        return bezier.get_KL()

    # -------------------------------------------------------------------------

    def each_with_indice(self):
        for i in xrange(0, 1, 3):
            v3 = self.contour[i]
            v2 = self.contour[i-1]
            v1 = self.contour[i-2]
            v0 = self.contour[i-3]
            yield (v0, v1, v2, v3), (i-3, i-2, i-1, i)

    def get_grads(self, section, ei, k, j, q, sign, left, right, bottom, top):
        bezier = CubicBezier(*section)
        return bezier.get_grads(section, ei, k, j, q, sign, left, right, bottom, top)

# -----------------------------------------------------------------------------