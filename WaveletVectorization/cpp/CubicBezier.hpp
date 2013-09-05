#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <tuple>
using namespace std;


template <typename T>
struct List : public vector<T>
{
    List() : vector() {}
    List(size_type n, const T& val = T()) : vector<T>(n, val) {}
    List(const List& list) : vector<T>(list) {}

    template <class InputIterator>
    List(InputIterator first, InputIterator last) : vector<T>(first, last) {}
};

// manual
int solveCubic(double c3, double c2, double c1, double c0, 
               double & s0, double & s1, double & s2);

inline List<double> cubic(double a, double b, double c, double d)
{
    double s0, s1, s2;
    int num = solveCubic(a, b, c, d, s0, s1, s2);

    List<double> result;
    if (num > 0)
        result.push_back(s0);
    if (num > 1)
        result.push_back(s1);
    if (num > 2)
        result.push_back(s2);

    return result;
}

struct Point
{
    Point(double x, double y):
        x(x), y(y)
    {
    }

    double x, y;
};
//

template <typename T>
List<T> Filter(const List<T>& li, function<bool(double)> f)
{
    List<T> result;

    for (auto __item : li)
        if (f(__item))
            result.push_back(__item);

    return result;
}

template <typename T1, typename T2>
List<T2> Map(const List<T1>& li, function<T2(T1)> f)
{
    List<T2> result;

    for (auto __item : li)
        result.push_back(f(__item));

    return result;
}


class CubicBezier
{
public:
    CubicBezier(double x0, double y0, double x1, double y1, 
                double x2, double y2, double x3, double y3):
        x0(x0), y0(y0), x1(x1), y1(y1), 
        x2(x2), y2(y2), x3(x3), y3(y3)
    {
    }

    tuple<double, double> evaluate(double t)
    {
        return make_tuple(this->x0*pow((1-t),3) + 3*this->x1*pow((1-t),2)*t
            + 3*this->x2*(1-t)*pow(t,2) + this->x3*pow(t,3),
            this->y0*pow((1-t),3) + 3*this->y1*pow(1-t,2)*t
            + 3*this->y2*(1-t)*pow(t,2) + this->y3*pow(t,3));
    }

    CubicBezier subsection(double t0, double t1)
    {
        auto u0 = 1.0 - t0;
        auto u1 = 1.0 - t1;

        auto qxa = this->x0*u0*u0 + this->x1*2*t0*u0 + this->x2*t0*t0;
        auto qxb = this->x0*u1*u1 + this->x1*2*t1*u1 + this->x2*t1*t1;
        auto qxc = this->x1*u0*u0 + this->x2*2*t0*u0 + this->x3*t0*t0;
        auto qxd = this->x1*u1*u1 + this->x2*2*t1*u1 + this->x3*t1*t1;

        auto qya = this->y0*u0*u0 + this->y1*2*t0*u0 + this->y2*t0*t0;
        auto qyb = this->y0*u1*u1 + this->y1*2*t1*u1 + this->y2*t1*t1;
        auto qyc = this->y1*u0*u0 + this->y2*2*t0*u0 + this->y3*t0*t0;
        auto qyd = this->y1*u1*u1 + this->y2*2*t1*u1 + this->y3*t1*t1;

        CubicBezier sec(qxa*u0 + qxc*t0, qya*u0 + qyc*t0,
                        qxa*u1 + qxc*t1, qya*u1 + qyc*t1,
                        qxb*u0 + qxd*t0, qyb*u0 + qyd*t0,
                        qxb*u1 + qxd*t1, qyb*u1 + qyd*t1);
        return sec;
    }

    // manual
    List<Point> clipTrue(double left, double right, double bottom, double top)
    //
    {
        auto is_t_in = [this, left, right, bottom, top](double t, double eps) -> bool
        {
            auto pt = this->evaluate(t);
            return left-eps<=get<0>(pt)&&get<0>(pt)<=right+eps && top-eps<=get<1>(pt)&&get<1>(pt)<=bottom+eps;
        };
        auto is_subsection_valid = [right, bottom](CubicBezier sec, double eps) -> bool
        {
            return  abs(sec.x0 - right) < eps && abs(sec.x1 - right) < eps
                && abs(sec.x2 - right) < eps && abs(sec.x3 - right) < eps
                || abs(sec.y0 -bottom) < eps && abs(sec.y1 -bottom) < eps
                && abs(sec.y2 -bottom) < eps && abs(sec.y3 -bottom) < eps;
        };

        auto ax = -this->x0 + 3*this->x1 - 3*this->x2 + this->x3;
        auto bx = 3*this->x0 - 6*this->x1 + 3*this->x2;
        auto cx = 3*this->x1 - 3*this->x0;
        auto _dx = this->x0;
        auto ay = -this->y0 + 3*this->y1 - 3*this->y2 + this->y3;
        auto by = 3*this->y0 - 6*this->y1 + 3*this->y2;
        auto cy = 3*this->y1 - 3*this->y0;
        auto _dy = this->y0;

        List<double> ts;
        ts.push_back(0);
        auto __tmp0 = cubic(ax, bx, cx, _dx-left);
        ts.insert(ts.end(), __tmp0.begin(), __tmp0.end());
        auto __tmp1 = cubic(ax, bx, cx, _dx-right);
        ts.insert(ts.end(), __tmp1.begin(), __tmp1.end()); 
        auto __tmp2 = cubic(ay, by, cy, _dy-bottom);
        ts.insert(ts.end(), __tmp2.begin(), __tmp2.end());
        auto __tmp3 = cubic(ay, by, cy, _dy-top);
        ts.insert(ts.end(), __tmp3.begin(), __tmp3.end());
        ts.push_back(1);
        ts = Filter(ts, [is_t_in](double t){ return 0 <= t && t <= 1 && is_t_in(t, 1e-5); });
        sort(ts.begin(), ts.end());

        // manual
        List<double> tmp;
        for (int i = 0; i < ts.size(); i++)
        {
            auto t = ts[i];

            if (i > 0 && t != ts[i-1] || t != ts.back())
                tmp.push_back(t);
        }
        ts = tmp;

        List<Point> pairs;
        for (int i = 0; i < ts.size(); i++)
        {
            auto t = ts[i];

            if (i > 0 && is_t_in((t + ts[i-1]) * 0.5, 1e-5))
                pairs.push_back(Point(ts[i-1], t));
        }

        List<Point> sections;
        for (auto __item : pairs)
        {
            auto a = __item.x;
            auto b = __item.y;

            auto sec = this->subsection(a, b);
            if (!is_subsection_valid(sec, 1e-5))
                sections.push_back(__item);
        }

        return sections;
        //
    }

    // manual
    List<CubicBezier> clipFalse(double left, double right, double bottom, double top)
    //
    {
        auto is_t_in = [this, left, right, bottom, top](double t, double eps) -> bool
        {
            auto pt = this->evaluate(t);
            return left-eps<=get<0>(pt)&&get<0>(pt)<=right+eps && top-eps<=get<1>(pt)&&get<1>(pt)<=bottom+eps;
        };
        auto is_subsection_valid = [right, bottom](CubicBezier sec, double eps) -> bool
        {
            return  abs(sec.x0 - right) < eps && abs(sec.x1 - right) < eps
                && abs(sec.x2 - right) < eps && abs(sec.x3 - right) < eps
                || abs(sec.y0 -bottom) < eps && abs(sec.y1 -bottom) < eps
                && abs(sec.y2 -bottom) < eps && abs(sec.y3 -bottom) < eps;
        };

        auto ax = -this->x0 + 3*this->x1 - 3*this->x2 + this->x3;
        auto bx = 3*this->x0 - 6*this->x1 + 3*this->x2;
        auto cx = 3*this->x1 - 3*this->x0;
        auto _dx = this->x0;
        auto ay = -this->y0 + 3*this->y1 - 3*this->y2 + this->y3;
        auto by = 3*this->y0 - 6*this->y1 + 3*this->y2;
        auto cy = 3*this->y1 - 3*this->y0;
        auto _dy = this->y0;

        List<double> ts;
        ts.push_back(0);
        auto __tmp0 = cubic(ax, bx, cx, _dx-left);
        ts.insert(ts.end(), __tmp0.begin(), __tmp0.end());
        auto __tmp1 = cubic(ax, bx, cx, _dx-right);
        ts.insert(ts.end(), __tmp1.begin(), __tmp1.end()); 
        auto __tmp2 = cubic(ay, by, cy, _dy-bottom);
        ts.insert(ts.end(), __tmp2.begin(), __tmp2.end());
        auto __tmp3 = cubic(ay, by, cy, _dy-top);
        ts.insert(ts.end(), __tmp3.begin(), __tmp3.end());
        ts.push_back(1);
        ts = Filter(ts, [is_t_in](double t){ return 0 <= t && t <= 1 && is_t_in(t, 1e-5); });
        sort(ts.begin(), ts.end());

        // manual
        List<double> tmp;
        for (int i = 0; i < ts.size(); i++)
        {
            auto t = ts[i];

            if (i > 0 && t != ts[i-1] || t != ts.back())
                tmp.push_back(t);
        }
        ts = tmp;

        List<Point> pairs;
        for (int i = 0; i < ts.size(); i++)
        {
            auto t = ts[i];

            if (i > 0 && is_t_in((t + ts[i-1]) * 0.5, 1e-5))
                pairs.push_back(Point(ts[i-1], t));
        }

        List<CubicBezier> sections;
        for (auto __item : pairs)
        {
            auto a = __item.x;
            auto b = __item.y;

            auto sec = this->subsection(a, b);
            if (!is_subsection_valid(sec, 1e-5))
                sections.push_back(sec);
        }

        return sections;
        //
    }

    tuple<Point, Point> get_KL(double eps = 1e-5)
    {
        auto Kx = 0.0, Ky = 0.0, Lx = 0.0, Ly = 0.0;
        for (auto sec : this->clipFalse(0, 1, 1, 0))
        {
            auto v3 = Point(sec.x0, sec.y0);
            auto v2 = Point(sec.x1, sec.y1);
            auto v1 = Point(sec.x2, sec.y2);
            auto v0 = Point(sec.x3, sec.y3);
            if (abs(v0.x - 1) < eps && abs(v1.x - 1) < eps
                && abs(v2.x - 1) < eps && abs(v3.x - 1) < eps
                || abs(v0.y - 1) < eps && abs(v1.y - 1) < eps
                && abs(v2.y - 1) < eps && abs(v3.y - 1) < eps)
                continue;

            Kx += 1./4 * (v0.y - v3.y);
            Ky += 1./4 * (v3.x - v0.x);
            Lx += 1./80* (6 * v2.y*v3.x + 3 * v1.y*(v2.x+v3.x)
                        + v0.y * (6*v1.x+3*v2.x+v3.x)
                        - 6 * v2.x*v3.y - 3 * v1.x*(v2.y+v3.y)
                        - 10 * v3.x*v3.y
                        + v0.x * (10*v0.y-6*v1.y-3*v2.y-v3.y) );
            Ly += 1./80* (6 * v2.y*v3.x + 3 * v1.y*(v2.x+v3.x)
                        + v0.y * (6*v1.x+3*v2.x+v3.x)
                        - 6 * v2.x*v3.y - 3 * v1.x*(v2.y+v3.y)
                        + 10 * v3.x*v3.y
                        - v0.x * (10*v0.y+6*v1.y+3*v2.y+v3.y) );
        }

        return make_tuple(Point(Kx, Ky), Point(Lx, Ly));
    }

    //////////////////////////////////////////////////////////////////////////

    double F_c10_x0(double t)
    {
        return (this->y0/2. - (3*this->y1)/2. + (3*this->y2)/2. - this->y3/2.)*pow(t,6) + ((39*this->y1)/5. - 3*this->y0 - (33*this->y2)/5. + (9*this->y3)/5.)*pow(t,5) + ((15*this->y0)/2. - (33*this->y1)/2. + (45*this->y2)/4. - (9*this->y3)/4.)*pow(t,4) + (18*this->y1 - 10*this->y0 - 9*this->y2 + this->y3)*pow(t,3) + ((15*this->y0)/2. - (21*this->y1)/2. + 3*this->y2)*pow(t,2) + (3*this->y1 - 3*this->y0)*t;
    }
    
    double F_c10_x1(double t)
    {
        return ((9*this->y1)/2. - (3*this->y0)/2. - (9*this->y2)/2. + (3*this->y3)/2.)*pow(t,6) + ((36*this->y0)/5. - 18*this->y1 + (72*this->y2)/5. - (18*this->y3)/5.)*pow(t,5) + (27*this->y1 - (27*this->y0)/2. - (63*this->y2)/4. + (9*this->y3)/4.)*pow(t,4) + (12*this->y0 - 18*this->y1 + 6*this->y2)*pow(t,3) + ((9*this->y1)/2. - (9*this->y0)/2.)*pow(t,2);
    }

    double F_c10_x2(double t)
    {
        return ((3*this->y0)/2. - (9*this->y1)/2. + (9*this->y2)/2. - (3*this->y3)/2.)*pow(t,6) + ((63*this->y1)/5. - (27*this->y0)/5. - 9*this->y2 + (9*this->y3)/5.)*pow(t,5) + ((27*this->y0)/4. - (45*this->y1)/4. + (9*this->y2)/2.)*pow(t,4) + (3*this->y1 - 3*this->y0)*pow(t,3);
    }

    double F_c10_x3(double t)
    {
        return ((3*this->y1)/2. - this->y0/2. - (3*this->y2)/2. + this->y3/2.)*pow(t,6) + ((6*this->y0)/5. - (12*this->y1)/5. + (6*this->y2)/5.)*pow(t,5) + ((3*this->y1)/4. - (3*this->y0)/4.)*pow(t,4);
    }

    double K_c10_y0(double t)
    {
        return (this->x0/2. - (3*this->x1)/2. + (3*this->x2)/2. - this->x3/2.)*pow(t,6) + ((36*this->x1)/5. - 3*this->x0 - (27*this->x2)/5. + (6*this->x3)/5.)*pow(t,5) + ((15*this->x0)/2. - (27*this->x1)/2. + (27*this->x2)/4. - (3*this->x3)/4.)*pow(t,4) + (12*this->x1 - 10*this->x0 - 3*this->x2)*pow(t,3) + ((15*this->x0)/2. - (9*this->x1)/2.)*pow(t,2) - 3*this->x0*t;
    }
    
    double L_c10_y0(double t)
    {
        return -pow((t - 1),3);
    }
    
    double K_c10_y1(double t)
    {
        return ((9*this->x1)/2. - (3*this->x0)/2. - (9*this->x2)/2. + (3*this->x3)/2.)*pow(t,6) + ((39*this->x0)/5. - 18*this->x1 + (63*this->x2)/5. - (12*this->x3)/5.)*pow(t,5) + (27*this->x1 - (33*this->x0)/2. - (45*this->x2)/4. + (3*this->x3)/4.)*pow(t,4) + (18*this->x0 - 18*this->x1 + 3*this->x2)*pow(t,3) + ((9*this->x1)/2. - (21*this->x0)/2.)*pow(t,2) + 3*this->x0*t;
    }
    
    double L_c10_y1(double t)
    {
        return 3*t*pow((t - 1),2);
    }
    
    double K_c10_y2(double t)
    {
        return ((3*this->x0)/2. - (9*this->x1)/2. + (9*this->x2)/2. - (3*this->x3)/2.)*pow(t,6) + ((72*this->x1)/5. - (33*this->x0)/5. - 9*this->x2 + (6*this->x3)/5.)*pow(t,5) + ((45*this->x0)/4. - (63*this->x1)/4. + (9*this->x2)/2.)*pow(t,4) + (6*this->x1 - 9*this->x0)*pow(t,3) + 3*this->x0*pow(t,2);
    }
    
    double L_c10_y2(double t)
    {
        return -3*pow(t,2)*(t - 1);
    }
    
    double K_c10_y3(double t)
    {
        return ((3*this->x1)/2. - this->x0/2. - (3*this->x2)/2. + this->x3/2.)*pow(t,6) + ((9*this->x0)/5. - (18*this->x1)/5. + (9*this->x2)/5.)*pow(t,5) + ((9*this->x1)/4. - (9*this->x0)/4.)*pow(t,4) + this->x0*pow(t,3);   
    }
    
    double L_c10_y3(double t)
    {
        return pow(t,3);
    }

    double K_c01_x0(double t)
    {
        return (this->y0/2. - (3*this->y1)/2. + (3*this->y2)/2. - this->y3/2.)*pow(t,6) + ((36*this->y1)/5. - 3*this->y0 - (27*this->y2)/5. + (6*this->y3)/5.)*pow(t,5) + ((15*this->y0)/2. - (27*this->y1)/2. + (27*this->y2)/4. - (3*this->y3)/4.)*pow(t,4) + (12*this->y1 - 10*this->y0 - 3*this->y2)*pow(t,3) + ((15*this->y0)/2. - (9*this->y1)/2.)*pow(t,2) - 3*this->y0*t;
    }
    
    double L_c01_x0(double t)
    {
        return -pow((t - 1),3);
    }
    
    double K_c01_x1(double t)
    {
        return ((9*this->y1)/2. - (3*this->y0)/2. - (9*this->y2)/2. + (3*this->y3)/2.)*pow(t,6) + ((39*this->y0)/5. - 18*this->y1 + (63*this->y2)/5. - (12*this->y3)/5.)*pow(t,5) + (27*this->y1 - (33*this->y0)/2. - (45*this->y2)/4. + (3*this->y3)/4.)*pow(t,4) + (18*this->y0 - 18*this->y1 + 3*this->y2)*pow(t,3) + ((9*this->y1)/2. - (21*this->y0)/2.)*pow(t,2) + 3*this->y0*t;
    }
    
    double L_c01_x1(double t)
    {
        return 3*t*pow((t - 1),2);
    }
    
    double K_c01_x2(double t)
    {
        return ((3*this->y0)/2. - (9*this->y1)/2. + (9*this->y2)/2. - (3*this->y3)/2.)*pow(t,6) + ((72*this->y1)/5. - (33*this->y0)/5. - 9*this->y2 + (6*this->y3)/5.)*pow(t,5) + ((45*this->y0)/4. - (63*this->y1)/4. + (9*this->y2)/2.)*pow(t,4) + (6*this->y1 - 9*this->y0)*pow(t,3) + 3*this->y0*pow(t,2);
    }
    
    double L_c01_x2(double t)
    {
        return -3*pow(t,2)*(t - 1);
    }
    
    double K_c01_x3(double t)
    {
        return ((3*this->y1)/2. - this->y0/2. - (3*this->y2)/2. + this->y3/2.)*pow(t,6) + ((9*this->y0)/5. - (18*this->y1)/5. + (9*this->y2)/5.)*pow(t,5) + ((9*this->y1)/4. - (9*this->y0)/4.)*pow(t,4) + this->y0*pow(t,3);
    }
    
    double L_c01_x3(double t)
    {
        return pow(t,3);
    }

    double F_c01_y0(double t)
    {
        return (this->x0/2. - (3*this->x1)/2. + (3*this->x2)/2. - this->x3/2.)*pow(t,6) + ((39*this->x1)/5. - 3*this->x0 - (33*this->x2)/5. + (9*this->x3)/5.)*pow(t,5) + ((15*this->x0)/2. - (33*this->x1)/2. + (45*this->x2)/4. - (9*this->x3)/4.)*pow(t,4) + (18*this->x1 - 10*this->x0 - 9*this->x2 + this->x3)*pow(t,3) + ((15*this->x0)/2. - (21*this->x1)/2. + 3*this->x2)*pow(t,2) + (3*this->x1 - 3*this->x0)*t;
    }
    
    double F_c01_y1(double t)
    {
        return ((9*this->x1)/2. - (3*this->x0)/2. - (9*this->x2)/2. + (3*this->x3)/2.)*pow(t,6) + ((36*this->x0)/5. - 18*this->x1 + (72*this->x2)/5. - (18*this->x3)/5.)*pow(t,5) + (27*this->x1 - (27*this->x0)/2. - (63*this->x2)/4. + (9*this->x3)/4.)*pow(t,4) + (12*this->x0 - 18*this->x1 + 6*this->x2)*pow(t,3) + ((9*this->x1)/2. - (9*this->x0)/2.)*pow(t,2);
    }
    
    double F_c01_y2(double t)
    {
        return ((3*this->x0)/2. - (9*this->x1)/2. + (9*this->x2)/2. - (3*this->x3)/2.)*pow(t,6) + ((63*this->x1)/5. - (27*this->x0)/5. - 9*this->x2 + (9*this->x3)/5.)*pow(t,5) + ((27*this->x0)/4. - (45*this->x1)/4. + (9*this->x2)/2.)*pow(t,4) + (3*this->x1 - 3*this->x0)*pow(t,3);
    }
    
    double F_c01_y3(double t)
    {
        return ((3*this->x1)/2. - this->x0/2. - (3*this->x2)/2. + this->x3/2.)*pow(t,6) + ((6*this->x0)/5. - (12*this->x1)/5. + (6*this->x2)/5.)*pow(t,5) + ((3*this->x1)/4. - (3*this->x0)/4.)*pow(t,4);
    }
    
    double F_c11_x0(double t)
    {
        return (this->y0/2. - (3*this->y1)/2. + (3*this->y2)/2. - this->y3/2.)*pow(t,6) + ((39*this->y1)/5. - 3*this->y0 - (33*this->y2)/5. + (9*this->y3)/5.)*pow(t,5) + ((15*this->y0)/2. - (33*this->y1)/2. + (45*this->y2)/4. - (9*this->y3)/4.)*pow(t,4) + (18*this->y1 - 10*this->y0 - 9*this->y2 + this->y3)*pow(t,3) + ((15*this->y0)/2. - (21*this->y1)/2. + 3*this->y2)*pow(t,2) + (3*this->y1 - 3*this->y0)*t;
    }
    
    double F_c11_x1(double t)
    {
        return ((9*this->y1)/2. - (3*this->y0)/2. - (9*this->y2)/2. + (3*this->y3)/2.)*pow(t,6) + ((36*this->y0)/5. - 18*this->y1 + (72*this->y2)/5. - (18*this->y3)/5.)*pow(t,5) + (27*this->y1 - (27*this->y0)/2. - (63*this->y2)/4. + (9*this->y3)/4.)*pow(t,4) + (12*this->y0 - 18*this->y1 + 6*this->y2)*pow(t,3) + ((9*this->y1)/2. - (9*this->y0)/2.)*pow(t,2);
    }
    
    double F_c11_x2(double t)
    {
        return ((3*this->y0)/2. - (9*this->y1)/2. + (9*this->y2)/2. - (3*this->y3)/2.)*pow(t,6) + ((63*this->y1)/5. - (27*this->y0)/5. - 9*this->y2 + (9*this->y3)/5.)*pow(t,5) + ((27*this->y0)/4. - (45*this->y1)/4. + (9*this->y2)/2.)*pow(t,4) + (3*this->y1 - 3*this->y0)*pow(t,3);
    }
    
    double F_c11_x3(double t)
    {
        return ((3*this->y1)/2. - this->y0/2. - (3*this->y2)/2. + this->y3/2.)*pow(t,6) + ((6*this->y0)/5. - (12*this->y1)/5. + (6*this->y2)/5.)*pow(t,5) + ((3*this->y1)/4. - (3*this->y0)/4.)*pow(t,4);
    }

    double K_c11_y0(double t)
    {
        return (this->x0/2. - (3*this->x1)/2. + (3*this->x2)/2. - this->x3/2.)*pow(t,6) + ((36*this->x1)/5. - 3*this->x0 - (27*this->x2)/5. + (6*this->x3)/5.)*pow(t,5) + ((15*this->x0)/2. - (27*this->x1)/2. + (27*this->x2)/4. - (3*this->x3)/4.)*pow(t,4) + (12*this->x1 - 10*this->x0 - 3*this->x2)*pow(t,3) + ((15*this->x0)/2. - (9*this->x1)/2.)*pow(t,2) - 3*this->x0*t;
    }
    
    double L_c11_y0(double t)
    {
        return -pow((t - 1),3);
    }
    
    double K_c11_y1(double t)
    {
        return ((9*this->x1)/2. - (3*this->x0)/2. - (9*this->x2)/2. + (3*this->x3)/2.)*pow(t,6) + ((39*this->x0)/5. - 18*this->x1 + (63*this->x2)/5. - (12*this->x3)/5.)*pow(t,5) + (27*this->x1 - (33*this->x0)/2. - (45*this->x2)/4. + (3*this->x3)/4.)*pow(t,4) + (18*this->x0 - 18*this->x1 + 3*this->x2)*pow(t,3) + ((9*this->x1)/2. - (21*this->x0)/2.)*pow(t,2) + 3*this->x0*t;
    }
    
    double L_c11_y1(double t)
    {
        return 3*t*pow((t - 1),2);
    }
    
    double K_c11_y2(double t)
    {
        return ((3*this->x0)/2. - (9*this->x1)/2. + (9*this->x2)/2. - (3*this->x3)/2.)*pow(t,6) + ((72*this->x1)/5. - (33*this->x0)/5. - 9*this->x2 + (6*this->x3)/5.)*pow(t,5) + ((45*this->x0)/4. - (63*this->x1)/4. + (9*this->x2)/2.)*pow(t,4) + (6*this->x1 - 9*this->x0)*pow(t,3) + 3*this->x0*pow(t,2);
    }
    
    double L_c11_y2(double t)
    {
        return -3*pow(t,2)*(t - 1);
    }
    
    double K_c11_y3(double t)
    {
        return ((3*this->x1)/2. - this->x0/2. - (3*this->x2)/2. + this->x3/2.)*pow(t,6) + ((9*this->x0)/5. - (18*this->x1)/5. + (9*this->x2)/5.)*pow(t,5) + ((9*this->x1)/4. - (9*this->x0)/4.)*pow(t,4) + this->x0*pow(t,3);
    }
    
    double L_c11_y3(double t)
    {
        return pow(t,3);
    }
    
    double F_c00_x0(double t)
    {
        return (this->y0/2. - (3*this->y1)/2. + (3*this->y2)/2. - this->y3/2.)*pow(t,6) + ((39*this->y1)/5. - 3*this->y0 - (33*this->y2)/5. + (9*this->y3)/5.)*pow(t,5) + ((15*this->y0)/2. - (33*this->y1)/2. + (45*this->y2)/4. - (9*this->y3)/4.)*pow(t,4) + (18*this->y1 - 10*this->y0 - 9*this->y2 + this->y3)*pow(t,3) + ((15*this->y0)/2. - (21*this->y1)/2. + 3*this->y2)*pow(t,2) + (3*this->y1 - 3*this->y0)*t;
    }

    double F_c00_x1(double t)
    {
        return ((9*this->y1)/2. - (3*this->y0)/2. - (9*this->y2)/2. + (3*this->y3)/2.)*pow(t,6) + ((36*this->y0)/5. - 18*this->y1 + (72*this->y2)/5. - (18*this->y3)/5.)*pow(t,5) + (27*this->y1 - (27*this->y0)/2. - (63*this->y2)/4. + (9*this->y3)/4.)*pow(t,4) + (12*this->y0 - 18*this->y1 + 6*this->y2)*pow(t,3) + ((9*this->y1)/2. - (9*this->y0)/2.)*pow(t,2);
    }
    
    double F_c00_x2(double t)
    {
        return ((3*this->y0)/2. - (9*this->y1)/2. + (9*this->y2)/2. - (3*this->y3)/2.)*pow(t,6) + ((63*this->y1)/5. - (27*this->y0)/5. - 9*this->y2 + (9*this->y3)/5.)*pow(t,5) + ((27*this->y0)/4. - (45*this->y1)/4. + (9*this->y2)/2.)*pow(t,4) + (3*this->y1 - 3*this->y0)*pow(t,3);
    }
    
    double F_c00_x3(double t)
    {
        return ((3*this->y1)/2. - this->y0/2. - (3*this->y2)/2. + this->y3/2.)*pow(t,6) + ((6*this->y0)/5. - (12*this->y1)/5. + (6*this->y2)/5.)*pow(t,5) + ((3*this->y1)/4. - (3*this->y0)/4.)*pow(t,4);
    }
    
    double F_c00_y0(double t)
    {
        return (this->x0/2. - (3*this->x1)/2. + (3*this->x2)/2. - this->x3/2.)*pow(t,6) + ((36*this->x1)/5. - 3*this->x0 - (27*this->x2)/5. + (6*this->x3)/5.)*pow(t,5) + ((15*this->x0)/2. - (27*this->x1)/2. + (27*this->x2)/4. - (3*this->x3)/4.)*pow(t,4) + (12*this->x1 - 10*this->x0 - 3*this->x2)*pow(t,3) + ((15*this->x0)/2. - (9*this->x1)/2.)*pow(t,2) - 3*this->x0*t;
    }
    
    double F_c00_y1(double t)
    {
        return ((9*this->x1)/2. - (3*this->x0)/2. - (9*this->x2)/2. + (3*this->x3)/2.)*pow(t,6) + ((39*this->x0)/5. - 18*this->x1 + (63*this->x2)/5. - (12*this->x3)/5.)*pow(t,5) + (27*this->x1 - (33*this->x0)/2. - (45*this->x2)/4. + (3*this->x3)/4.)*pow(t,4) + (18*this->x0 - 18*this->x1 + 3*this->x2)*pow(t,3) + ((9*this->x1)/2. - (21*this->x0)/2.)*pow(t,2) + 3*this->x0*t;
    }
    
    double F_c00_y2(double t)
    {
        return ((3*this->x0)/2. - (9*this->x1)/2. + (9*this->x2)/2. - (3*this->x3)/2.)*pow(t,6) + ((72*this->x1)/5. - (33*this->x0)/5. - 9*this->x2 + (6*this->x3)/5.)*pow(t,5) + ((45*this->x0)/4. - (63*this->x1)/4. + (9*this->x2)/2.)*pow(t,4) + (6*this->x1 - 9*this->x0)*pow(t,3) + 3*this->x0*pow(t,2);
    }
    
    double F_c00_y3(double t)
    {
        return ((3*this->x1)/2. - this->x0/2. - (3*this->x2)/2. + this->x3/2.)*pow(t,6) + ((9*this->x0)/5. - (18*this->x1)/5. + (9*this->x2)/5.)*pow(t,5) + ((9*this->x1)/4. - (9*this->x0)/4.)*pow(t,4) + this->x0*pow(t,3);
    }

    double partial_c00_x0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_x0(t1) - this->F_c00_x0(t0));
    }

    double partial_c00_x1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_x1(t1) - this->F_c00_x1(t0));
    }
    
    double partial_c00_x2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_x2(t1) - this->F_c00_x2(t0));
    }
    
    double partial_c00_x3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_x3(t1) - this->F_c00_x3(t0));
    }

    double partial_c00_y0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_y0(t1) - this->F_c00_y0(t0));
    }
    
    double partial_c00_y1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_y1(t1) - this->F_c00_y1(t0));
    }
    
    double partial_c00_y2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_y2(t1) - this->F_c00_y2(t0));
    }
    
    double partial_c00_y3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * (this->F_c00_y3(t1) - this->F_c00_y3(t0));
    }
    
    double partial_c01_x0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (-pow(2,j) * (this->K_c01_x0(t1) - this->K_c01_x0(t0)) +   (k.y) * (this->L_c01_x0(t1) - this->L_c01_x0(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (-pow(2,j) * (this->K_c01_x0(t1) - this->K_c01_x0(t0)) + (k.y+1) * (this->L_c01_x0(t1) - this->L_c01_x0(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (-pow(2,j) * (this->K_c01_x0(t1) - this->K_c01_x0(t0)) +   (k.y) * (this->L_c01_x0(t1) - this->L_c01_x0(t0)));
        else                            return sign * (-pow(2,j) * (this->K_c01_x0(t1) - this->K_c01_x0(t0)) + (k.y+1) * (this->L_c01_x0(t1) - this->L_c01_x0(t0)));
    }

    double partial_c01_x1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (-pow(2,j) * (this->K_c01_x1(t1) - this->K_c01_x1(t0)) +   (k.y) * (this->L_c01_x1(t1) - this->L_c01_x1(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (-pow(2,j) * (this->K_c01_x1(t1) - this->K_c01_x1(t0)) + (k.y+1) * (this->L_c01_x1(t1) - this->L_c01_x1(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (-pow(2,j) * (this->K_c01_x1(t1) - this->K_c01_x1(t0)) +   (k.y) * (this->L_c01_x1(t1) - this->L_c01_x1(t0)));
        else                            return sign * (-pow(2,j) * (this->K_c01_x1(t1) - this->K_c01_x1(t0)) + (k.y+1) * (this->L_c01_x1(t1) - this->L_c01_x1(t0)));
    }

    double partial_c01_x2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (-pow(2,j) * (this->K_c01_x2(t1) - this->K_c01_x2(t0)) +   (k.y) * (this->L_c01_x2(t1) - this->L_c01_x2(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (-pow(2,j) * (this->K_c01_x2(t1) - this->K_c01_x2(t0)) + (k.y+1) * (this->L_c01_x2(t1) - this->L_c01_x2(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (-pow(2,j) * (this->K_c01_x2(t1) - this->K_c01_x2(t0)) +   (k.y) * (this->L_c01_x2(t1) - this->L_c01_x2(t0)));
        else                            return sign * (-pow(2,j) * (this->K_c01_x2(t1) - this->K_c01_x2(t0)) + (k.y+1) * (this->L_c01_x2(t1) - this->L_c01_x2(t0)));
    }
    
    double partial_c01_x3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (-pow(2,j) * (this->K_c01_x3(t1) - this->K_c01_x3(t0)) +   (k.y) * (this->L_c01_x3(t1) - this->L_c01_x3(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (-pow(2,j) * (this->K_c01_x3(t1) - this->K_c01_x3(t0)) + (k.y+1) * (this->L_c01_x3(t1) - this->L_c01_x3(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (-pow(2,j) * (this->K_c01_x3(t1) - this->K_c01_x3(t0)) +   (k.y) * (this->L_c01_x3(t1) - this->L_c01_x3(t0)));
        else                            return sign * (-pow(2,j) * (this->K_c01_x3(t1) - this->K_c01_x3(t0)) + (k.y+1) * (this->L_c01_x3(t1) - this->L_c01_x3(t0)));
    }
    
    double partial_c01_y0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return -(sign * pow(2,j) * (this->F_c01_y0(t1) - this->F_c01_y0(t0)));
    }
    
    double partial_c01_y1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return -(sign * pow(2,j) * (this->F_c01_y1(t1) - this->F_c01_y1(t0)));
    }
    
    double partial_c01_y2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return -(sign * pow(2,j) * (this->F_c01_y2(t1) - this->F_c01_y2(t0)));
    }
    
    double partial_c01_y3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return -(sign * pow(2,j) * (this->F_c01_y3(t1) - this->F_c01_y3(t0)));
    }

    double partial_c10_x0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c10_x0(t1) - this->F_c10_x0(t0));
    }
    
    double partial_c10_x1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c10_x1(t1) - this->F_c10_x1(t0));
    }
    
    double partial_c10_x2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c10_x2(t1) - this->F_c10_x2(t0));
    }
    
    double partial_c10_x3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c10_x3(t1) - this->F_c10_x3(t0));
    }
    
    double partial_c10_y0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c10_y0(t1) - this->K_c10_y0(t0)) -   (k.x) * (this->L_c10_y0(t1) - this->L_c10_y0(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c10_y0(t1) - this->K_c10_y0(t0)) -   (k.x) * (this->L_c10_y0(t1) - this->L_c10_y0(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c10_y0(t1) - this->K_c10_y0(t0)) - (k.x+1) * (this->L_c10_y0(t1) - this->L_c10_y0(t0)));
        else                            return sign * (pow(2,j) * (this->K_c10_y0(t1) - this->K_c10_y0(t0)) - (k.x+1) * (this->L_c10_y0(t1) - this->L_c10_y0(t0)));
    }
    
    double partial_c10_y1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c10_y1(t1) - this->K_c10_y1(t0)) -   (k.x) * (this->L_c10_y1(t1) - this->L_c10_y1(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c10_y1(t1) - this->K_c10_y1(t0)) -   (k.x) * (this->L_c10_y1(t1) - this->L_c10_y1(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c10_y1(t1) - this->K_c10_y1(t0)) - (k.x+1) * (this->L_c10_y1(t1) - this->L_c10_y1(t0)));
        else                            return sign * (pow(2,j) * (this->K_c10_y1(t1) - this->K_c10_y1(t0)) - (k.x+1) * (this->L_c10_y1(t1) - this->L_c10_y1(t0)));
    }
    
    double partial_c10_y2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c10_y2(t1) - this->K_c10_y2(t0)) -   (k.x) * (this->L_c10_y2(t1) - this->L_c10_y2(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c10_y2(t1) - this->K_c10_y2(t0)) -   (k.x) * (this->L_c10_y2(t1) - this->L_c10_y2(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c10_y2(t1) - this->K_c10_y2(t0)) - (k.x+1) * (this->L_c10_y2(t1) - this->L_c10_y2(t0)));
        else                            return sign * (pow(2,j) * (this->K_c10_y2(t1) - this->K_c10_y2(t0)) - (k.x+1) * (this->L_c10_y2(t1) - this->L_c10_y2(t0)));
    }
    
    double partial_c10_y3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c10_y3(t1) - this->K_c10_y3(t0)) -   (k.x) * (this->L_c10_y3(t1) - this->L_c10_y3(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c10_y3(t1) - this->K_c10_y3(t0)) -   (k.x) * (this->L_c10_y3(t1) - this->L_c10_y3(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c10_y3(t1) - this->K_c10_y3(t0)) - (k.x+1) * (this->L_c10_y3(t1) - this->L_c10_y3(t0)));
        else                            return sign * (pow(2,j) * (this->K_c10_y3(t1) - this->K_c10_y3(t0)) - (k.x+1) * (this->L_c10_y3(t1) - this->L_c10_y3(t0)));
    }

    double partial_c11_x0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c11_x0(t1) - this->F_c11_x0(t0));
    }
    
    double partial_c11_x1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c11_x1(t1) - this->F_c11_x1(t0));
    }
    
    double partial_c11_x2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c11_x2(t1) - this->F_c11_x2(t0));
    }
    
    double partial_c11_x3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        return sign * pow(2,j) * (this->F_c11_x3(t1) - this->F_c11_x3(t0));
    }
    
    double partial_c11_y0(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c11_y0(t1) - this->K_c11_y0(t0)) -   (k.x) * (this->L_c11_y0(t1) - this->L_c11_y0(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c11_y0(t1) - this->K_c11_y0(t0)) -   (k.x) * (this->L_c11_y0(t1) - this->L_c11_y0(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c11_y0(t1) - this->K_c11_y0(t0)) - (k.x+1) * (this->L_c11_y0(t1) - this->L_c11_y0(t0)));
        else                            return sign * (pow(2,j) * (this->K_c11_y0(t1) - this->K_c11_y0(t0)) - (k.x+1) * (this->L_c11_y0(t1) - this->L_c11_y0(t0)));
    }
    
    double partial_c11_y1(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c11_y1(t1) - this->K_c11_y1(t0)) -   (k.x) * (this->L_c11_y1(t1) - this->L_c11_y1(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c11_y1(t1) - this->K_c11_y1(t0)) -   (k.x) * (this->L_c11_y1(t1) - this->L_c11_y1(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c11_y1(t1) - this->K_c11_y1(t0)) - (k.x+1) * (this->L_c11_y1(t1) - this->L_c11_y1(t0)));
        else                            return sign * (pow(2,j) * (this->K_c11_y1(t1) - this->K_c11_y1(t0)) - (k.x+1) * (this->L_c11_y1(t1) - this->L_c11_y1(t0)));
    }
    
    double partial_c11_y2(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c11_y2(t1) - this->K_c11_y2(t0)) -   (k.x) * (this->L_c11_y2(t1) - this->L_c11_y2(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c11_y2(t1) - this->K_c11_y2(t0)) -   (k.x) * (this->L_c11_y2(t1) - this->L_c11_y2(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c11_y2(t1) - this->K_c11_y2(t0)) - (k.x+1) * (this->L_c11_y2(t1) - this->L_c11_y2(t0)));
        else                            return sign * (pow(2,j) * (this->K_c11_y2(t1) - this->K_c11_y2(t0)) - (k.x+1) * (this->L_c11_y2(t1) - this->L_c11_y2(t0)));
    }
    
    double partial_c11_y3(const Point& k, int j, double t0, double t1, const Point& q, double sign)
    {
        if (q.x == 0 && q.y == 0)       return sign * (pow(2,j) * (this->K_c11_y3(t1) - this->K_c11_y3(t0)) -   (k.x) * (this->L_c11_y3(t1) - this->L_c11_y3(t0)));
        else if (q.x == 0 && q.y == 1)  return sign * (pow(2,j) * (this->K_c11_y3(t1) - this->K_c11_y3(t0)) -   (k.x) * (this->L_c11_y3(t1) - this->L_c11_y3(t0)));
        else if (q.x == 1 && q.y == 0)  return sign * (pow(2,j) * (this->K_c11_y3(t1) - this->K_c11_y3(t0)) - (k.x+1) * (this->L_c11_y3(t1) - this->L_c11_y3(t0)));
        else                            return sign * (pow(2,j) * (this->K_c11_y3(t1) - this->K_c11_y3(t0)) - (k.x+1) * (this->L_c11_y3(t1) - this->L_c11_y3(t0)));
    }

    static double cpsi_1d(double x)
    {
        if (x < 0)
            return 0.0;
        else if (0 <= x && x < 0.5)
            return double(x);
        else if (0.5 <= x && x < 1)
            return 1.0 - double(x);
        else
            return 0.0;
    }

    static double cpsi_1d_jk(double x, int j, double k)
    {
        return pow(2, -j) * cpsi_1d(pow(2, j) * x - k);
    }

    static double dF_dp(int i, double t)
    {
        if (i == 0)
            return pow((1.0 - t), 3);
        else if (i == 1)
            return 3.0 * pow((1.0 - t), 2) * t;
        else if (i == 2)
            return 3.0 * (1.0 - t) * pow(t, 2);
        else
            return pow(t, 3);
    }

    static double F(double t, const tuple<double, double, double, double>& para)
    {
        double result = 0.0;

        result += dF_dp(0, t) * get<0>(para);
        result += dF_dp(1, t) * get<1>(para);
        result += dF_dp(2, t) * get<2>(para);
        result += dF_dp(3, t) * get<3>(para);

        return result;
    }

    static double dFp_dp(int i, double t)
    {
        if (i == 0)
            return -3.0 * pow((1.0 - t), 2);
        else if (i == 1)
            return 3.0 * (-2.0 * (1.0 - t) * t + pow((1.0 - t), 2));
        else if (i == 2)
            return 3.0 * (-pow(t, 2) + (1.0 - t) * 2 * t);
        else
            return 3.0 * pow(t, 2);
    }

    static double Fp(double t, const tuple<double, double, double, double>& para)
    {
        double result = 0.0;

        result += dFp_dp(0, t) * get<0>(para);
        result += dFp_dp(1, t) * get<1>(para);
        result += dFp_dp(2, t) * get<2>(para);
        result += dFp_dp(3, t) * get<3>(para);

        return result;
    }

    static tuple<double, double, double, double> cubicCoefficients(double p0, double p1, double p2, double p3)
    {
        auto a = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
        auto b = 3.0 * p0 - 6.0 * p1 + 3.0 * p2;
        auto c = -3.0 * p0 + 3.0 * p1;
        auto d = p0;

        return make_tuple(a, b, c, d);
    }

    double X(double t)
    {
        return F(t, make_tuple(this->x0, this->x1, this->x2, this->x3));
    }

    double Xp(double t)
    {
        return Fp(t, make_tuple(this->x0, this->x1, this->x2, this->x3));
    }

    double Y(double t)
    {
        return F(t, make_tuple(this->y0, this->y1, this->y2, this->y3));
    }

    double Yp(double t)
    {
        return Fp(t, make_tuple(this->y0, this->y1, this->y2, this->y3));
    }

    double impulse01_x(int i, int j, const Point& k)
    {
        auto eps = 1e-8;
        auto kx = k.x;
        auto ky = k.y;

        auto helper = [this, eps, i, j ,ky](double t) -> double
        {
            if (this->Xp(t) >= -eps)  // >= 0
                return cpsi_1d_jk(this->Y(t), j, ky) * dF_dp(i, t);
            else // < 0
                return -cpsi_1d_jk(this->Y(t), j, ky) * dF_dp(i, t);
        };

        auto result = 0.0;
        auto __tmp0 = cubicCoefficients(this->x0, this->x1, this->x2, this->x3);
        auto a = get<0>(__tmp0);
        auto b = get<1>(__tmp0);
        auto c = get<2>(__tmp0);
        auto d = get<3>(__tmp0);

        // t0
        auto roots = cubic(a, b, c, d - double(kx) / pow(2, j));
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result -= helper(root) * pow(2, j);

        // t1
        roots = cubic(a, b, c, d - (1.0 + kx) / pow(2, j));
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result += helper(root) * pow(2, j);

        return result;
    }

    double impulse00_y(int i)
    {
        auto eps = 1e-8;

        auto helper = [this, eps, i](double t) -> double
        {
            if (this->Yp(t) >= -eps)  // >= 0
                return cpsi_1d(this->X(t)) * dF_dp(i, t);
            else  // < 0
                return -cpsi_1d(this->X(t)) * dF_dp(i, t);
        };

        auto result = 0.0;
        auto __tmp0 = cubicCoefficients(this->y0, this->y1, this->y2, this->y3);
        auto a = get<0>(__tmp0);
        auto b = get<1>(__tmp0);
        auto c = get<2>(__tmp0);
        auto d = get<3>(__tmp0);

        // t0
        auto roots = cubic(a, b, c, d);
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result += helper(root);

        // t1
        roots = cubic(a, b, c, d - 1.0);
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result -= helper(root);

        return result;
    }

    double impulse10_y(int i, int j, const Point& k)
    {
        auto eps = 1e-8;
        auto kx = k.x;
        auto ky = k.y;

        auto helper = [this, eps, i, j, kx](double t) -> double
        {
            if (this->Yp(t) >= -eps)  // >= 0
                return cpsi_1d_jk(this->X(t), j, kx) * dF_dp(i, t);
            else // < 0
                return -cpsi_1d_jk(this->X(t), j, kx) * dF_dp(i, t);
        };

        auto result = 0.0;
        auto __tmp0 = cubicCoefficients(this->y0, this->y1, this->y2, this->y3);
        auto a = get<0>(__tmp0);
        auto b = get<1>(__tmp0);
        auto c = get<2>(__tmp0);
        auto d = get<3>(__tmp0);

        // t0
        auto roots = cubic(a, b, c, d - double(ky) / pow(2, j));
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result += helper(root) * pow(2, j);

        // t1
        roots = cubic(a, b, c, d - (1.0 + ky) / pow(2, j));
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result -= helper(root) * pow(2, j);

        return result;
    }

    double impulse11_y(int i, int j, const Point& k)
    {
        auto eps = 1e-8;
        auto kx = k.x;
        auto ky = k.y;

        auto helper = [this, eps, i, j, kx](double t) -> double
        {
            if (this->Yp(t) >= -eps)  // >= 0
                return cpsi_1d_jk(this->X(t), j, kx) * dF_dp(i, t);
            else  // < 0
                return -cpsi_1d_jk(this->X(t), j, kx) * dF_dp(i, t);
        };

        auto result = 0.0;
        auto __tmp0 = cubicCoefficients(this->y0, this->y1, this->y2, this->y3);
        auto a = get<0>(__tmp0);
        auto b = get<1>(__tmp0);
        auto c = get<2>(__tmp0);
        auto d = get<3>(__tmp0);

        auto roots = cubic(a, b, c, d - double(ky) / pow(2, j));
        for (auto root : roots)
            if (eps <= root && root <= 1 + eps)
                result += helper(root) * pow(2, j);
    
        // t1
        roots = cubic(a, b, c, d - (0.5 + ky) / pow(2, j));
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result -= helper(root) * pow(2, (j + 1));

        // t2
        roots = cubic(a, b, c, d - (1.0 + ky) / pow(2, j));
        for (auto root : roots)
            if (-eps <= root && root <= 1 + eps)
                result += helper(root) * pow(2, j);

        return result;
    }

    // manual
    List<double> get_grads(const List<double>& section, int ei, const Point& k, int j, const Point& q, double sign, double left, double right, double bottom, double top)
    {
        typedef double (CubicBezier::* MethodType)(const Point&, int, double, double, const Point&, double);

        vector<vector<MethodType>> partial_matrix;

        MethodType tmp0[] = {&CubicBezier::partial_c00_x0,  &CubicBezier::partial_c01_x0,  &CubicBezier::partial_c10_x0,  &CubicBezier::partial_c11_x0};
        partial_matrix.push_back(vector<MethodType>(tmp0, tmp0 + 4));

        MethodType tmp1[] = {&CubicBezier::partial_c00_y0,  &CubicBezier::partial_c01_y0,  &CubicBezier::partial_c10_y0,  &CubicBezier::partial_c11_y0};
        partial_matrix.push_back(vector<MethodType>(tmp1, tmp1 + 4));

        MethodType tmp2[] = {&CubicBezier::partial_c00_x1,  &CubicBezier::partial_c01_x1,  &CubicBezier::partial_c10_x1,  &CubicBezier::partial_c11_x1};
        partial_matrix.push_back(vector<MethodType>(tmp2, tmp2 + 4));

        MethodType tmp3[] = {&CubicBezier::partial_c00_y1,  &CubicBezier::partial_c01_y1,  &CubicBezier::partial_c10_y1,  &CubicBezier::partial_c11_y1};
        partial_matrix.push_back(vector<MethodType>(tmp3, tmp3 + 4));

        MethodType tmp4[] = {&CubicBezier::partial_c00_x2,  &CubicBezier::partial_c01_x2,  &CubicBezier::partial_c10_x2,  &CubicBezier::partial_c11_x2};
        partial_matrix.push_back(vector<MethodType>(tmp4, tmp4 + 4));

        MethodType tmp5[] = {&CubicBezier::partial_c00_y2,  &CubicBezier::partial_c01_y2,  &CubicBezier::partial_c10_y2,  &CubicBezier::partial_c11_y2};
        partial_matrix.push_back(vector<MethodType>(tmp5, tmp5 + 4));

        MethodType tmp6[] = {&CubicBezier::partial_c00_x3,  &CubicBezier::partial_c01_x3,  &CubicBezier::partial_c10_x3,  &CubicBezier::partial_c11_x3};
        partial_matrix.push_back(vector<MethodType>(tmp6, tmp6 + 4));

        MethodType tmp7[] = {&CubicBezier::partial_c00_y3,  &CubicBezier::partial_c01_y3,  &CubicBezier::partial_c10_y3,  &CubicBezier::partial_c11_y3};
        partial_matrix.push_back(vector<MethodType>(tmp7, tmp7 + 4));

        List<double> grads(8, 0.0);

        for (auto __item : this->clipTrue(left, right, bottom, top))
        {
            auto t0 = __item.x;
            auto t1 = __item.y;

            for (int i = 0; i < 8; i++)
            {
                auto method = partial_matrix[i][ei];
                grads[i] += (this->*method)(k, j, t0, t1, q, sign);
            }
        }

        return grads;
    }
    //

    List<double> get_impulses(const List<double>& section, int ei, const Point& k, int j)
    {
        List<double> impulses(8, 0);

        if (ei == 0)
            for (int i = 1; i < 8; i += 2)
                impulses[i] += this->impulse00_y(i / 2);
        else if (ei == 1)
            for (int i = 0; i < 8; i += 2)
                impulses[i] += this->impulse01_x(i / 2, j, k);
        else if (ei == 2)
            for (int i = 1; i < 8; i += 2)
                impulses[i] += this->impulse10_y(i / 2, j, k);
        else
            for (int i = 1; i < 8 ; i += 2)
                impulses[i] += this->impulse11_y(i / 2, j, k);

        return impulses;
    }

private:
    double x0, y0, x1, y1, x2, y2, x3, y3;
};


class Contour
{
public:
    Contour(const List<Point>& contour) :
        contour(contour)
    {
    }

    void process(function<Point(Point)> f)
    {
        this->contour = Map(this->contour, f);
    }

    double area()
    {
        auto det = [](const Point& a, const Point& b)
        {
            return a.x * b.y - a.y * b.x;
        };
        auto s = 0.0;
        for (auto __item : this->each())
        {
            auto v0 = get<0>(__item);
            auto v1 = get<1>(__item);
            auto v2 = get<2>(__item);
            auto v3 = get<3>(__item);

            s += 3./10 * det(v0,v1) + 3./20 * det(v1,v2) + 3./10 * det(v2,v3)
                + 3./20 * det(v0,v2) + 3./20 * det(v1,v3) + 1./20 * det(v0,v3);
        }
        return s;
    }

    // manual
    List<tuple<Point, Point, Point, Point>> each()
    {
        List<tuple<Point, Point, Point, Point>> result;

        for (int i = 0; i < this->contour.size(); i += 3)
        {
            auto v3 = this->contour[i];
            auto v2 = this->contour[(i-1 + this->contour.size()) % this->contour.size()];
            auto v1 = this->contour[(i-2 + this->contour.size()) % this->contour.size()];
            auto v0 = this->contour[(i-3 + this->contour.size()) % this->contour.size()];

            result.push_back(make_tuple(v0, v1, v2, v3));
        }

        return result;
    }
    //

    tuple<Point, Point> get_KL(const List<double>& section)
    {
        CubicBezier bezier(section[0], section[4], section[1], section[5], section[2], section[6], section[3], section[7]);
        return bezier.get_KL();
    }

    // manual
    List<tuple<tuple<Point, Point, Point, Point>, tuple<int, int, int, int>>> each_with_indice()
    {
        List<tuple<tuple<Point, Point, Point, Point>, tuple<int, int, int, int>>> result;

        for (int i = 0; i < this->contour.size(); i += 3)
        {
            auto v3 = this->contour[i];
            auto v2 = this->contour[(i-1 + this->contour.size()) % this->contour.size()];
            auto v1 = this->contour[(i-2 + this->contour.size()) % this->contour.size()];
            auto v0 = this->contour[(i-3 + this->contour.size()) % this->contour.size()];

            result.push_back(make_tuple(make_tuple(v0, v1, v2, v3), make_tuple(i-3, i-2, i-1, i)));
        }

        return result;
    }
    //

    List<double> get_grads(const List<double>& section, int ei, const Point& k, int j, const Point& q, double sign, double left, double right, double bottom, double top)
    {
        CubicBezier bezier(section[0], section[4], section[1], section[5], section[2], section[6], section[3], section[7]);
        return bezier.get_grads(section, ei, k, j, q, sign, left, right, bottom, top);
    }

    List<double> get_impulses(const List<double>& section, int ei, const Point& k, int j)
    {
        CubicBezier bezier(section[0], section[4], section[1], section[5], section[2], section[6], section[3], section[7]);
        return bezier.get_impulses(section, ei, k, j);
    }

    List<Point> contour;
};