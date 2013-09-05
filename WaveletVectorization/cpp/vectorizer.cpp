#include <cmath>
#include <functional>
#include <tuple>
#include <unordered_map>
#include "CubicBezier.hpp"
using namespace std;

// manual
template<> 
struct hash<tuple<int, int, int>>
{
    template<typename T>
    static void hash_combine(size_t& seed, const T& v)
    {
        seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    size_t operator()(const tuple<int, int, int>& x) const
    {
        size_t seed = 0;
        hash_combine(seed, get<0>(x));
        hash_combine(seed, get<1>(x));
        hash_combine(seed, get<2>(x));

        return seed;
    }
};
//

class Rasterizer
{
public:
    Rasterizer(const Contour& contour, double w, double h)
        : contour(contour), w(w), h(h)
    {
        this->max_j = int(ceil(log(max(w, h)) / log(2))) - 1;
        this->wh = pow(2, (this->max_j + 1));

        auto normalize = [this](const Point& p) -> Point
        {
            return Point(p.x / double(this->wh), p.y / double(this->wh));
        };

        this->contour.process(normalize);
        this->area = this->contour.area();
        for (int x = 0; x < h; x++)
            for (int y = 0; y < w; y++)
                this->lattice.push_back(normalize(Point(x, y)));
        
        //manual
        // prepare all c
        for (int j = 0; j <= this->max_j; j++)
            for (int kx = 0; kx <= (1 << j); kx++)
                for (int ky = 0; ky <= (1 << j); ky++)
                    this->all_c[make_tuple(j, kx, ky)] = this->c(j, Point(kx, ky));
        //
    }

    static double psi(const Point& p, const Point& e, int j, const Point& k)
    {
        auto psi_1d = [](double p, int e) -> double
        {
            if (e == 0)
                return 0 <= p && p < 1 ? 1.0 : 0.0;
            else
                return 0 <= p && p < 1 ? (0 <= p && p < 0.5 ? 1.0 : -1.0) : 0.0;
        };

        return pow(2, j) * psi_1d(pow(2, j) * p.x - k.x, e.x) * psi_1d(pow(2, j) * p.y - k.y, e.y);
    }

    tuple<double, double, double> c(double j, const Point& k)
    {
        auto transform = [j, k](const tuple<Point, Point, Point, Point>& section, const Point& Q) -> List<double>
        {
            List<double> result;

            result.push_back(pow(2, (j + 1)) * (std::get<0>(section)).x - k.x * 2 - Q.x);
            result.push_back(pow(2, (j + 1)) * (std::get<1>(section)).x - k.x * 2 - Q.x);
            result.push_back(pow(2, (j + 1)) * (std::get<2>(section)).x - k.x * 2 - Q.x);
            result.push_back(pow(2, (j + 1)) * (std::get<3>(section)).x - k.x * 2 - Q.x);
            result.push_back(pow(2, (j + 1)) * (std::get<0>(section)).y - k.y * 2 - Q.y);
            result.push_back(pow(2, (j + 1)) * (std::get<1>(section)).y - k.y * 2 - Q.y);
            result.push_back(pow(2, (j + 1)) * (std::get<2>(section)).y - k.y * 2 - Q.y);
            result.push_back(pow(2, (j + 1)) * (std::get<3>(section)).y - k.y * 2 - Q.y);

            return result;
        };

        Point Q_00(0, 0);
        Point Q_01(0, 1);
        Point Q_10(1, 0);
        Point Q_11(1, 1);
        auto c10 = 0.0, c01 = 0.0, c11 = 0.0;
        for (auto section : this->contour.each())
        {
            auto __tmp0 = this->contour.get_KL(transform(section, Q_00));
            auto KQ00 = std::get<0>(__tmp0);
            auto LQ00 = std::get<1>(__tmp0);
            auto __tmp1 = this->contour.get_KL(transform(section, Q_01));
            auto KQ01 = std::get<0>(__tmp1);
            auto LQ01 = std::get<1>(__tmp1);
            auto __tmp2 = this->contour.get_KL(transform(section, Q_10));
            auto KQ10 = std::get<0>(__tmp2);
            auto LQ10 = std::get<1>(__tmp2);
            auto __tmp3 = this->contour.get_KL(transform(section, Q_11));
            auto KQ11 = std::get<0>(__tmp3);
            auto LQ11 = std::get<1>(__tmp3);
            c10 += LQ00.x + LQ01.x + KQ10.x 
                - LQ10.x + KQ11.x - LQ11.x;
            c01 += LQ00.y + LQ10.y + KQ01.y
                - LQ01.y + KQ11.y - LQ11.y;
            c11 += LQ00.x - LQ01.x + KQ10.x
                - LQ10.x - KQ11.x + LQ11.x;
        }

        return make_tuple(c01, c10, c11);
    }

    double g(const Point& p)
    {
        auto s = this->area;
        Point tmp[] = {Point(0, 1), Point(1, 0), Point(1, 1)};
        List<Point> E(tmp, tmp + 3);

        for (int j = 0; j <= this->max_j; j++)
            for (int kx = 0; kx < pow(2, j); kx++)
                for (int ky = 0; ky < pow(2, j); ky++)
                {
                    Point k(kx, ky);
                    auto cs = this->all_c[make_tuple(j, kx, ky)];

                    auto e = E[0];
                    auto psi = this->psi(p, e, j, k);
                    if (psi > 0)
                        s += std::get<0>(cs);
                    else if (psi < 0)
                        s -= std::get<0>(cs);
                    
                    e = E[1];
                    psi = this->psi(p, e, j, k);
                    if (psi > 0)
                        s += std::get<1>(cs);
                    else if (psi < 0)
                        s -= std::get<1>(cs);

                    e = E[2];
                    psi = this->psi(p, e, j, k);
                    if (psi > 0)
                        s += std::get<2>(cs);
                    else if (psi < 0)
                        s -= std::get<2>(cs);
                }

        return s;
    }

    List<List<double>> get()
    {
        List<double> px_arr;
        for (auto p : this->lattice)
            px_arr.push_back(this->g(p));
        List<List<double>> px_mat;
        for (int i = 0; i < this->h; i++)
        {
            List<double> __tmp0;
            for (int j = i * this->w; j < (i + 1) * this->w; j++)
                __tmp0.push_back(px_arr[j]);
            px_mat.push_back(__tmp0);
        }
        return px_mat;
    }

private:
    int w, h, max_j;
    double wh, area;
    Contour contour;
    List<Point> lattice;
    unordered_map<tuple<int, int, int>, tuple<double, double, double>, 
                  hash<tuple<int, int, int>>> all_c;
};

int main()
{
    double tmp[] = {1, 1, 3, 1, 7, 3, 7, 7, 3, 7, 1, 3};
    List<double> oriPath(tmp, tmp + sizeof(tmp) / sizeof(double));

    auto convertPathToContour = [](const List<double>& path) -> Contour
    {
        List<Point> pts;
        for (int i = 0; i < path.size(); i += 2)
            pts.push_back(Point(path[i], path[i + 1]));
        return Contour(pts);
    };

    auto oriRaster = Rasterizer(convertPathToContour(oriPath), 8, 8).get();
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
            printf("%f ", oriRaster[i][j]);
        printf("\n");
    }
}