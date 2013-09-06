//////////////////////////////////////////////////////////////////
// Auto-Generated                                               //
//                                                              //
// This code was auto-generated by Artanis's Jupiter Project.   //
// You bear the risk of using it. No warranties, guarantees or  //
// conditions are provided.                                     //
//                                                              //
// Jupiter Version: 0.2.1                                       //
//////////////////////////////////////////////////////////////////

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
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
    // manual
        : w(w), h(h), contour(contour)
    //
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
            for (int kx = 0; kx < (1 << j); kx++)
                for (int ky = 0; ky < (1 << j); ky++)
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

        return pow(2, j) * psi_1d(pow(2, j) * p.x - k.x, (int)e.x) * psi_1d(pow(2, j) * p.y - k.y, (int)e.y);
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
        List<Point> E = {Point(0, 1), Point(1, 0), Point(1, 1)};

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

    // manual
    List<List<double>> get()
    {
        List<double> px_arr;
        for (auto p : this->lattice)
            px_arr.push_back(this->g(p));
        List<List<double>> px_mat;
        for (int i = 0; i < this->h; i++)
            px_mat.push_back(List<double>(px_arr.begin() + i * this->w, px_arr.begin() + (i + 1) * this->w));
        return px_mat;
    }
    //

private:
    int w, h, max_j;
    double wh, area;
    Contour contour;
    List<Point> lattice;
    unordered_map<tuple<int, int, int>, tuple<double, double, double>, 
                  hash<tuple<int, int, int>>> all_c;
};

// manual
template<typename T>
void addmul(List<T>& A, const List<T>& B, double mul = 1)
{
    for (int i = 0; i < A.size(); i++)
        A[i] += mul * B[i];
}
//

class Vectorizer
{
public:
    Vectorizer(const Contour& contour, const List<List<double>>& org_img) :
    // manual
        contour(contour)
    //
    {
        // manual
        int h = org_img.size(), w = org_img[0].size();
        //

        this->w = w;
        this->h = h;
        this->org_img = org_img;
        this->max_j = int(ceil(log(max(w, h)) / log(2))) - 1;
        this->wh = pow(2, (this->max_j + 1));
        this->num = (this->contour).contour.size() * 2;
        for (int x = 0; x < h; x++)
            for (int y = 0; y < w; y++)
                this->lattice.push_back(Point(x, y));

        //manual
        // prepare all dc_dX
        for (int j = 0; j <= this->max_j; j++)
            for (int kx = 0; kx < (1 << j); kx++)
                for (int ky = 0; ky < (1 << j); ky++)
                {
                    auto dc_dx1 = this->dc_dX(j, Point(kx, ky), 1);
                    auto dc_dx2 = this->dc_dX(j, Point(kx, ky), 2);
                    auto dc_dx3 = this->dc_dX(j, Point(kx, ky), 3);
                    this->all_dc[make_tuple(j, kx, ky)] = make_tuple(dc_dx1, dc_dx2, dc_dx3);
                }
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

        return pow(2, j) * psi_1d(pow(2, j) * p.x - k.x, (int)e.x) * psi_1d(pow(2, j) * p.y - k.y, (int)e.y);
    }

    List<double> dc_dX(int j, const Point& k, int ei)
    {
        auto normalize = [this](double p)
        {
            return p / double(this->wh);
        };

        List<Point> Q = {Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1)};
        List<List<double>> sign = {{1, 1, 1, 1}, {1, -1, 1, -1}, {+1, +1, -1, -1}, {+1, -1, -1, +1}};
        List<double> grads(this->num, 0);

        for (auto __item : this->contour.each_with_indice())
        {
            auto section = get<0>(__item);
            auto indice = get<1>(__item);

            // manual
            List<double> __section;
            __section.push_back(normalize(get<0>(section).x));
            __section.push_back(normalize(get<1>(section).x));
            __section.push_back(normalize(get<2>(section).x));
            __section.push_back(normalize(get<3>(section).x));
            __section.push_back(normalize(get<0>(section).y));
            __section.push_back(normalize(get<1>(section).y));
            __section.push_back(normalize(get<2>(section).y));
            __section.push_back(normalize(get<3>(section).y));

            for (int i = 0; i < Q.size(); i++)
            //
            {
                // manual
                Point q = Q[i];
                double left = (k.x + q.x * 0.5) / pow(2, j);
                double right = (k.x + q.x * 0.5 + 0.5) / pow(2, j);
                double bottom = (k.y + q.y * 0.5 + 0.5) / pow(2, j);
                double top = (k.y + q.y * 0.5) / pow(2, j);

                auto sec_grads = this->contour.get_grads(__section, ei, k, j, q,
                    sign[ei][i], left, right, bottom, top);
                //

                grads[(this->num + get<0>(indice) * 2) % this->num] += sec_grads[0];
                grads[(this->num + get<0>(indice) * 2 + 1) % this->num] += sec_grads[1];
                grads[(this->num + get<1>(indice) * 2) % this->num] += sec_grads[2];
                grads[(this->num + get<1>(indice) * 2 + 1) % this->num] += sec_grads[3];
                grads[(this->num + get<2>(indice) * 2) % this->num] += sec_grads[4];
                grads[(this->num + get<2>(indice) * 2 + 1) % this->num] += sec_grads[5];
                grads[(this->num + get<3>(indice) * 2) % this->num] += sec_grads[6];
                grads[(this->num + get<3>(indice) * 2 + 1) % this->num] += sec_grads[7];
            }

            // manual
            auto sec_impulses = this->contour.get_impulses(__section, ei, k, j);
            //
            grads[(this->num + get<0>(indice) * 2) % this->num] += sec_impulses[0];
            grads[(this->num + get<0>(indice) * 2 + 1) % this->num] += sec_impulses[1];
            grads[(this->num + get<1>(indice) * 2) % this->num] += sec_impulses[2];
            grads[(this->num + get<1>(indice) * 2 + 1) % this->num] += sec_impulses[3];
            grads[(this->num + get<2>(indice) * 2) % this->num] += sec_impulses[4];
            grads[(this->num + get<2>(indice) * 2 + 1) % this->num] += sec_impulses[5];
            grads[(this->num + get<3>(indice) * 2) % this->num] += sec_impulses[6];
            grads[(this->num + get<3>(indice) * 2 + 1) % this->num] += sec_impulses[7];
        }

        for (int i = 0; i < this->num; i++)
            grads[i] /= double(this->wh);

        return grads;
    }

    List<double> dLike_dX()
    {
        List<Point> E = {Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1)};
        auto raster = Rasterizer(this->contour, this->w, this->h).get();
        List<double> grads(this->num, 0);

        for (auto __item : this->lattice)
        {
            // manual
            int x = (int)__item.x;
            int y = (int)__item.y;
            //
            Point p(x / double(this->wh), y / double(this->wh));

            // dR/dX
            List<double> grads_R(this->num, 0);
            Point k0(0, 0);
            addmul(grads_R, this->dc_dX(0, k0, 0), this->psi(p, E[0], 0, k0));
            for (int j = 0; j <= max_j; j++)
                for (int kx = 0; kx < pow(2, j); kx++)
                    for (int ky = 0; ky < pow(2, j); ky++)
                    {
                        Point k(kx, ky);
                        auto dcs = this->all_dc[make_tuple(j, kx, ky)];

                        addmul(grads_R, get<0>(dcs), this->psi(p, E[1], j, k));
                        addmul(grads_R, get<1>(dcs), this->psi(p, E[2], j, k));
                        addmul(grads_R, get<2>(dcs), this->psi(p, E[3], j, k));
                    }
            addmul(grads, grads_R, 2 * (double(raster[x][y]) - double(this->org_img[x][y])));
        }

        return grads;
    }

    double like()
    {
        auto raster = Rasterizer(this->contour, this->w, this->h).get();
        double s = 0;
        for (auto __item : this->lattice)
        {
            int x = (int)__item.x;
            int y = (int)__item.y;
            s += pow((double(raster[x][y]) - double(this->org_img[x][y])), 2);
        }
    
        return s;
    }

private:
    int w, h, max_j, num;
    double wh;
    List<List<double>> org_img;
    Contour contour;
    List<Point> lattice;
    unordered_map<tuple<int, int, int>, tuple<List<double>, List<double>, List<double>>, 
        hash<tuple<int, int, int>>> all_dc;
};

int main()
{
    List<double> oriPath = {1, 1, 3, 1, 7, 3, 7, 7, 3, 7, 1, 3};
    List<double> optPath = {2, 1, 4, 1, 7, 3, 7, 7, 3, 7, 4, 3};

    auto convertPathToContour = [](const List<double>& path) -> Contour
    {
        List<Point> pts;
        for (int i = 0; i < path.size(); i += 2)
            pts.push_back(Point(path[i], path[i + 1]));
        return Contour(pts);
    };

    auto oriRaster = Rasterizer(convertPathToContour(oriPath), 8, 8).get();

    auto approx_dLike_dX = [convertPathToContour, oriRaster](const List<double>& X, double eps) -> List<double>
    {
        List<double> grads(X.size(), 0);

        for (int i = 0; i < X.size(); i++)
        {
            List<double> pathLarge = X;
            pathLarge[i] += eps;
            auto contourLarge = convertPathToContour(pathLarge);

            List<double> pathSmall = X;
            pathSmall[i] -= eps;
            auto contourSmall = convertPathToContour(pathSmall);

            auto likeLarge = Vectorizer(contourLarge, oriRaster).like();
            auto likeSmall = Vectorizer(contourSmall, oriRaster).like();

            grads[i] = (likeLarge - likeSmall) / (2 * eps);
        }

        return grads;
    };

    // manual
    cout << "Analytical dLike_dX" << endl;
    //
    auto __tmp0 = Vectorizer(convertPathToContour(optPath), oriRaster).dLike_dX();
    // manual
    for (auto __item : __tmp0)
        cout << __item << " ";
    cout << endl;

    cout << endl;
    cout << "Approximate dLike_dX" << endl;
    //
    auto __tmp1 = approx_dLike_dX(optPath, 1e-4);
    // manual
    for (auto __item : __tmp1)
        cout << __item << " ";
    cout << endl;
    //
}