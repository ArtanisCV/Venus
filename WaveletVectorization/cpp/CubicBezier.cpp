#include <iostream>
using namespace std;

class CubicBezier
{
public:
	CubicBezier(double x0, double y0, double x1, double y1, 
				double x2, double y2, double x3, double y3):
		x0(x0), y0(y0), x1(x1), y1(y1), 
		x2(x2), y2(y2), x3(x3), y3(y3)
	{
	}

	double evaluate(double t)
	{
		return (self.x0*(1-t)**3 + 3*self.x1*(1-t)**2*t 
				+ 3*self.x2*(1-t)*t**2 + self.x3*t**3,
				self.y0*(1-t)**3 + 3*self.y1*(1-t)**2*t \
													                + 3*self.y2*(1-t)*t**2 + self.y3*t**3)
	double x0, y0, x1, y1, x2, y2, x3, y3;
};

int main()
{
	CubicBezier(0,1,2,3,4,5,6,7);
}
