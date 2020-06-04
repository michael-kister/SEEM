
#include "projection_utilities.h"

int mfun(short int i)
{
    return i ? (int)(1 << i) : 0;
}

double nfun(short int i)
{
    return i ? (double)(1 << i) : 0.5;
}

int factorial(int n)
{   
    return (n <= 1) ? 1 : n * factorial(n-1);
}

int binomial_coefficient(int n, int k)
{
    return factorial(n) / (factorial(k) * factorial(n-k));
}

inline double chebyshev_type_1(int n, double x)
{
    if (n == 0)
	return 1.0;
    else if (n == 1)
	return x;
    else if (n == 2)
	return 2.0*x*x - 1.0;
    else if (n == 3)
	return x*(4.0*x*x - 3.0);
    else if (n == 4)
	return 8.0*x*x*x*x - 8.0*x*x + 1.0;
    else if (n == 5)
	return x*(16.0*x*x*x*x - 20.0*x*x + 5.0);
    else if (n == 6)
	return 32.0*x*x*x*x*x*x - 48.0*x*x*x*x + 18.0*x*x - 1.0;
    else if (n == 7)
	return x*(64.0*x*x*x*x*x*x - 112.0*x*x*x*x + 56.0*x*x - 7.0);
    else if (n == 8)
	return 128.0*x*x*x*x*x*x*x*x - 256.0*x*x*x*x*x*x + 160.0*x*x*x*x - 32.0*x*x + 1.0;
    else
	return recursive_chebyshev_type_1(n, x);
}

double recursive_chebyshev_type_1(int n, double x)
{
    if (n <= 8)
	return chebyshev_type_1(n, x);
    else
	return 2.0*x*recursive_chebyshev_type_1(n-1, x) - recursive_chebyshev_type_1(n-2, x);
}

inline double chebyshev_type_2(int n, double x)
{
    if (n == 0)
	return 1.0;
    else if (n == 1)
	return 2.0*x;
    else if (n == 2)
	return 4.0*x*x - 1.0;
    else if (n == 3)
	return x*(8.0*x*x - 4.0);
    else if (n == 4)
	return 16.0*x*x*x*x - 12.0*x*x + 1.0;
    else if (n == 5)
	return x*(32.0*x*x*x*x - 32.0*x*x + 6.0);
    else if (n == 6)
	return 64.0*x*x*x*x*x*x - 80.0*x*x*x*x + 24.0*x*x - 1.0;
    else if (n == 7)
	return x*(128.0*x*x*x*x*x*x - 192.0*x*x*x*x + 80.0*x*x - 8.0);
    else if (n == 8)
	return 256.0*x*x*x*x*x*x*x*x - 448.0*x*x*x*x*x*x + 240.0*x*x*x*x - 40.0*x*x + 1.0;
    else
	return recursive_chebyshev_type_2(n, x);
}

double recursive_chebyshev_type_2(int n, double x)
{
    if (n <= 8)
	return chebyshev_type_2(n, x);
    else
	return 2.0*x*recursive_chebyshev_type_2(n-1, x) - recursive_chebyshev_type_2(n-2, x);
}

