
#include "miscellaneous.h"



/**
 * Print elapsed time, given milliseconds as an input. The default is to print
 * a newline, but that can be disabled.
 */
void elapsed_time(int ms, int ret)
{
    int HR = ms / 1000 / 60 / 60;
    ms -= 1000 * 60 * 60 * HR;
    int MN = ms / 1000 / 60;
    ms -= 1000 * 60 * MN;
    int SC = ms / 1000;
    ms -= 1000 * SC;

    if (SC == 0) {
 	printf("--h --m --s %03dms", ms);
    } else if (MN == 0) {
 	printf("--h --m %02ds %03dms", SC, ms);
    } else if (HR == 0) {
 	printf("--h %02dm %02ds %03dms", MN, SC, ms);
    } else {
 	printf("%02dh %02dm %02ds %03dms", HR, MN, SC, ms);
    }
    
    if (ret)
	printf("\n");
}



int factorial(int n)
{   
    return (n <= 1) ? 1 : n * factorial(n-1);
}

/**
 * Compute row-major index of multi-dimensional array, within
 * single dimensional array.
 *
 * n : length-d array of (zero-based) indices
 * N : length-d array of array dimensions
 * d : number of dimensions
 */
int rowMajorIndex(const int* n, const int* N, int d)
{
    if (d < 1)
	return 0;
    
    if (d-- == 1)
	return n[0];
    else
	return n[d] + (N[d] * rowMajorIndex(n, N, d));
}


double logit(double x)
{
    return log( x / (1.0-x) );
}

double inverse_logit(double x)
{
    return exp(x) / (1.0 + exp(x));
}

double change_bounds(double x, double a, double b, double c, double d)
{
    // if x is between a and b, and y = f(x) is between c and d
    return ((d-c)*(x-a)/(b-a)) + c;
}

void insert_sort_double(double* X, int* I, int N)
{
    // ensure that indices are how we want them
    for (int i = 0; i < N; i++)
	I[i] = i;
    
    int i = 1;
    int j = 1;
    while (i < N) {
	j = i;
	while (j > 0 && X[j-1] > X[j]) {
	    std::swap(X[j], X[j-1]);
	    std::swap(I[j], I[j-1]);
	    j--;
	}
	i++;
    }
}

/*------------------------------------------------------------------------------
 * These are functions relevant to computing the determinant
 *----------------------------------------------------------------------------*/

void heaps_determinant
(int n, int N, int* P, int* one, const std::vector<std::vector<double>>& M, double* det, double* tmp)
{
    if (n > 1) {
	heaps_determinant(--n, N, P, one, M, det, tmp);
	for (int i = 0; i < n; i++) {
	    std::swap((n%2) ? P[i] : P[0], P[n]);
	    one[0] *= -1;
	    heaps_determinant(n, N, P, one, M, det, tmp);
	}
    } else {
	tmp[0] = 1.0;
	for (int i = 0; i < N; i++)
	    tmp[0] *= M[i][P[i]];
	det[0] += one[0] * tmp[0];
    }
}

void heaps_determinant
(int n, int N, int* P, int* one, const double* M, double* det, double* tmp)
{
    if (n > 1) {
	heaps_determinant(--n, N, P, one, M, det, tmp);
	for (int i = 0; i < n; i++) {
	    std::swap((n%2) ? P[i] : P[0], P[n]);
	    one[0] *= -1;
	    heaps_determinant(n, N, P, one, M, det, tmp);
	}
    } else {
	tmp[0] = 1.0;
	for (int i = 0; i < N; i++)
	    tmp[0] *= M[N*i+P[i]];
	det[0] += one[0] * tmp[0];
    }
}

double calculate_determinant
(int N, const std::vector<std::vector<double>>& M)
{
    double det = 0.0;
    double tmp = 1.0;
    int one = 1;
    int permutations[N];
    for (int i = 0; i < N; i++)
	permutations[i] = i;
    heaps_determinant(N, N, permutations, &one, M, &det, &tmp);
    return det;    
}

double calculate_determinant
(int N, const double* M)
{
    double det = 0.0;
    double tmp = 1.0;
    int one = 1;
    int permutations[N];
    for (int i = 0; i < N; i++)
	permutations[i] = i;
    heaps_determinant(N, N, permutations, &one, M, &det, &tmp);
    return det;    
}

/*------------------------------------------------------------------------------
 * These are functions relevant to computing the determinant
 *----------------------------------------------------------------------------*/

