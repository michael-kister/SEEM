#ifndef __MISC_UTILS__
#define __MISC_UTILS__

#include <cmath>
#include <algorithm>
#include <vector>

void elapsed_time(int ms, int ret = 1);

int factorial(int n);

/**
 * Compute row-major index of multi-dimensional array, within
 * single dimensional array.
 *
 * n : length-d array of (zero-based) indices
 * N : length-d array of array dimensions
 * d : number of dimensions
 */
int rowMajorIndex(int* n, int* N, int d);
int rowMajorIndex(const int* n, int* N, int d);
int rowMajorIndex(int* n, const int* N, int d);
int rowMajorIndex(const int* n, const int* N, int d);

double logit(double);
double inverse_logit(double);
double change_bounds(double, double, double, double, double);

void insert_sort_double
(double* X, int* I, int N);

void heaps_determinant
(int n, int N, int* P, int* one, const std::vector<std::vector<double>>& M, double* det, double* tmp);

void heaps_determinant
(int n, int N, int* P, int* one, const double* M, double* det, double* tmp);

double calculate_determinant
(int N, const std::vector<std::vector<double>>& M);

double calculate_determinant
(int N, const double* M);

#endif
