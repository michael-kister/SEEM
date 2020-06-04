#ifndef __BFGS_H__
#define __BFGS_H__


#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

#include <functional>

#include <mkl.h>
#include <matrices.h>

#include <cmath>
#include <random>


#ifndef BFGS_CALL
#define BFGS_CALL(x) do { if ((x) != 0) { \
	    printf("ERROR: %s : %d\n", __FILE__, __LINE__); \
	    return 1;}} while(0)
#endif

#ifndef LAM_CHECK
#define LAM_CHECK(x) do { if (isnan(x)) { \
	    printf("ERROR: λ is NaN. %s : %d\n", __FILE__, __LINE__); \
	    return 1;}} while(0)
#endif


#define BFGS_VERBOSE 2
#define TOO_BIG 1000000000

typedef std::function<double(const double*)> fun_ref;

double vector_norm(const double*, int);

int check_convergence(double, double, double);

int update_Hmat(double*, double*, double*, double, int);

void numerical_gradient_2(fun_ref &, double*, double*, int);

void numerical_hessian_2(fun_ref &, double*, double*, int);

void numerical_gradient_1(fun_ref &, double*, double, double*, int);

int line_search(fun_ref &, double*, double*, double*, double*, int);

int bfgs_optimize(fun_ref &, double*, double*, double*, int);


#endif

