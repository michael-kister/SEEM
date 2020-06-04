#ifndef __MVN_H__
#define __MVN_H__

#include <mkl.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <random>

#include <matrices.h>

#ifndef MVN_CALL
#define MVN_CALL(x) do { if ((x) != 0) { \
	    printf("ERROR: %s : %d (%s)\n", \
		   __FILE__, __LINE__, __func__);    \
	    return 1;}} while(0)
#endif



void partition_covariance(const double* SIGMA, int n1, int n2,
			  double* sig_11, double* sig_12, double* sig_21, double* sig_22);

int conditional_normal_12(const double* MU, const double* SIGMA, int N,
			   double* mu, double* x2, double* sigma, int n1, int m);

int conditional_normal_21(const double* MU, const double* SIGMA, int N,
			   double* mu, double* x1, double* sigma, int n2, int m);

int mvn_transform(const double*, const double*, double*, int, int, int = 1);

void standard_normal_sample(double*, int);

void standard_normal_sample(double*, int, int);

int log_mvn_density(double*, const double*, const double*, const double*, int, int); 


// void mvn_sample(double *, double *, int, int, double *);

// double log_mvn_density(const double *, const double *, int);


#endif
