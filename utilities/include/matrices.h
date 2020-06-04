#ifndef _MATRIX_FUNS_
#define _MATRIX_FUNS_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <mkl.h>



void square_transpose(double*, int);

void kronecker_product(double*, const double*, const double*, int, int, int, int);

void triple_matrix_product(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
			   const CBLAS_TRANSPOSE transc,
			   const MKL_INT m, const MKL_INT j, const MKL_INT k, const MKL_INT n,
			   const double alpha, const double beta,
			   const double* a, const MKL_INT lda,
			   const double* b, const MKL_INT ldb,
			   const double* c, const MKL_INT ldc,
			   double* d, const MKL_INT ldd);

void printmat4 (double*, int, int, int, int);
void printmat4 (const double*, int, int, int, int);

void savemat(double*, int, int, const char*);

void symmetrify(double*, int);

#endif
