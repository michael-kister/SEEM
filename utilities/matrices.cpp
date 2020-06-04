#include "matrices.h"




void savemat(double* M, int nrow, int ncol, const char* filename)
{
    FILE* matfile = fopen(filename, "w");
    for (int i = 0; i < nrow; i++) {
	for (int j = 0; j < ncol; j++) {
	    fprintf(matfile, "%+012.6f ", M[ncol*i + j]);
	}
	fprintf(matfile, "\n");
    }
    fclose(matfile);
}



void symmetrify(double* M, int N)
{
    // takes an N x N matrix and takes the average of it and its transpose

    for (int i = 1; i < N; i++)
	for (int j = 0; j < i; j++)
	    M[N*i+j] = M[N*j+i] = 0.5 * (M[N*i+j] + M[N*j+i]);
}


void square_transpose(double* M, int n)
{
    for (int i = 0; i < n; i++)
	for (int j = 0; j < i; j++)
	    std::swap(M[n*i+j], M[n*j+i]);
}


void kronecker_product(double* M_out, const double* A, const double* B, int r1, int c1, int r2, int c2)
{
    // note: this assumes row major layout
    for (int i = 0; i < r1; i++)
	for (int j = 0; j < c1; j++)
	    for (int p = 0; p < r2; p++)
		for (int q = 0; q < c2; q++)
		    M_out[r2*c1*c2*i+c1*c2*p+c2*j+q] = A[c1*i+j] * B[c2*p+q];
}


void triple_matrix_product(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
			   const CBLAS_TRANSPOSE transc,
			   const MKL_INT m, const MKL_INT j, const MKL_INT k, const MKL_INT n,
			   const double alpha, const double beta,
			   const double* a, const MKL_INT lda,
			   const double* b, const MKL_INT ldb,
			   const double* c, const MKL_INT ldc,
			   double* d, const MKL_INT ldd)
{
    // this function does not allow for column-major layout, or number of columns
    // which is different from the leading dimension of the matrix. There is a very
    // minor optimization which recognizes the option of performing the multiplcations
    // in different orders.
    //
    // [m x j] [j x k] [k x n] + [m x n]

    int size_tmp;
    
    if (m*j*k + m*k*n <= m*j*n + j*k*n)
	
	size_tmp = m*k;
    else
	size_tmp = j*n;

    double* tmp = new double[size_tmp];
    
    if (m*j*k + m*k*n <= m*j*n + j*k*n) {

	// do the first two first
	cblas_dgemm(CblasRowMajor, transa, transb, m, k, j, 1.0, a, lda, b,ldb, 0.0, tmp, k);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, transc, m, n, k, alpha, tmp, k, c, ldc, beta, d, ldd);
	
    } else {

	// do the second two first
	cblas_dgemm(CblasRowMajor, transb, transc, j, n, k, 1.0, b, ldb, c, ldc, 0.0, tmp, n);
	cblas_dgemm(CblasRowMajor, transa, CblasNoTrans, m, n, j, alpha, a, lda, tmp, n, beta, d, ldd);
	
    }
    
    delete[] tmp;
}


void printmat4 (double* M, int d1, int d2, int d3, int d4) {
    // This function prints a matrix, assuming that it is
    // given in generalized row-major format (where the last
    // dimension is adjacent)
    double value;
    printf("┌");
    for (int i = 0; i < (10*d2+2)*d4+2; i++)
	printf("─");
    printf("┐\n");
    // go through each row of blocks
    for (int i = 0; i < d3; i++) {
	printf("│ ");
	for (int j = 0; j < d4; j++) {
	    printf("┌");
	    for (int k = 0; k < 10*d2; k++)
		printf("─");
	    printf("┐");
	}
	printf(" │\n");
	for (int j = 0; j < d1; j++) {
	    printf("│ ");
	    for (int k = 0; k < d4; k++) {
		printf("│");
		for (int l = 0; l < d2; l++) {
		    value = M[d4*d3*d2*j + d4*d3*l + d4*i + k];
		    if (fabs(value) < 0.001)
			if (value == 0.0)
			    printf("        0 ");
			else if (fabs(value) < 1.0e-10)
			    if (value < 0.0)
				printf("       -0 ");
			    else
				printf("       +0 ");
			else
			    printf("%+9.2e ", value);
		    else if (fabs(value) >= 1000.0)
			printf("%9.2e ", value);
		    else
			printf("%9.4f ", value);
		    }
		printf("│");
	    }
	    printf(" │\n");
	}
	printf("│ ");
	for (int j = 0; j < d4; j++) {
	    printf("└");
	    for (int k = 0; k < 10*d2; k++)
		printf("─");
	    printf("┘");
	}
	printf(" │\n");
    }
    printf("└");
    for (int i = 0; i < (10*d2+2)*d4+2; i++)
	printf("─");
    printf("┘\n");
}


void printmat4 (const double* M, int d1, int d2, int d3, int d4) {
    double tmp[d1*d2*d3*d4];
    for (int i = 0; i < d1*d2*d3*d4; i++)
	tmp[i] = M[i];
    printmat4(tmp, d1, d2, d3, d4);
}
