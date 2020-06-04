
#include "mvn.h"


void partition_covariance(const double* SIGMA, int n1, int n2,
			  double* sig_11, double* sig_12, double* sig_21, double* sig_22)
{
    int N = n1 + n2;
    
    for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++) {
	    if (i < n1) {
		if (j < n1) {
		    // top left
		    sig_11[n1*i + j] = SIGMA[N*i + j];
		} else {
		    // top right
		    sig_12[n2*i + j - n1] = SIGMA[N*i + j];
		}
	    } else {
		if (j < n1) {
		    // bottom left
		    sig_21[n1*(i-n1) + j] = SIGMA[N*i + j];
		} else {
		    // bottom right
		    sig_22[n2*(i-n1) + j - n1] = SIGMA[N*i + j];
		}
	    }
	}
    }
}


/**
 *  Σ(2|1)  = Σ22 - (Σ21 * (Σ11 \ Σ12))
 *  μ(2|1)' = μ22' + (x1' - μ1') * (Σ11 \ Σ12)
 *
 *  Where N is the dimension of the full random variable, n is the dimension
 *  of the conditional distribution, and m is the number of observations.
 */
int conditional_normal_21(const double* MU, const double* SIGMA, int N,
			  double* mu, double* x1, double* sigma, int n2, int m)
{
    int n1 = N - n2;
    
    double sig_11[n1 * n1];
    double sig_12[n1 * n2];
    double sig_21[n2 * n1];
    double x1mMu1[m * n1];

    for (int i = 0; i < m; i++)
	for (int j = 0; j < N; j++)
	    if (j < n1)
		x1mMu1[n1*i + j] = x1[n1*i + j] - MU[j];
	    else
		mu[n2*i + j - n1] = MU[j];
    
    partition_covariance(SIGMA, n1, n2, sig_11, sig_12, sig_21, sigma);
    
    MVN_CALL(LAPACKE_dposv(LAPACK_ROW_MAJOR, 'L', n1, n2, sig_11, n1, sig_12, n2));

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n2, n2, n1,
		-1.0, sig_21, n1, sig_12, n2, 1.0, sigma, n2);
	
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n2, n1,
		1.0, x1mMu1, n1, sig_12, n2, 1.0, mu, n2);

    return 0;
}



int mvn_transform(const double* mu, const double* sigma, double* x, int n_var, int n_samp, int n_mu)
{
    double cov[n_var * n_var];
    for (int i = 0; i < n_var*n_var; i++)
	cov[i] = sigma[i];

    MVN_CALL(LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n_var, cov, n_var));
    for (int i = 0; i < n_var; i++)
	for (int j = i+1; j < n_var; j++)
	    cov[n_var*i + j] = 0.0;
    
    cblas_dtrmm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
		n_samp, n_var, 1.0, cov, n_var, x, n_var);

    if (n_mu == 1)
	for (int i = 0; i < n_samp; i++)
	    for (int j = 0; j < n_var; j++)
		x[n_var*i + j] += mu[j];
    else
	for (int i = 0; i < n_samp; i++)
	    for (int j = 0; j < n_var; j++)
		x[n_var*i + j] += mu[n_var*i + j];
    
    return 0;
} 


void standard_normal_sample(double* sample, int N)
{
    std::mt19937_64 gen;
    std::normal_distribution<double> norm(0.0, 1.0);

    for (int i = 0; i < N; i++)
	sample[i] = norm(gen);
}


void standard_normal_sample(double* sample, int N, int seed)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    for (int i = 0; i < N; i++)
	sample[i] = norm(gen);
}


int log_mvn_density(double* y, const double* x, const double* mu, const double* sigma, int d, int n)
{
    // y is an array of length n, and is the output
    // x is an array of length n (rows) multiplied by d (columns)
    // mu is an array of length d
    // sigma is an array of length d*d
    //
    // the log density is given by:
    //
    //  -0.5 * log(det(2πΣ)) - 0.5 * (x-μ)'(Σ \ (x-μ)) = 
    //
    //  -0.5 * { d*log(2π) + log(det(Σ)) + (x-μ)(Σ\(x-μ)') }
    //
    // Note that the transpose is swapped because we're defining our vectors in rows.
    // 
    // Basically, you want to factorize sigma, so that you can grab the diagonals
    // and compute the determinant. Then you solve the equation: Σ\(x-μ). Then you
    // obtain the products with the (x-μ)'.

    double log2pi = 1.8378770664093453;

    for (int i = 0; i < d; i++)
	if (isnan(mu[i]))
	    printf("WRONG: NaN in mu! (element %d)\n", i);

    double xmmt[d * n]; // "x minus mu, transposed" (d x n)
    for (int i = 0; i < d; i++)
	for (int j = 0; j < n; j++)
	    xmmt[n*i + j] = x[d*j + i] - mu[i];
    
    double sig[d * d];
    for (int i = 0; i < d*d; i++)
	sig[i] = sigma[i];

    lapack_int ipiv[d];
    MVN_CALL(LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', d, sig, d, ipiv));
    //MVN_CALL(LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', d, sig, d));

    double coeff = 1.0;
    for (int i = 0; i < d; i++)
	coeff *= sig[d*i + i];
	//coeff *= (sig[d*i + i] * sig[d*i + i]);
    coeff = log(coeff);
    coeff += d * log2pi;

    double xmmt_bu[d * n];
    for (int i = 0; i < d*n; i++)
	xmmt_bu[i] = xmmt[i];

    int info = LAPACKE_dsytrs(LAPACK_ROW_MAJOR, 'L', d, n, sig, d, ipiv, xmmt, n);
    //int info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', d, n, sig, d, xmmt, n);
    if (info != 0) {
	printf("info : %d\n", info);
	printmat4(x, 10, d, 1, 1);
	return 1;
    }
    
    for (int i = 0; i < n; i++){
	y[i] = 0.0;
	for (int j = 0; j < d; j++)
	    y[i] += (x[d*i+j] - mu[j]) * xmmt[j*n+i];
	
	y[i] += coeff;
	y[i] *= -0.5;
    }

    return 0;
}










// I don't think there's anything wrong with these functions; I'm just not using them atm

/*
double log_mvn_density(const double *mu, const double *sigma, int N) {

    lapack_int info;
    lapack_int ipiv[N];
    
    double log2pi = 1.8378770664093453;
    double S[N*N];
    double M[N];
    double out;
    double kernel[1];

    // Fill upper triangular part of matrix
    for (size_t i = 0; i < N; i++) {
	for (size_t j = i; j < N; j++)
	    S[j+i*N] = sigma[j+i*N];
	M[i] = mu[i];
    }

    // COEFFICIENT
    
    // Compute Bunch-Kaufman factorization of covariance matrix
    info = LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'U', N, S, N, ipiv);

    out = 1.0;
    for (size_t i = 0; i < N; i++)
	out *= S[i+i*N];
    out = log( out );
    out += N*log2pi;
    
    // KERNEL

    // M = S\mu
    info = LAPACKE_dsytrs(LAPACK_ROW_MAJOR, 'U', N, 1, S, N, ipiv, M, 1);

    // kernel = mu'*M
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 1, N, 1.0, mu, 1, M, 1, 0.0, kernel, 1);
    
    out += kernel[0];
    out *= -0.5;
    return out;
    
}

void mvn_sample(double *mu, double *sigma, int N, int D, double *sample) {

    // Produces a N x D array in sample
    double std_norm [N*D];
    double upper    [D*D];

    lapack_int info;    // output of factorization
    lapack_int rank[1]; // rank of matrix (output of factorization)
    lapack_int piv[D];  // array with pivots (output of factorization)

    
    std::default_random_engine       generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < N*D; i++) {
	std_norm[i] = distribution(generator);
    }

    // obtain upper triangular factorization of sigma
    for (int i = 0; i < D; i++)
	for (int j = 0; j < D; j++)
	    upper[j+i*D] = sigma[j+i*D];
    info = LAPACKE_dpstrf(LAPACK_ROW_MAJOR, 'U', D, upper, D, piv, rank, -1);
    for (int i = 1; i < D; i++)
	for (int j = 0; j < i; j++)
	    upper[j+i*D] = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, D, D, 1.0, std_norm, D, upper, D, 0.0, sample, D);

    for (int n = 0; n < N; n++)
	for (int d = 0; d < D; d++)
	    sample[d+D*n] += mu[d];
    
    for (int i = 0; i < N; i++)
	printf("%+7.2f  %+7.2f\n", sample[0+i*D], sample[1+i*D]);
    
    std::cout << "hi" << std::endl;
    
}
*/
