
#include "kalman_utilities.h"



int steady_state_covariance(int n, int n_e, const double* Tmat, const double* Rmat, const double* Qmat,
			    double* P_0)
{
    triple_matrix_product(CblasNoTrans, CblasNoTrans, CblasTrans, n, n_e, n_e, n,
			  1.0, 0.0, Rmat, n_e, Qmat, n_e, Rmat, n_e, P_0, n);

    KF_CALL(steady_state_covariance(n, Tmat, P_0, P_0));
    
    return 0;
}



int steady_state_covariance(int n, const double* Tmat, const double* RQRmat,
			    double* P_0)
{
    for (int i = 0; i < n*n; i++)
	P_0[i] = RQRmat[i];
    
    double TT[n*n*n*n];
    kronecker_product(TT, Tmat, Tmat, n,n,n,n);
    
    for (int i = 0; i < n*n; i++)
	for (int j = 0; j < n*n; j++)
	    TT[n*n*i + j] *= -1.0;
    for (int i = 0; i < n*n; i++)
	TT[n*n*i + i] += 1.0;
    
    lapack_int ipiv[n * n];
    KF_CALL(LAPACKE_dgesv(LAPACK_ROW_MAJOR, n*n, 1, TT, n*n, ipiv, P_0, 1));

    symmetrify(P_0, n);
    
    return 0;
}



void predict_mu_1st_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			 const double* Tmat, const double* Ivec)
{
    for (int i = 0; i < n_out; i++)
	mu_out[i] = Ivec[i];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_out, 1, n_in,
		1.0, Tmat, n_in, mu_in, 1, 1.0, mu_out, 1);
}


void predict_mu_1st_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			 const double* Tmat)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_out, 1, n_in,
		1.0, Tmat, n_in, mu_in, 1, 0.0, mu_out, 1);
}


void predict_mu_2nd_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			  const double* TTmat, const double* Tmat, const double* Ivec,
			  double* mu_k_mu)
{
    predict_mu_1st_order(n_in, n_out, mu_in, mu_out, Tmat, Ivec);

    kronecker_product(mu_k_mu, mu_in, mu_in, n_in, 1, n_in, 1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_out, 1, n_in * n_in,
		0.5, TTmat, n_in * n_in, mu_k_mu, 1, 1.0, mu_out, 1);
}


void predict_mu_2nd_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			 const double* TTmat, const double* Tmat, double* mu_k_mu)
{
    predict_mu_1st_order(n_in, n_out, mu_in, mu_out, Tmat);

    kronecker_product(mu_k_mu, mu_in, mu_in, n_in, 1, n_in, 1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_out, 1, n_in * n_in,
		0.5, TTmat, n_in * n_in, mu_k_mu, 1, 1.0, mu_out, 1);
}


void predict_cov_1st_order(int nx, int ne, const double* P_in, double* P_out,
			 const double* Tmat, const double* Rmat, const double* Qmat)
{
    triple_matrix_product(CblasNoTrans, CblasNoTrans, CblasTrans, nx, ne, ne, nx,
			  1.0, 0.0, Rmat, ne, Qmat, ne, Rmat, ne, P_out, nx);

    triple_matrix_product(CblasNoTrans, CblasNoTrans, CblasTrans, nx, nx, nx, nx,
			  1.0, 1.0, Tmat, nx, P_in, nx, Tmat, nx, P_out, nx);
    
    symmetrify(P_out, nx);
}


void predict_cov_1st_order(int nx, const double* P_in, double* P_out,
			 const double* Tmat, const double* RQRmat)
{
    for (int i = 0; i < nx*nx; i++)
	P_out[i] = RQRmat[i];
    
    triple_matrix_product(CblasNoTrans, CblasNoTrans, CblasTrans, nx, nx, nx, nx,
			  1.0, 1.0, Tmat, nx, P_in, nx, Tmat, nx, P_out, nx);

    symmetrify(P_out, nx);
}


void compute_Fmat(int nx, int ny, const double* P_in, const double* Zmat,
		  const double* Hmat, double* Fmat)
{
    for (int i = 0; i < ny*ny; i++)
	Fmat[i] = Hmat[i];

    triple_matrix_product(CblasNoTrans, CblasNoTrans, CblasTrans, ny, nx, nx, ny,
			  1.0, 1.0, Zmat, nx, P_in, nx, Zmat, nx, Fmat, ny);

    symmetrify(Fmat, ny);
}


int compute_kalman_gain(int nx, int ny, const double* x_in, const double* P_in,
			double* K_out, const double* Zmat, const double* Hmat)
{
    // F = Z * P * Z' + H;       K = P * Z' / F = P * (F \ Z)'

    double Fmat[ny*ny];// = new double[ny * ny];
    compute_Fmat(nx, ny, P_in, Zmat, Hmat, Fmat);
    
    double iFZ[ny*nx];// = new double[ny * nx];
    for (int i = 0; i < ny*nx; i++)
	iFZ[i] = Zmat[i];

    KF_CALL(LAPACKE_dposv(LAPACK_ROW_MAJOR, 'L', ny, nx, Fmat, ny, iFZ, nx));

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nx, ny, nx,
		1.0, P_in, nx, iFZ, nx, 0.0, K_out, ny);
    
    return 0;
}


int compute_kalman_gain(int nx, int ny, const double* x_in, const double* P_in,
			double* K_out, const double* Zmat, const double* Hmat, double* Fmat)
{
    // F = Z * P * Z' + H;       K = P * Z' / F = P * (F \ Z)'
    
    double iFZ[ny*nx];// = new double[ny * nx];
    for (int i = 0; i < ny*nx; i++)
	iFZ[i] = Zmat[i];

    KF_CALL(LAPACKE_dposv(LAPACK_ROW_MAJOR, 'L', ny, nx, Fmat, ny, iFZ, nx));

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nx, ny, nx,
		1.0, P_in, nx, iFZ, nx, 0.0, K_out, ny);
    
    return 0;
}


void update_mu(int nx, int ny, const double* data_error, const double* P_in,
	       const double* Kmat, double* mu_io)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx, 1, ny,
		1.0, Kmat, ny, data_error, 1, 1.0, mu_io, 1);    
}

void update_cov(int nx, int ny, const double* Zmat, const double* Kmat, double* P_io)
{
    double P_tmp[nx*nx];// = new double[nx * nx];
    for (int i = 0; i < nx*nx; i++)
	P_tmp[i] = P_io[i];
    
    triple_matrix_product(CblasNoTrans, CblasNoTrans, CblasNoTrans, nx, ny, nx, nx,
			  -1.0, 1.0, Kmat, ny, Zmat, nx, P_io, nx, P_tmp, nx);
    
    for (int i = 0; i < nx*nx; i++)
	P_io[i] = P_tmp[i];

    symmetrify(P_io, nx);
}


int update_mu_cov_1st_order(int nx, int ny, double* mu_io, double* P_io, const double* data,
			    const double* Zmat, const double* Hmat, const double* Dvec, double* ll_out)
{
    double Fmat[ny*ny];
    compute_Fmat(nx, ny, P_io, Zmat, Hmat, Fmat);

    double v_t[ny];// = new double[ny];
    predict_mu_1st_order(nx, ny, mu_io, v_t, Zmat, Dvec);
    
    // use this opportunity to compute the likelihood...
    KF_CALL(log_mvn_density(ll_out, v_t, data, Fmat, ny, 1));
    
    for (int i = 0; i < ny; i++)
	v_t[i] = data[i] - v_t[i];

    double Kmat[nx*ny];
    
    KF_CALL(compute_kalman_gain(nx, ny, mu_io, P_io, Kmat, Zmat, Hmat, Fmat));

    update_mu(nx, ny, v_t, P_io, Kmat, mu_io);
    
    update_cov(nx, ny, Zmat, Kmat, P_io);

    return 0;
}


int update_mu_cov_1st_order(int nx, int ny, double* mu_io, double* P_io, const double* data,
			    const double* Zmat, const double* Hmat, const double* Dvec)
{
    double v_t[ny];// = new double[ny];

    predict_mu_1st_order(nx, ny, mu_io, v_t, Zmat, Dvec);

    for (int i = 0; i < ny; i++)
	v_t[i] = data[i] - v_t[i];

    double Kmat[nx*ny];// = new double[ny * ny];
    
    KF_CALL(compute_kalman_gain(nx, ny, mu_io, P_io, Kmat, Zmat, Hmat));

    update_mu(nx, ny, v_t, P_io, Kmat, mu_io);
    
    update_cov(nx, ny, Zmat, Kmat, P_io);

    return 0;
}
