#ifndef __KALMAN_UTILS__
#define __KALMAN_UTILS__

#include <mkl.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <random>

#include <mvn.h>
#include <matrices.h>

#ifndef KF_CALL
#define KF_CALL(x) do { if ((x) != 0) { \
	    printf("ERROR: %s : %d (%s)\n", __FILE__, __LINE__, __func__); \
	    return 1;}} while(0)
#endif

int steady_state_covariance(int n, const double* Tmat, const double* RQRmat,
			    double* P_0);

int steady_state_covariance(int n, int n_e, const double* Tmat, const double* Rmat, const double* Qmat,
			    double* P_0);



void predict_mu_1st_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			 const double* Tmat, const double* Ivec);

void predict_mu_1st_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			 const double* Tmat);

void predict_mu_2nd_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			  const double* TTmat, const double* Tmat, const double* Ivec,
			  double* mu_k_mu);

void predict_mu_2nd_order(int n_in, int n_out, const double* mu_in, double* mu_out,
			 const double* TTmat, const double* Tmat, double* mu_k_mu);

void predict_cov_1st_order(int nx, int ne, const double* P_in, double* P_out,
			 const double* Tmat, const double* Rmat, const double* Qmat);

void predict_cov_1st_order(int nx, const double* P_in, double* P_out,
			 const double* Tmat, const double* RQRmat);



int update_mu_cov_1st_order(int nx, int ny, double* mu_io, double* P_io, const double* data,
			    const double* Zmat, const double* Hmat, const double* Dvec);

int update_mu_cov_1st_order(int nx, int ny, double* mu_io, double* P_io, const double* data,
			    const double* Zmat, const double* Hmat, const double* Dvec, double* ll_out);



void compute_Fmat(int nx, int ny, const double* P_in, const double* Zmat,
		  const double* Hmat, double* Fmat);

int compute_kalman_gain(int nx, int ny, const double* x_in, const double* P_in,
			double* K_out, const double* Zmat, const double* Hmat);

int compute_kalman_gain(int nx, int ny, const double* x_in, const double* P_in,
			double* K_out, const double* Zmat, const double* Hmat, double* Fmat);

void update_mu(int nx, int ny, const double* data_error, const double* P_in,
	       const double* Kmat, double* mu_io);

void update_cov(int nx, int ny, const double* Zmat, const double* Kmat, double* P_io);


#endif
