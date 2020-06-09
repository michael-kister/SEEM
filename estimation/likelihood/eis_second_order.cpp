#include "eis_second_order.h"


//========================================//
//                                        //
//         PUBLIC MODEL METHODS           //
//                                        //
//========================================//

int eis_model::eis_likelihood(mod_2nd_order* mod, eis_opts* opts, double* data, double* LL_out)
{
    // import options
    int T = opts->T;
    int n_samp = opts->n_samp;
    int n_y = opts->n_y;
    int n_e = opts->n_e;
    int n_p = opts->n_p;
    int n_q = opts->n_q;
    int n_qpp = n_q + n_p + n_p;
    int n_x = n_q + n_p;

    int verbose = 0;
    
    // constants within the function
    int max_iter = 10;
    double eis_tol = .0001;
    double* CRN = new double[n_qpp * n_samp]();
    
    // containers based on options
    double* omega = new double [n_samp]();
    double log_likelihood_t;
    
    // containers for importance distribution, along with marginals/conditionals
    double* mu_t = new double   [n_qpp]();
    double* sig_t = new double  [n_qpp*n_qpp]();
    
    double* mu_tm1 = new double  [n_qpp]();
    double* sig_tm1 = new double [n_qpp*n_qpp]();

    double* mar_mu_t = new double  [n_samp * n_x]();
    double* mar_sig_t = new double [n_x * n_x]();

    double* mar_mu_tm1 = new double  [n_samp * n_x]();
    double* mar_sig_tm1 = new double [n_x * n_x]();
    
    double* cond_mu_t = new double  [n_samp * n_p]();
    double* cond_sig_t = new double [n_p * n_p]();

    // containers for pieces we want to examine (for convenience)
    double* qpp_t = new double [n_samp * n_qpp]();
    double* qp_t = new double  [n_samp * n_x]();
    double* qp_tm1 = new double[n_samp * n_x]();
    double* q_t = new double   [n_samp * n_q]();
    double* p_t = new double   [n_samp * n_p]();
    double* q_tm1 = new double [n_samp * n_q]();
    double* p_tm1 = new double [n_samp * n_p]();

    double* y_hat = new double [n_samp * n_y]();
    double* x_hat = new double [n_x]();
    double* p_hat = new double [n_samp * n_p]();
    double* x_kron_x = new double [n_x * n_x]();

    double* np_zeros = new double [n_p]();
    //for (int i = 0; i < n_p; i++)
    //np_zeros[i] = 0.0;

    // containers for dependent variable in the regression (and IS density)
    double* p_yt_xt = new double     [n_samp]();
    double* p_pt_xtm1 = new double   [n_samp]();
    double* p_xtm1_ytm1 = new double [n_samp]();
    double* jacobian = new double    [n_samp]();
    
    double* p_import = new double  [n_samp]();


    // relevant to EIS regressions
    int n_suff_stats = 1 + n_qpp + 0.5*n_qpp*(n_qpp+1);
    double* suff_stats = new double[n_samp * n_suff_stats]();
    int suff_ind;

    double* beta = new double[n_suff_stats]();
    double* beta_prev = new double[n_suff_stats]();

    // set the matrices for the EKF
    double* Zmat = new double[n_y * n_qpp]();
    double* Tmat = new double[n_qpp * n_qpp]();
    
    double* Hmat = mod->Hmat;
    double* Rmat = new double[n_qpp * n_e]();
    double* Qmat = mod->Qmat;
    double* RQRp = new double[n_qpp * n_qpp]();

    double* Cvec = new double[n_qpp]();
    double* Dvec = new double[n_y]();

    for (int i = 0; i < n_y; i++)
	for (int j = 0; j < n_qpp; j++)
	    Zmat[n_qpp*i + j] = (j < n_x) ? mod->gx_lev_1[n_x*i + j] : 0.0;
	
    for (int i = 0; i < n_x; i++)
	for (int j = 0; j < n_qpp; j++)
	    Tmat[n_qpp*i + j] = (j < n_x) ? mod->hx_lev_1[n_x*i + j] : 0.0;
    
    for (int i = 0; i < n_p; i++)
	Tmat[n_qpp*(n_x+i) + i + n_q] = 1.0;

    for (int i = n_q; i < n_x; i++)
	Rmat[n_e*i + i - n_q] = 1.0;

    triple_matrix_product(CblasNoTrans, CblasNoTrans, CblasTrans, n_qpp, n_e, n_e, n_qpp,
			  1.0, 0.0, Rmat, n_e, Qmat, n_e, Rmat, n_e, RQRp, n_qpp);

    for (int i = 0; i < n_qpp; i++)
	Cvec[i] = (i < n_x) ? mod->h_lev_1[i] : 0.0;
	
    for (int i = 0; i < n_y; i++)
	Dvec[i] = mod->g_lev_1[i];

    
    //--------------------------------------------------------------------------
    // set the starting values (and the "previous marginal values"
    EIS_CALL(steady_state_covariance(n_qpp, Tmat, RQRp, sig_tm1));
    
    for (int i = 0; i < n_x; i++)
	for (int j = 0; j < n_x; j++)
	    mar_sig_tm1[n_x*i + j] = sig_tm1[n_qpp*i + j];
    
    for (int i = 0; i < n_x; i++)
	mar_mu_tm1[i] = mu_tm1[i] = mod->xss[i];
    for (int i = 0; i < n_p; i++)
	mu_tm1[n_x + i] = mod->xss[n_q + i];
    
    int iter;
    int fini;

    *LL_out = 0.0;

    if (verbose) {
	printf("==================================\n");
	printf(" Time  Iterations  Log-likelihood \n");
	printf(" ----  ----------  -------------- \n");
    }

    for (int t = 0; t < T; t++) {

	if (t == 0)
	    standard_normal_sample(CRN, n_qpp*n_samp, seed);
	else
	    standard_normal_sample(CRN, n_qpp*n_samp);
	
	iter = 0;
	fini = 0;
	while (fini == 0) {  
	    
	    if (iter == 0) {
		// construct linear approximation to p(q(t), p(t), p(t-1)) using
		// previously obtained p(q(t-1), p(t-1))
		//
		// [ q_t ]   [[ hx ] 0 ] [q_tm1]
		// [ p_t ] = [[ hx ] 0 ] [p_tm1]
		// [p_tm1]   [ 0  I  0 ] [p_tm2]
		predict_mu_1st_order(n_qpp, n_qpp, mu_tm1, mu_t, Tmat, Cvec);
		predict_cov_1st_order(n_qpp, sig_tm1, sig_t, Tmat, RQRp);

		// condition contemporaneous mean on data:
		//
		// [ q_t ]     [ q_tt ]
		// [ p_t ] --> [ p_tt ]
		// [p_tm1]     [ p_tm1] (this one shouldn't change)
		EIS_CALL(update_mu_cov_1st_order(n_qpp, n_y, mu_t, sig_t, data+(n_y*t),
						 Zmat, Hmat, Dvec));

		// extract the marginal distribution of [q_tt' p_tt']'
		for (int i = 0; i < n_x; i++) {
		    mar_mu_t[i] = mu_t[i];
		    for (int j = 0; j < n_x; j++)
			mar_sig_t[n_x*i + j] = sig_t[n_qpp*i + j];
		}

		// draw [q_tt' p_tt']'
		for (int i = 0; i < n_samp; i++)
		    for (int j = 0; j < n_x; j++)
			qp_t[n_x*i + j] = CRN[n_qpp*i + j];
		EIS_CALL(mvn_transform(mar_mu_t, mar_sig_t, qp_t, n_x, n_samp, 1));

		
		// condition [p_tm1] on [q_tt' p_tt']'
		EIS_CALL(conditional_normal_21(mu_t, sig_t, n_qpp, cond_mu_t, qp_t,
					       cond_sig_t, n_p, n_samp));
		
		// draw [p_tm1']' conditional on [q_tt' p_tt']'
		for (int i = 0; i < n_samp; i++)
		    for (int j = 0; j < n_p; j++)
			p_tm1[n_p*i + j] = CRN[n_qpp*i + j + n_x];
		EIS_CALL(mvn_transform(cond_mu_t, cond_sig_t, p_tm1, n_p, n_samp, n_samp));

		
		// pull elements together
		for (int i = 0; i < n_samp; i++)
		    for (int j = 0; j < n_qpp; j++)
			if (j < n_x)
			    qpp_t[n_qpp*i + j] = qp_t[n_x*i + j];
			else
			    qpp_t[n_qpp*i + j] = p_tm1[n_p*i + j - n_x];

		
		// set initial beta values
		for (int i = 0; i < n_suff_stats; i++)
		    beta_prev[i] = 1.0;
		
	    } else {
		
		// sample [q_t' p_t' p_tm1']' using previous results
		for (int i = 0; i < n_samp; i++)
		    for (int j = 0; j < n_qpp; j++)
			qpp_t[n_qpp*i + j] = CRN[n_qpp*i + j];
		
		EIS_CALL(mvn_transform(mu_t, sig_t, qpp_t, n_qpp, n_samp, 1));

	    }
	    
	    for (int j = 0; j < n_qpp; j++)
		if (fabs(qpp_t[j]) > 10.0) {
		    printf("Whoa, big! (i = %d, j = %d) (iteration %d)\n", 0, j, iter);
		    printmat4(mu_t, 1, n_qpp, 1,1);
		    printmat4(Tmat, n_qpp, n_qpp, 1,1);
		    printmat4(Cvec, n_qpp, 1,1,1);
		    break;
		}
	    
	    // create the independent variable
	    for (int i = 0; i < n_samp; i++) {
		suff_stats[n_suff_stats*i] = 1.0;
		for (int j = 0; j < n_qpp; j++)
		    suff_stats[n_suff_stats*i + j+1] = qpp_t[n_qpp*i + j];
		suff_ind = n_suff_stats*i + n_qpp + 1;
		
		for (int j = 0; j < n_qpp; j++)
		    for (int k = 0; k <= j; k++)
			suff_stats[suff_ind++] = (j == k) ?
			    -0.5 * qpp_t[n_qpp*i + j] * qpp_t[n_qpp*i + k] :
			    -1.0 * qpp_t[n_qpp*i + j] * qpp_t[n_qpp*i + k];
	    }

	    // pull elements apart
	    for (int i = 0; i < n_samp; i++)
		for (int j = 0; j < n_qpp; j++)
		    if (j < n_q)
			q_t   [n_q*i + j]       = qpp_t[n_qpp*i + j];
		    else if (j < n_x)
			p_t   [n_p*i + j - n_q] = qpp_t[n_qpp*i + j];
		    else
			p_tm1 [n_p*i + j - n_x] = qpp_t[n_qpp*i + j];
	    
	    
	    // obtain the q_t1 along with the jacobian
	    for (int i = 0; i < n_samp; i++) {
		EIS_CALL(invert_distribution(q_t+i*n_q, p_t+i*n_p, q_tm1+i*n_q, p_tm1+i*n_p, jacobian+i));
		for (int j = 0; j < n_x; j++)
		    qp_tm1[n_x*i + j] = (j < n_q) ? q_tm1[n_q*i+j] : p_tm1[n_p*i + j - n_q];
	    }
	    
	    
	    //------------------------------------------------------------------
	    // Make the dependent variable
	    
	    // log measurement density
	    for (int i = 0; i < n_samp; i++)
		predict_mu_2nd_order(n_x, n_y, qpp_t+(n_qpp*i), y_hat+(n_y*i),
				     mod->gxx_lev_2, mod->gx_lev_2, mod->g_lev_2, x_kron_x);
	    EIS_CALL(log_mvn_density(p_yt_xt, y_hat, data+(n_y*t), mod->Hmat, n_y, n_samp));

	    
	    // log transition density
	    for (int i = 0; i < n_samp; i++) {
		predict_mu_2nd_order(n_x, n_x, qp_tm1+(n_x*i), x_hat,
				     mod->hxx_lev_2, mod->hx_lev_2, mod->h_lev_2, x_kron_x);

		//predict_mu_1st_order(n_x, n_x, qp_tm1+(n_x*i), x_hat,
		//		     mod->hx_lev_1, mod->h_lev_1);

		for (int j = 0; j < n_p; j++)
		    p_hat[n_p*i + j] = x_hat[n_q + j] - p_t[n_p*i + j];
	    }

	    EIS_CALL(log_mvn_density(p_pt_xtm1, p_hat, np_zeros, Qmat, n_p, n_samp));
	    
	    // previous filtered density
	    EIS_CALL(log_mvn_density(p_xtm1_ytm1, qp_tm1, mar_mu_tm1, mar_sig_tm1, n_x, n_samp));

	    // set the dependent variable (where p_yt_xt was)
	    for (int i = 0; i < n_samp; i++)
		p_yt_xt[i] += p_pt_xtm1[i] + p_xtm1_ytm1[i] + log(jacobian[i]);
	    //------------------------------------------------------------------
	    
	    // run regression
	    EIS_CALL_G(ordinary_least_squares(suff_stats, p_yt_xt, beta, n_suff_stats, 1, n_samp));

	    // put the coefficients into the hyperparameter locations
	    EIS_CALL_G(map_from_beta(beta, mu_t, sig_t, n_qpp));
	    
	    iter++;
	    if (iter >= max_iter)
		fini = 1;
	    else
		fini = check_convergence(beta_prev, beta, n_suff_stats, eis_tol);
	    
	    for (int i = 0; i < n_suff_stats; i++)
		beta_prev[i] = beta[i];
	    
	} LLCALC:
	
        //----------------------------------------------------------------------
	// calculate the ratios of the likelihood
	EIS_CALL(log_mvn_density(p_import, qpp_t, mu_t, sig_t, n_qpp, n_samp));

	// calculate the weights
	for (int i = 0; i < n_samp; i++)
	    omega[i] = exp( p_yt_xt[i] - p_import[i]);

	// store it as the log-likelihood
	log_likelihood_t = log(calculate_mean(omega, n_samp ));
	*LL_out += log_likelihood_t;

	if (verbose)
	    printf(" %4d  %10d  %+14.3f\n", t, iter, log_likelihood_t);

	
	// prepare for the next time
	for (int i = 0; i < n_qpp; i++) {
	    mu_tm1[i] = mu_t[i];
	    for (int j = 0; j < n_qpp; j++)
		sig_tm1[n_qpp*i + j] = sig_t[n_qpp*i + j];
	}
	for (int i = 0; i < n_x; i++) {
	    mar_mu_tm1[i] = mu_t[i];
	    for (int j = 0; j < n_x; j++)
		mar_sig_tm1[n_x*i + j] = sig_t[n_qpp*i + j];
	}
    }
    
    if (verbose)
	printf("==================================\n");


    delete[] CRN;
    delete[] omega;
    delete[] mu_t;
    delete[] sig_t;
    delete[] mu_tm1;
    delete[] sig_tm1;
    delete[] mar_mu_t;
    delete[] mar_sig_t;
    delete[] mar_mu_tm1;
    delete[] mar_sig_tm1;
    delete[] cond_mu_t;
    delete[] cond_sig_t;
    delete[] qpp_t;
    delete[] qp_t;
    delete[] qp_tm1;
    delete[] q_t;
    delete[] p_t;
    delete[] q_tm1;
    delete[] p_tm1;
    delete[] y_hat;
    delete[] x_hat;
    delete[] p_hat;
    delete[] x_kron_x;
    delete[] np_zeros;

    delete[] p_yt_xt;
    delete[] p_pt_xtm1;
    delete[] p_xtm1_ytm1;
    delete[] jacobian;
    delete[] p_import;
    delete[] suff_stats;
    delete[] beta;
    delete[] beta_prev;
    delete[] Zmat;
    delete[] Tmat;
    delete[] Rmat;
    delete[] RQRp;
    delete[] Cvec;
    delete[] Dvec;
    
    
    return 0;
}


//========================================//
//                                        //
//    FUNCTION FOR MAPPING REGRESSIONS    //
//                                        //
//========================================//


int eis_model::map_from_beta(const double* beta, double* mu, double* sigma, int n_qpp)
{
    // given beta, obtain mu and sigma

    double mu_tmp[n_qpp];
    double sig_tmp[n_qpp*n_qpp];
    
    double sig_inv[n_qpp * n_qpp];
    int beta_ind = n_qpp + 1;
    for (int i = 0; i < n_qpp; i++)
	for (int j = 0; j <= i; j++)
	    if (i == j)
		sig_inv[n_qpp*i + j] = beta[beta_ind++];
	    else
		sig_inv[n_qpp*i + j] = sig_inv[n_qpp*j + i] = beta[beta_ind++];
    
    symmetrify(sig_inv, n_qpp);
    //EIS_CALL(LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n_qpp, sig_inv, n_qpp));
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n_qpp, sig_inv, n_qpp);
    if (info != 0) {
	//printmat4(beta, 33,3,1,1);
	return 1;
    }
    
    // solve for mu
    for (int i = 0; i < n_qpp; i++)
	mu_tmp[i] = beta[i+1];
    EIS_CALL(LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n_qpp, 1, sig_inv, n_qpp, mu_tmp, 1));

    // solve for sigma
    for (int i = 0; i < n_qpp; i++)
	for (int j = 0; j < n_qpp; j++)
	    sig_tmp[n_qpp*i + j] = (i == j) ? 1.0 : 0.0;
    EIS_CALL(LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n_qpp, n_qpp, sig_inv, n_qpp, sig_tmp, n_qpp));

    // pass back the results
    for (int i = 0; i < n_qpp; i++)
	mu[i] = mu_tmp[i];

    for (int i = 0; i < n_qpp*n_qpp; i++)
	sigma[i] = sig_tmp[i];

    symmetrify(sigma, n_qpp);
    
    return 0;
}


//========================================//
//                                        //
//           UTILITY FUNCTIONS            //
//                                        //
//========================================//


double calculate_mean(double* x, int n)
{
    double x_bar = 0.0;
    for (int i = 0; i < n; i++)
	x_bar += x[i];
    x_bar /= (double)n;
    return x_bar;
}


double calculate_sample_variance(double* x, int n)
{
    double s2 = 0.0;
    double mu = calculate_mean(x,n);
    for (int i = 0; i < n; i++)
	s2 += (x[i] - mu) * (x[i] - mu);
    s2 /= (double)(n-1);
    return s2;
}


int check_convergence_tmp(double* x_old, double* x_new, int n, double tol)
{
    double slope[n];
    for (int i = 0; i < n; i++)
	slope[i] = fabs((x_new[i] - x_old[i])/(x_old[i]));

    double max_slope = slope[0];

    for (int i = 1; i < n; i++)
	max_slope = (max_slope > slope[i]) ? max_slope : slope[i];

    printf("Diff: %f\n", max_slope);

    if (max_slope > tol)
	return 0;
    else
	return 1;
}


int check_convergence(double* x_old, double* x_new, int n, double tol)
{
    int isconv = 1;
    double check = fabs((x_new[0] - x_old[0])/(x_old[0]));
    if (check > tol) {
	isconv = 0;
    } else {
	for (int i = 1; i < n; i++) {
	    check = fabs((x_new[i] - x_old[i])/(x_old[i]));
	    if (check > tol) {
		isconv = 0;
		break;
	    }
	}
    }
    return isconv;
}


int ordinary_least_squares(const double* X, const double* Y, double* beta, int nx, int ny, int nobs)
{
    
    double X_[nobs * nx];
    for (int i = 0; i < nobs*nx; i++)
	X_[i] = X[i];

    double Y_[nobs * ny];
    for (int i = 0; i < nobs*ny; i++)
	Y_[i] = Y[i];
    
    EIS_CALL(LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', nobs, nx, ny, X_, nx, Y_, ny));

    for (int i = 0; i < nx; i++)
	for (int j = 0; j < ny; j++)
	    beta[ny*i + j] = Y_[ny*i + j];
    
    return 0;
}


/*
int ordinary_least_squares_old(const double* X, const double* Y, double* beta, int nx, int ny, int nobs)
{
    double XX [nx * nx];
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nx, nx, nobs,
		1.0, X, nx, X, nx, 0.0, XX, nx);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nx, ny, nobs,
		1.0, X, nx, Y, ny, 0.0, beta, ny);

    symmetrify(XX, nx);

    EIS_CALL(LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nx, XX, nx));
    EIS_CALL(LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', nx, ny, XX, nx, beta, ny));
    
    return 0;
} 
*/
