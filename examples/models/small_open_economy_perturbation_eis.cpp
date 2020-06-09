#include "small_open_economy_perturbation_eis.h"

//========================================//
//                                        //
//          MODEL SPECIFICATION           //
//                                        //
//========================================//

SOE_SGU_EIS::SOE_SGU_EIS(data Y_, int nsamp_, short int tag_):
    SOE_SGU(Y_, tag_), eis_model()
{
    printf("\t\tConstructing 'SOE_SGU_EIS'\n\n");
    
    y_intercept_1 = new double [ny](); // this is just y_ss
    y_intercept_2 = new double [ny](); // this is y_ss + 0.5*gss*sigma*sigma
    x_intercept_1 = new double [nx](); // just zeros
    x_intercept_2 = new double [nx](); // just 0.5*hss*sigma*sigma
    
    mod  = mod_2nd_order(y_ss, y_intercept_1, gx, x_intercept_1, x_intercept_1, hx,
			 y_intercept_2, gx, gxx, x_intercept_2, hx, hxx,
			 Qmat, Hmat);
    opts = eis_opts(Y.T, nx, n_me, neps, n_p, n_q, nsamp_);

    printf("\t\tDone.\n\n");
}

SOE_SGU_EIS::~SOE_SGU_EIS(void)
{
    printf("\t\tDestructing 'SOE_SGU_EIS'... ");

    delete[] y_intercept_1;
    delete[] y_intercept_2;
    delete[] x_intercept_1;
    delete[] x_intercept_2;

    printf("Done.\n\n");
}

double SOE_SGU_EIS::operator()(const double* theta_unbounded, int bounded)
{
    SOE_SGU::operator()(theta_unbounded, bounded);
    
    // set the intercepts
    for (int i = 0; i < ny; i++) {
	y_intercept_1[i] = y_ss[i];
	y_intercept_2[i] = y_ss[i] + (0.5 * gss[i] * theta[8] * theta[8]);
    }
    for (int i = 0; i < nx; i++) {
	x_intercept_1[i] = 0.0;
	x_intercept_2[i] = 0.5 * hss[i] * theta[8] * theta[8];
    }

    
    double LL_out = 0.0;
    int info = eis_model::eis_likelihood(&mod, &opts, Y.array, &LL_out);

    if (verbose) {
	if (info == 0) {
	    printf("Log-likelihood: %31.5f\n", LL_out);
	} else {
	    printf("Log-likelihood: ???????????????????????????????\n");
	}
	printf("------------------------------------------------\n");
    }
    
    return (info == 0) ? -1.0*LL_out : 1.0/0.0;
}


//========================================//
//                                        //
//          INVERSION FUNCTION            //
//                                        //
//========================================//


/**
 * We have singular transitions, and we must therefore solve for q_tm1 using some
 * solution method.
 *
 * The issue here is that I've learned it for solving in terms of deviation from
 * the steady state, but now we use the particles in levels.
 *
 * It's possible that we can just switch everything to the level version, however
 * it seems safer to just make local copies, consider everything in deviations from
 * the steady state, and then set them back at the end.
 */
int SOE_SGU_EIS::invert_distribution(double* q_t, double* p_t, double* q_tm1, double* p_tm1, double* jac)
{
    int n_q = 2; // this is a model-specific file, so I don't mind 
    int n_p = 4; // hard-coding these constants in...
    
    double q_t_loc[n_q]; 
    double p_tm1_loc[n_p]; 

    for (int i = 0; i < nx; i++)
	if ( i < n_q)
	    q_t_loc[i] = q_t[i];
	else
	    p_tm1_loc[i-n_q] = p_tm1[i-n_q];
	
    // a*x*x + b*x + c = 0
    
    // first we must solve the quadratic equation for d:

    double a_d = 0.0;
    double b_d = hx[0];
    double c_d = 0.0 - q_t_loc[0];
    
    for (int i = 2; i < nx; i++)
	c_d += p_tm1_loc[i-2] * hx[i];
        
    for (int i = 0; i < nx; i++) {
	for (int j = 0; j < nx; j++) {
	    if (i == 0) {
		if (j == 0) {
		    a_d += hxx[0];
		} else if (j >= 2) {
		    b_d += hxx[j] * p_tm1_loc[j - 2];
		}
	    } else if (i >= 2) {
		if (j == 0) {
		    b_d += hxx[nx*i] * p_tm1_loc[i-2];
		} else if (j >= 2) {
		    c_d += hxx[nx*i+j] * p_tm1_loc[j-2] * p_tm1_loc[i-2];
		}
	    }
	}
    }
    q_tm1[0] = (-1.0 * b_d + sqrt(b_d*b_d - 4.0 * a_d * c_d)) / (2.0 * a_d);
    if (isnan(q_tm1[0])) {
	printf("ERROR: NaN in inversion. %s : %d\n", __FILE__, __LINE__);
	return 1;
    }

    
    // then we solve the quadratic equation for k:
    
    double a_k = 0.0;
    double b_k = hx[7];
    double c_k = 0.0 - q_t_loc[1] + hx[nx] * q_tm1[0];
    
    for (int i = 2; i < nx; i++)
	c_k += p_tm1_loc[i-2] * hx[nx+i];
	
    int h_ind;
    for (int i = 0; i < nx; i++) {
	for (int j = 0; j < nx; j++) {
	    h_ind = nx*nx + nx*i + j;
	    if (i == 0) { // you need to work with q_tm1[0]
		if (j == 0) {
		    c_k += hxx[h_ind] * q_tm1[0] * q_tm1[0];
		} else if (j == 1) {
		    b_k += hxx[h_ind] * q_tm1[0];
		} else if (j >= 2) {
		    c_k += hxx[h_ind] * q_tm1[0] * p_tm1_loc[j-2];
		}
	    } else if (i == 1) { // focus on q_tm1[1]
		if (j == 0) {
		    b_k += hxx[h_ind] * q_tm1[0];
		} else if (j == 1) {
		    a_k += hxx[h_ind];
		} else if (j >= 2) {
		    b_k += hxx[h_ind] * p_tm1_loc[j-2];
		}
	    } else { // use the rest of p_tm1
		if (j == 0) {
		    c_k += hxx[h_ind] * p_tm1_loc[i-2] * q_tm1[0];
		} else if (j == 1) {
		    b_k += hxx[h_ind] * p_tm1_loc[i-2];
		} else if (j >= 2) {
		    c_k += hxx[h_ind] * p_tm1_loc[i-2] * p_tm1_loc[j-2];
		}
	    }
	}
    }
    q_tm1[1] = (-1.0*b_k + sqrt(b_k*b_k - 4.0*a_k*c_k)) / (2.0*a_k);
    if (isnan(q_tm1[1])) {
	printf("ERROR: NaN in inversion. %s : %d\n", __FILE__, __LINE__);
	//printmat4(q_t, 1, n_q,1,1);
	//printmat4(p_t, 1, n_p,1,1);
	//printmat4(q_tm1, 1, n_q,1,1);
	//printmat4(p_tm1, 1, n_p,1,1);
	printf("\ta : %f\n\tb : %f\n\tc : %f\n", a_k, b_k, c_k);
	return 1;
    }

    // additionally, set the value of the jacobian, which is the derivative
    // of the inverse transformation w.r.t. q_t, or d_t & k_t.
    
    *jac = 1.0 / sqrt( (b_d*b_d - 4.0*a_d*c_d) * (b_k*b_k - 4.0*a_k*c_k) );
    if (isnan(jac[0])) {
	printf("ERROR: NaN in inversion. %s : %d\n", __FILE__, __LINE__);
	return 1;
    }

    return 0;
}
