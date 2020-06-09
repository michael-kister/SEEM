#ifndef __EIS_SEC_ORD__
#define __EIS_SEC_ORD__


#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

#include <mkl.h>
#include <matrices.h>
#include <mvn.h>
#include <kalman_utilities.h>


#ifndef EIS_CALL
#define EIS_CALL(x) do { if ((x) != 0) { \
	    printf("ERROR: %s : %4d (%s)\n", __FILE__, __LINE__, __func__); \
	    return 1;}} while(0)
#endif

#ifndef EIS_CALL_G
#define EIS_CALL_G(x) do { if ((x) != 0) { goto LLCALC;} } while(0)
#endif


struct mod_2nd_order
{
    double* yss;
    double* xss;
    
    double* g_lev_1;
    double* gx_lev_1;

    double* h_lev_1;
    double* hx_lev_1;

    double* g_lev_2;
    double* gx_lev_2;
    double* gxx_lev_2;

    double* h_lev_2;
    double* hx_lev_2;
    double* hxx_lev_2;

    double* Qmat;
    double* Hmat;

    mod_2nd_order(){};
    mod_2nd_order(double* yss_, double* g_lev_1_, double* gx_lev_1_,
		  double* xss_, double* h_lev_1_, double* hx_lev_1_,
		  double* g_lev_2_, double* gx_lev_2_, double* gxx_lev_2_,
		  double* h_lev_2_, double* hx_lev_2_, double* hxx_lev_2_,
		  double* Qmat_, double* Hmat_) :
	yss(yss_), g_lev_1(g_lev_1_), gx_lev_1(gx_lev_1_),
	xss(xss_), h_lev_1(h_lev_1_), hx_lev_1(hx_lev_1_),
	g_lev_2(g_lev_2_), gx_lev_2(gx_lev_2_), gxx_lev_2(gxx_lev_2_),
	h_lev_2(h_lev_2_), hx_lev_2(hx_lev_2_), hxx_lev_2(hxx_lev_2_),
	Qmat(Qmat_), Hmat(Hmat_)
    {
    }	
};

struct eis_opts
{
    int T;
    int n_x;
    int n_y;
    int n_e;
    int n_p;
    int n_q;
    int n_samp;

    eis_opts(){};
    eis_opts(int T_, int n_x_, int n_y_, int n_e_, int n_p_, int n_q_, int n_samp_) :
	T(T_), n_x(n_x_),  n_y(n_y_),  n_e(n_e_),  n_p(n_p_),  n_q(n_q_), n_samp(n_samp_)
    {
    } 
};


int ordinary_least_squares(const double* X, const double* Y, double* beta, int nx, int ny, int nobs);

int check_convergence(double* x_old, double* x_new, int n, double tol);

double calculate_mean(double*, int);

double calculate_sample_variance(double*, int);

// void mvn_transform(const double* mu, const double* sigma, double* x, int n_var, int n_samp);

/**
 * Holds the functionality of the EIS-filter for 2nd-order solution method.
 *
 * Given the output from the S-G&U solution, this class should be able to imple-
 * ment Efficient Importance Sampling to estimate the likelihood of a set of
 * parameters.
 *    Essentially there is one big loop over time periods. For each time period,
 * we enter a while-loop for convergence to an ideal importance sampler. We alt-
 * ernate between obtaining hyperparameters, and sampling from that distribu-
 * tion. The first step at each period is to obtain the initial sampler.
 *    The hyperparameters are obtained by minimizing the squared differences of
 * the log PDFs, where the independent variables are sampled from the previous
 * IS distribution.
 *    In possession of the matrices and the data, this should be self-contained,
 * other than the model-specific inversion of states. This class also makes use
 * of functions for the (extended) Kalman filter.
 */
class eis_model
{
private:

    int seed;
    
    //void map_to_beta(double*, const double*, const double*, int);
    int map_from_beta(const double*, double*, double*, int);
    
protected:
    
    virtual int invert_distribution(double*, double*, double*, double*, double*) = 0;
    
public:

    void set_seed(int seed_){
	seed = seed_;
    }
    
    eis_model(){
	seed = 9252742;
    }
    
    int eis_likelihood(mod_2nd_order*, eis_opts*, double*, double*);

};


#endif
