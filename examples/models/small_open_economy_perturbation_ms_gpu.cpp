#include "small_open_economy_perturbation_kim.h"

//========================================//
//                                        //
//        BASE MODEL SPECIFICATION        //
//                                        //
//========================================//

SOE_SGU_MSGPU::SOE_SGU_MSGPU(data Y_, short int tag_):
    SOE_SGU(Y_, tag_)
{
    printf("\t\tConstructor for 'SOE_SGU_kim.' (%p)\n", this);
    
    y_intercept = new double [ny](); // this is just y_ss
    x_intercept = new double [nx](); // just zeros
 
    mod = kim_model(gx, Hmat, hx, Rmat, Qmat, x_intercept, y_intercept, x_intercept);
    
    printf("\t\tDone.\n");
}

SOE_SGU_MSGPU::~SOE_SGU_MSGPU(void)
{
    printf("\t\tDestructing 'SOE_SGU_MSGPU' (%p)... ", this);
    delete[] y_intercept;
    delete[] x_intercept;
    printf("Done.\n");
}

SOE_SGU_MSGPU::SOE_SGU_MSGPU(const SOE_SGU_MSGPU& that):
    SOE_SGU(that.Y), mod(that.mod)
{
    printf("\t\tCopy constructor for 'SOE_SGU_MSGPU.' (%p <- %p)\n", this, &that);
    
    y_intercept = new double [ny](); // this is just y_ss
    x_intercept = new double [nx](); // just zeros

    memcpy(y_intercept, that.y_intercept, ny*sizeof(double));
    memcpy(x_intercept, that.x_intercept, nx*sizeof(double));

    printf("\t\tDone.\n\n");
}

void swap(SOE_SGU_MSGPU& first, SOE_SGU_MSGPU& second)
{
    printf("\t\tSwap idiom for 'SOE_SGU_MSGPU.' (%p <-> %p)\n", &first, &second);
    swap(static_cast<SOE_SGU&>(first), static_cast<SOE_SGU&>(second));
    
    std::swap(first.x_intercept, second.x_intercept);
    std::swap(first.y_intercept, second.y_intercept);

    swap(first.mod, second.mod);

    printf("\t\tDone.\n");
}

void SOE_SGU_MSGPU::operator()(const double* theta_unbounded)
{
    // this loads the parameters in and does the ADOL-C stuff
    SOE_SGU::operator()(theta_unbounded);
    
    // set the intercepts
    for (int i = 0; i < ny; i++)
	y_intercept[i] = y_ss[i];
}














//========================================//
//                                        //
//  MARKOV SWITCHING MODEL SPECIFICATION  //
//                                        //
//========================================//

MS_SOE_SGU_MSGPU::MS_SOE_SGU_MSGPU(data Y_):
    Y(Y_)
{
    theta_all = new double [npar_all];
    theta_ind = new double [npar];

    int num_mods_tot = 1;
    for (int i = 0; i < num_ms_vars; i++)
	num_mods_tot *= ms_var_dims[i];

    
    SOE_SGU_MSGPU tmp = SOE_SGU_MSGPU(Y_);
    mods = std::vector<SOE_SGU_MSGPU>(num_mods_tot, tmp);

    k_mods = new kim_model* [num_mods_tot];
    for (int i = 0; i < num_mods_tot; i++) {
	mods[i] = SOE_SGU_MSGPU(Y_);
	k_mods[i] = &(mods[i].mod);
    }
    
    Pmats = new double* [num_ms_vars];
    for (int i = 0; i < num_ms_vars; i++)
	Pmats[i] = new double [ms_var_dims[i] * ms_var_dims[i]];

    k_opts = kim_opts(Y_.T, nx, n_me, neps, Pmats, num_ms_vars, ms_var_dims);
}

MS_SOE_SGU_MSGPU::~MS_SOE_SGU_MSGPU(void)
{
    printf("\tDestructing 'MS_SOE_SGU_MSGPU'\n");
    delete[] theta_all;
    delete[] theta_ind;
    delete[] k_mods;

    for (int i = 0; i < num_ms_vars; i++)
	delete[] Pmats[i];
    delete[] Pmats;
    printf("\tDone.\n");
}

/**
 * We're going to assume that there are two markov states, one affecting sig_r
 * and the other affecting sig_v.
 */
double MS_SOE_SGU_MSGPU::operator()(const double* theta_in)
{
    Pmats[0][0] = 0.0 + theta_in[0]; // Pr[1|1]
    Pmats[0][1] = 1.0 - theta_in[1]; // Pr[1|2]
    Pmats[0][2] = 1.0 - theta_in[0]; // Pr[2|1]
    Pmats[0][3] = 0.0 + theta_in[1]; // Pr[2|2]

    Pmats[1][0] = 0.0 + theta_in[2]; // Pr[1|1]
    Pmats[1][1] = 1.0 - theta_in[3]; // Pr[1|2]
    Pmats[1][2] = 1.0 - theta_in[2]; // Pr[2|1]
    Pmats[1][3] = 0.0 + theta_in[3]; // Pr[2|2]
    
    for (int i = 0; i < npar; i++)
	theta_ind[i] = theta_in[6 + i];

    int i_ms[] = {10, 12};
    
    double ms_sig_r[] = {theta_in[10+6], theta_in[4]};
    double ms_sig_v[] = {theta_in[12+6], theta_in[5]};
    
    for (int i_r = 0; i_r < 2; i_r++) {
	for (int i_v = 0; i_v < 2; i_v++) {
	    theta_ind[i_ms[0]] = ms_sig_r[i_r];
	    theta_ind[i_ms[1]] = ms_sig_v[i_v];
	    mods[2*i_r+i_v](theta_ind);
	}
    }
    
    double LL_out = 0.0;
    int info = kim_filter_1st_order(k_mods, &k_opts, Y.array, &LL_out);

    int verbose = 1;
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
