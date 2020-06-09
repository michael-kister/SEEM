#include "small_open_economy_perturbation_gpu.h"


SOE_SGU_GPU::SOE_SGU_GPU(data Y_, short int tag_, int nsamp_):
    SOE_SGU(Y_, tag_), nsamp(nsamp_)
{
    printf("\t\tConstructing 'SOE_SGU_GPU'\n\n");
    
    RQ_L     = new double [nx * neps]();
    HmatL    = new double [n_me * n_me]();
    det_Hmat = new double [1];
    X0       = new double [nx];
    P0_L     = new double [nx * nx];
    
    mod  = gpu_model_(nx, 4, neps, g_lev_2, gx_lev_2, gxx_lev_2,
		      h_lev_1, hx_lev_1, RQ_L, HmatL, det_Hmat, X0, P0_L);
    
    opts = gpu_options_(Y.T, nsamp_, 9252742);

    printf("\t\tDone.\n\n");
}


SOE_SGU_GPU::~SOE_SGU_GPU(void)
{
    printf("\t\tDestructing 'SOE_SGU_GPU'... ");
    delete[] RQ_L;
    delete[] HmatL;
    delete[] det_Hmat;
    delete[] X0;
    delete[] P0_L;
    printf("Done.\n\n");
}


double SOE_SGU_GPU::operator()(const double* theta_unbounded, int bounded)
{
    // this loads the parameters in and does the ADOL-C stuff
    SOE_SGU::operator()(theta_unbounded, bounded);

    // now the only task that remains is to do the stuff that the GPU routine
    // doesn't want to deal with...
    
    HmatL [n_me*0 + 0] = theta[15];
    HmatL [n_me*1 + 1] = theta[16];
    HmatL [n_me*2 + 2] = theta[17];
    HmatL [n_me*3 + 3] = theta[18];

    det_Hmat[0] = (theta[15] * theta[15]) *
	          (theta[16] * theta[16]) *
	          (theta[17] * theta[17]) *
	          (theta[18] * theta[18]);
    
    RQ_L [neps*2 + 0] = theta[8];
    RQ_L [neps*3 + 1] = theta[10];
    RQ_L [neps*4 + 2] = theta[12];
    RQ_L [neps*5 + 3] = theta[14];
    
    for (int i = 0; i < nx; i++)
	X0[i] = x_ss[i];
    
    // call the routine for the SS covariance
    int info = steady_state_covariance(nx, neps, hx_lev_1, Rmat, Qmat, P0_L);
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nx, P0_L, nx);
    if (info != 0)
	printf("WARNING: Bad initial covariance. %s: Line %d\n", __FILE__, __LINE__);
    for (int i = 0; i < nx; i++)
	for (int j = i+1; j < nx; j++)
	    P0_L[nx*i + j] = 0.0;
    
    double LL_out = 0.0;
    info = gpu_particle_filter(Y.array, &mod, &opts, &LL_out);

    if (verbose) {
	if (info == 0) {
	    printf("Log-likelihood: %31.5f\n", LL_out);
	} else {
	    printf("Log-likelihood: ???????????????????????????????\n");
	}
	printf("------------------------------------------------\n");
    }
    
    return (info == 0) ? -1.0*LL_out : 1.0e20;
}

