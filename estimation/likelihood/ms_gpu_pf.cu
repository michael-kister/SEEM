
#include "gpu_pf.h"
#include "gpu_pf_utilities.h"


extern "C" ms3_gpu_model ms3_gpu_model_(int nx, int ny, int ne,
				double* g_lev_2, double* gx_lev_2, double* gxx_lev_2,
				double* h_lev_1, double* hx_lev_1,
				double** RQ_L, double* HmatL, double* det_Hmat,
				double* X0, double** P0_L, double** Pr_ss)
{
    ms3_gpu_model mod;

    mod.nx = nx; 
    mod.ny = ny;
    mod.ne = ne;

    mod.g_lev_2   = g_lev_2;
    mod.gx_lev_2  = gx_lev_2;
    mod.gxx_lev_2 = gxx_lev_2;
    
    mod.h_lev_1  = h_lev_1;
    mod.hx_lev_1 = hx_lev_1;

    mod.RQ_L     = RQ_L;
    mod.HmatL    = HmatL;
    mod.det_Hmat = det_Hmat;

    mod.X0   = X0;
    mod.P0_L = P0_L;

    mod.Pr_ss = Pr_ss;
    
    return mod;
}



void printmat(const double* M, int nrow, int ncol)
{
    int i, j;
    for (i = 0; i < nrow; i++) {
	for (j = 0; j < ncol; j++)
	    printf("%+8.4f ", M[ncol*i + j]);
	printf("\n");
    }
}

void printmat(const int* M, int nrow, int ncol)
{
    int i, j;
    for (i = 0; i < nrow; i++) {
	for (j = 0; j < ncol; j++)
	    printf("%+4d ", M[ncol*i + j]);
	printf("\n");
    }
}


/**
 * This is a markov-switching version of the ordinary PF. It has 3 (independent) markov switching states,
 * which are assumed to be unrelated to the transition equation (i.e. the particles move in the same way,
 * 
 */
extern "C" int ms3_gpu_particle_filter(double* data, ms3_gpu_model* mod, gpu_options* opts, double* LL_out)
{
    
    
    //---------------
    // useful scalars
        
    int i, j, k, t;
    int verbose = 0;
    double alpha, beta;
    int resample = 1;
    *LL_out = 0.0;

    // steady state for markov switching
    double p_s0_ss_1 = mod->Pr_ss[0][0];
    double p_s0_ss_2 = mod->Pr_ss[1][0];
    double p_s0_ss_3 = mod->Pr_ss[2][0];
    
    
    //----------------------------------
    // importing parameters and matrices
    
    int T = opts->T;
    int N = opts->npart;
    
    int nx = mod->nx;
    int ny = mod->ny;
    int ne = mod->ne;

    double det_Hmat = mod->det_Hmat[0];

    double log2pi  = 1.8378770664093453390819377091;
    double y_coeff = -0.5 * (ny * log2pi + log(det_Hmat));

    
    // fill in x-intercept
    double* h_host = (double*) malloc (nx * sizeof(double));
    for (i = 0; i < nx; i++)
	h_host[i] = mod->h_lev_1[i];
    
    double* h_devc;
    cudaMalloc((void**)&h_devc, nx * sizeof(double));
    cudaMemcpy(h_devc, h_host, nx*sizeof(double), cudaMemcpyHostToDevice);

    free(h_host);
    
    // fill in trans mat. using column major (where source is row-major)
    double* hx_host = (double*) malloc (nx * nx * sizeof(double));
    for (i = 0; i < nx; i++)
	for (j = 0; j < nx; j++)
	    hx_host[nx*i + j] = mod->hx_lev_1[nx*j + i];
    
    double* hx_devc;
    cudaMalloc((void**)&hx_devc, nx * nx * sizeof(double));
    cudaMemcpy(hx_devc, hx_host, nx*nx*sizeof(double), cudaMemcpyHostToDevice);

    free(hx_host);



    /**
     * Since we assume that the Q matrix is different for each of the 8 combinations
     * of states, we must bake that into our program. My first thought is that I'll
     * explicitly make 8 matrices, and then just package their pointers. Also, we define
     * a macro to make life a little easier for ourselves. (I'd like to take a look at
     * the preprocessor output to make sure this does what I think it does.)
     */
    double  *RQ_L_host = (double*) calloc (nx * ne, sizeof(double));
    double **RQ_L_devc_lookup = (double**) malloc (8 * sizeof(double*));
    double  *RQLD0, *RQLD1, *RQLD2, *RQLD3, *RQLD4, *RQLD5, *RQLD6, *RQLD7;

    #define SET_SHOCK_VARIANCE(___index___)                                               \
    for (i = 0; i < ne; i++)                                                              \
	for (j = i; j < nx; j++)                                                          \
	    RQ_L_host[nx*i + j] = mod->RQ_L[___index___][ne*j + i];                       \
    cudaMalloc((void**)&RQLD##___index___, nx * nx * sizeof(double));                       \
    cudaMemcpy(RQLD##___index___, RQ_L_host, nx*ne*sizeof(double), cudaMemcpyHostToDevice); \
    RQ_L_devc_lookup[___index___] = RQLD##___index___;

    SET_SHOCK_VARIANCE(0)
    SET_SHOCK_VARIANCE(1)
    SET_SHOCK_VARIANCE(2)
    SET_SHOCK_VARIANCE(3)
    SET_SHOCK_VARIANCE(4)
    SET_SHOCK_VARIANCE(5)
    SET_SHOCK_VARIANCE(6)
    SET_SHOCK_VARIANCE(7)
    
    free(RQ_L_host);



    
    
    // fill in M.E. cov. mat. using column major (where source is row-major)
    double* HmatL_host = (double*) malloc (ny * ny * sizeof(double));
    for (i = 0; i < ny; i++)
	for (j = 0; j < ny; j++)
	    HmatL_host[ny*i + j] = mod->HmatL[ny*j + i];
    
    double* HmatL_devc;
    cudaMalloc((void**)&HmatL_devc, ny * ny * sizeof(double));
    cudaMemcpy(HmatL_devc, HmatL_host, ny*ny*sizeof(double), cudaMemcpyHostToDevice);

    free(HmatL_host);
    
    // fill in gx & gxx using column major (where source is row-major)
    double* g_host   = (double*) malloc (ny * sizeof(double));
    double* gx_host  = (double*) malloc (ny * nx * sizeof(double));
    double* gxx_host = (double*) malloc (ny * nx * nx * sizeof(double));

    for (i = 0; i < ny; i++)
	g_host[i] = mod->g_lev_2[i];
    
    for (i = 0; i < nx; i++)
	for (j = 0; j < ny; j++)
	    gx_host[ny*i + j] = mod->gx_lev_2[nx*j + i];

    for (i = 0; i < nx*nx; i++)
	for (j = 0; j < ny; j++)
	    gxx_host[ny*i + j] = mod->gxx_lev_2[nx*nx*j + i];
    
    double* g_devc;
    double* gx_devc;
    double* gxx_devc;

    cudaMalloc((void**)&g_devc,   ny * sizeof(double));
    cudaMalloc((void**)&gx_devc,  ny * nx * sizeof(double));
    cudaMalloc((void**)&gxx_devc, ny * nx * nx * sizeof(double));

    cudaMemcpy(g_devc,   g_host,   ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gx_devc,  gx_host,  ny * nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gxx_devc, gxx_host, ny * nx * nx * sizeof(double), cudaMemcpyHostToDevice);

    free(g_host);
    free(gx_host);
    free(gxx_host);


    //-----------------------------------
    // allocating memory for the function
    
    double* X_devc;
    double* Xp_devc;
    double* Xp_devc_tmp;
    cudaMalloc((void**)&X_devc,  N*nx*sizeof(double));
    cudaMalloc((void**)&Xp_devc, N*nx*sizeof(double));
    cudaMalloc((void**)&Xp_devc_tmp, N*nx*sizeof(double));
    
    double* XX_devc;
    cudaMalloc((void**)&XX_devc, N*nx*nx*sizeof(double));

    double* YmMu_devc;
    double* HYmMu_devc;
    cudaMalloc((void**)&YmMu_devc, N*ny*sizeof(double));
    cudaMalloc((void**)&HYmMu_devc, N*ny*sizeof(double));

    char* ms1_devc;
    char* ms2_devc;
    char* ms3_devc;
    char* ms_all_devc;
    char* ms_all_devc_tmp;
    cudaMalloc((void**)&ms1_devc, N*sizeof(char));
    cudaMalloc((void**)&ms2_devc, N*sizeof(char));
    cudaMalloc((void**)&ms3_devc, N*sizeof(char));
    cudaMalloc((void**)&ms_all_devc, N*sizeof(char));
    cudaMalloc((void**)&ms_all_devc_tmp, N*sizeof(char));
    
    double* data_devc;
    cudaMalloc((void**)&data_devc, ny*sizeof(double));
    

    //----------------------------------
    // containers specific to likelihood
    
    double* W_host  = (double*) malloc ((N+1) * sizeof(double));
    W_host[0] = 0.0;
    double* W_devc;
    cudaMalloc((void**)&W_devc,  N*sizeof(double));
    
    double LL_host = 0.0;

    double* U_host  = (double*) malloc (N * sizeof(double));

    int* N_host = (int*) malloc (N * sizeof(int));
    int* N_devc;
    cudaMalloc((void**)&N_devc, N*sizeof(int));
    
    double d_offset;

    
    //------------------
    // CUDA Organization
    
    int nthread = 128;
    int nblock  = (N % nthread == 0) ? N/nthread : (N/nthread) + 1;

    
    //-------------------------------
    // set up random number generator

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);

    double* norms_devc;     // for transition
    double* norms_devc_tmp; // (for masking transition)
    double* unif_devc;      // for states

    cudaMalloc((void**)&norms_devc, N * ne * sizeof(double));
    cudaMalloc((void**)&norms_devc_tmp, N * ne * sizeof(double));
    cudaMalloc((void**)&unif_devc, N * sizeof(double));

    // seed the GPU P-RNG
    curandSetPseudoRandomGeneratorSeed(gen, opts->seed);
    // seed the C P-RNG (for resampling)
    srand(opts->seed);

    
    //------------------
    // set up for cuBLAS

    cublasHandle_t handle;
    cusolverDnHandle_t s_handle;

    cublasCreate(&handle);
    cusolverDnCreate(&s_handle);
    
    int* info_devc;
    cudaMalloc((void**)&info_devc, sizeof(int));

    
    //--------------------------
    // set the initial particles

    // first we set the markov-switching states

    curandGenerateUniformDouble(gen, unif_devc, N);
    set_states <<<nblock, nthread>>> (N, ms1_devc, unif_devc, p_s0_ss_1);
    
    curandGenerateUniformDouble(gen, unif_devc, N);
    set_states <<<nblock, nthread>>> (N, ms2_devc, unif_devc, p_s0_ss_2);
    
    curandGenerateUniformDouble(gen, unif_devc, N);
    set_states <<<nblock, nthread>>> (N, ms3_devc, unif_devc, p_s0_ss_3);
    
    set_total_states <<<nblock, nthread>>> (N, ms_all_devc, ms1_devc, ms2_devc, ms3_devc);

    // then the initial state vectors (which we assume are the same for all models)
    
    double* X0_host = (double*) malloc (nx * sizeof(double));
    for (i = 0; i < nx; i++)
	X0_host[i] = mod->X0[i];
    double* X0_devc;
    cudaMalloc((void**)&X0_devc, nx * sizeof(double));
    cudaMemcpy(X0_devc, X0_host, nx*sizeof(double), cudaMemcpyHostToDevice);




    
    

    /**
     * This is the same type of thing as the RQ_L, except slightly different size,
     * and you need to follow it up with the classic iteration over sampling, zeroing
     * out, and adding.
     */
    double  *P0_L_host = (double*) calloc (nx * nx, sizeof(double));
    double **P0_L_devc_lookup = (double**) malloc (8 * sizeof(double*));
    double  *P0LD0, *P0LD1, *P0LD2, *P0LD3, *P0LD4, *P0LD5, *P0LD6, *P0LD7;

    #define SET_SS_COVARIANCE(___index___)                                                  \
    for (i = 0; i < nx; i++)                                                                \
	for (j = i; j < nx; j++)                                                            \
	    P0_L_host[nx*i + j] = mod->P0_L[___index___][nx*j + i];                         \
    cudaMalloc((void**)&P0LD##___index___, nx * nx * sizeof(double));                       \
    cudaMemcpy(P0LD##___index___, P0_L_host, nx*nx*sizeof(double), cudaMemcpyHostToDevice); \
    P0_L_devc_lookup[___index___] = P0LD##___index___;

    SET_SS_COVARIANCE(0)
    SET_SS_COVARIANCE(1)
    SET_SS_COVARIANCE(2)
    SET_SS_COVARIANCE(3)
    SET_SS_COVARIANCE(4)
    SET_SS_COVARIANCE(5)
    SET_SS_COVARIANCE(6)
    SET_SS_COVARIANCE(7)
    
    free(P0_L_host);

    /**
     * Note that this time we're not adding the shocks to an existing matrix, but propagating
     * a target matrix with the shocks themselves. Basically this just means that for the first
     * loop, we'll zero out the target, but that for all subsequent iterations, we wish to keep
     * it; this is why beta starts out as 0.0, and then becomes 1.0;
     */
    curandGenerateNormalDouble(gen, Xp_devc, N*nx, 0.0, 1.0);
    alpha = 1.0;
    beta  = 0.0;
    for (i = 0; i < 2; i++) {
	for (j = 0; j < 2; j++) {
	    for (k = 0; k < 2; k++) {
		cudaMemcpy(Xp_devc_tmp, Xp_devc, N*nx*sizeof(double), cudaMemcpyDeviceToDevice);
		zero_out_shocks <<<nblock, nthread>>> (N, nx, Xp_devc_tmp, ms_all_devc, k+2*(j+2*i));
		cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nx,N,nx,&alpha,
			    P0_L_devc_lookup[k+2*(j+2*i)],nx, Xp_devc_tmp,nx, &beta, X_devc,nx);
		beta = 1.0;
	    }
	}
    }
    add_mean_vector <<<nblock, nthread>>> (N, nx, X_devc, X0_devc, 1.0);





    
    // set the initial weights as well
    double W_init = 1.0;// / ((double)N);
    set_weights <<<nblock, nthread>>> (N, W_devc, W_init);

    if (verbose == 1) {
	printf("======================\n");
	printf(" Time  Log-Likelihood \n");
	printf(" ----  -------------- \n\n");
    }
	
    for (t = 0; t < T; t++) {
	
	/**
	 * The first thing we do is sample the states. Since we assume independence of markov
	 * processes, we can simply handle each state one at a time using a Bernoulli random
	 * variable. Subsequently, we set the index of the combination of states, using a row-
	 * major indexing scheme.
	 */
	curandGenerateUniformDouble(gen, unif_devc, N);
	set_states <<<nblock, nthread>>> (N, ms1_devc, unif_devc, p_s0_ss_1);
	
	curandGenerateUniformDouble(gen, unif_devc, N);
	set_states <<<nblock, nthread>>> (N, ms2_devc, unif_devc, p_s0_ss_2);
	
	curandGenerateUniformDouble(gen, unif_devc, N);
	set_states <<<nblock, nthread>>> (N, ms3_devc, unif_devc, p_s0_ss_3);
	
	set_total_states <<<nblock, nthread>>> (N, ms_all_devc, ms1_devc, ms2_devc, ms3_devc);


	//----------------------------------------------------------------------
	// sample: x(i) ~ A*x(i)

	alpha = 1.0;
	beta  = 0.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nx,N,nx,&alpha,
		    hx_devc,nx, X_devc,nx, &beta, Xp_devc,nx);

	// get standard normal draws (Big matrix multiplication)
	curandGenerateNormalDouble(gen, norms_devc, N*ne, 0.0, 1.0);
	
	/**
	 * This is the part that gets affected by the markov-switching variances,
	 * since the shocks can be of 8 different types. I'm thinking we make a copy
	 * of the shocks, and then we zero out different ones for each of the state
	 * combinations, and then just call the normal multiplication
	 */
	alpha = 1.0;
	beta  = 1.0;
	for (i = 0; i < 2; i++) {
	    for (j = 0; j < 2; j++) {
		for (k = 0; k < 2; k++) {
		    cudaMemcpy(norms_devc_tmp, norms_devc, N*ne*sizeof(double), cudaMemcpyDeviceToDevice);
		    zero_out_shocks <<<nblock, nthread>>> (N, ne, norms_devc_tmp, ms_all_devc, k+2*(j+2*i));
		    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nx,N,ne,&alpha,
				RQ_L_devc_lookup[k+2*(j+2*i)],nx, norms_devc,ne, &beta, Xp_devc,nx);
		}
	    }
	}
	add_mean_vector <<<nblock, nthread>>> (N, nx, Xp_devc, h_devc, 1.0);


	//----------------------------------------------------------------------
	// calculate weights using f(yt | xt)

	// obtain second order portions, which means making a huge matrix...
	set_X_kron_X <<<nblock, nthread>>> (N, Xp_devc, XX_devc, nx);
	
	alpha = 0.5;
	beta  = 0.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, ny,N,nx*nx,&alpha,
		    gxx_devc,ny, XX_devc,nx*nx, &beta, YmMu_devc,ny);

	alpha = 1.0;
	beta  = 1.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, ny,N,nx,&alpha,
		    gx_devc,ny, Xp_devc,nx, &beta, YmMu_devc,ny);

	add_mean_vector <<<nblock, nthread>>> (N, ny, YmMu_devc, g_devc, 1);

	cudaMemcpy(data_devc, data+(ny*t), ny*sizeof(double), cudaMemcpyHostToDevice);

	add_mean_vector <<<nblock, nthread>>> (N, ny, YmMu_devc, data_devc, -1.0);
	
        // now that we have the means, we need to make a copy and then do the product
	cudaMemcpy(HYmMu_devc, YmMu_devc, N*ny*sizeof(double), cudaMemcpyDeviceToDevice);
	
	cusolverDnDpotrs(s_handle, CUBLAS_FILL_MODE_LOWER, ny,N,
			 HmatL_devc,ny, HYmMu_devc,ny, info_devc);

	// painful dot products, update the weights
	all_dot_products <<<nblock, nthread>>> (N, YmMu_devc, HYmMu_devc, W_devc, y_coeff, ny);

	//----------------------------------------------------------------------
	// work with the weights
	
	// compute likelihood (all positive, so use sum(abs()) )
	cublasDasum(handle, N, W_devc, 1, &LL_host);

	LL_host /= ((double)N);
	
	// obtain the likelihood
	if (verbose == 1) {
	    printf(" %04d  %14.2f \n", t, log(LL_host));
	}
	
	*LL_out += log( LL_host );
	LL_host  = 1.0 / LL_host;

	// scale weights by the likelihood
	cublasDscal(handle, N, &LL_host, W_devc, 1); // weights normalized
	
	if (resample == 0) {
	    
	    cudaMemcpy(X_devc, Xp_devc, N*nx*sizeof(double), cudaMemcpyDeviceToDevice);
	    
	} else {
	    
	    // compute ESS
	    cublasDnrm2(handle, N, W_devc, 1, &LL_host);
	    LL_host *= LL_host;

	    if (LL_host <= (N * 32.0)) {
		
		cudaMemcpy(X_devc, Xp_devc, N*nx*sizeof(double), cudaMemcpyDeviceToDevice); 
	    
	    } else {
		
		d_offset = (double)rand() / RAND_MAX;
		
		for (i = 0; i < N; i++)
		    U_host[i] = (d_offset + (double)i) / ((double)N);

		
		LL_host = 1.0 / ((double)N);
		cublasDscal(handle, N, &LL_host, W_devc, 1); // weights normalized
		gpu_cumsum(N, W_devc);

		cudaMemcpy(W_host+1, W_devc, N*sizeof(double), cudaMemcpyDeviceToHost);
		
		j = 0;
		for (i = 0; i < N; i++) {
		    while (U_host[i] >= W_host[j+1])
			j++;
		    
		    N_host[i] = j;
		}
		
		cudaMemcpy(N_devc, N_host, N*sizeof(int), cudaMemcpyHostToDevice);
	    
		reassign_particles <<<nblock, nthread>>> (N, nx, N_devc, X_devc, Xp_devc);

		/**
		 * You must also reassign the values for the particles, which we'll do by having
		 * one extra array that gets sampled into, and then gives the info to the different
		 * states. This is basically because we do in-place updating of the markov states,
		 * so if we reassigned all three, then we would need to reassign three times, and
		 * then copy back three times. Idk maybe this would be better.
		 */
		reassign_particles <<<nblock, nthread>>> (N, 1, N_devc, ms_all_devc_tmp, ms_all_devc);
		reassign_states <<<nblock, nthread>>> (N, ms_all_devc_tmp, ms1_devc, ms2_devc, ms3_devc);
		
		// lastly, reset the weights
		set_weights <<<nblock, nthread>>> (N, W_devc, W_init);
	    }
	}
    }

    if (verbose == 1) {
	printf("======================\n");
    }
    
    free(W_host);
    free(U_host);
    free(N_host);
    
    cudaFree(hx_devc);
    cudaFree(HmatL_devc);
    cudaFree(RQLD0);
    cudaFree(RQLD1);
    cudaFree(RQLD2);
    cudaFree(RQLD3);
    cudaFree(RQLD4);
    cudaFree(RQLD5);
    cudaFree(RQLD6);
    cudaFree(RQLD7);

    cudaFree(P0LD0);
    cudaFree(P0LD1);
    cudaFree(P0LD2);
    cudaFree(P0LD3);
    cudaFree(P0LD4);
    cudaFree(P0LD5);
    cudaFree(P0LD6);
    cudaFree(P0LD7);
    
    cudaFree(g_devc);
    cudaFree(gx_devc);
    cudaFree(gxx_devc);
        
    cudaFree(X_devc);
    cudaFree(Xp_devc);
    cudaFree(Xp_devc_tmp);
    cudaFree(XX_devc);

    cudaFree(ms1_devc);
    cudaFree(ms2_devc);
    cudaFree(ms3_devc);
    cudaFree(ms_all_devc);
    cudaFree(ms_all_devc_tmp);

    cudaFree(YmMu_devc);
    cudaFree(HYmMu_devc);
    cudaFree(data_devc);
    
    cudaFree(N_devc);
    cudaFree(W_devc);

    cudaFree(norms_devc);

    return 0;
}



	/*printf("Device Number: %d\n\n", i);
	printf("  Device name: %s\n", prop.name);
	printf("  Compute capability: %d.%d\n\n", prop.major, prop.minor);
	printf("  Total global memory: %lu bytes\n", prop.totalGlobalMem);
	printf("  Max shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
	printf("    Number of doubles: %lu\n", prop.sharedMemPerBlock/8);
	printf("  Max 32-bit registers per block: %d\n", prop.regsPerBlock);
	printf("  Warp size: %d\n\n", prop.warpSize);
	printf("  Max threads / block: %d\n", prop.maxThreadsPerBlock);
	printf("  Max threads dim: [%d %d %d]\n",prop.maxThreadsDim[0],
	                                         prop.maxThreadsDim[1],
	                                         prop.maxThreadsDim[2]);
	printf("  Number of multiprocessors: %d\n\n", prop.multiProcessorCount);*/


/*

  This was the matlab version of the systematic resampler:

function I = resample(W)
% This is a function to resample from the weights W using the
% systematic resampling approach.
    
    N = length(W);
    U1 = rand/N;
    U = U1 + ([1:N]-1)/N;
    W2 = cumsum(W);
    W1 = [0; W2(1:end-1)];
    Ns = NaN(N,1);
    for i = 1:N
        Ns(i) = length(find(U >= W1(i) & U <= W2(i)));
    end
    I = NaN(N,1);
    k = 1;
    for i = 1:N
        for j = 1:Ns(i)
            I(k) = i;
            k = k + 1;
        end
    end
end

*/
