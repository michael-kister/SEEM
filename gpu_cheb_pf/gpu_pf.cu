
#define NUM_STATE 6
#define NUM_MEASR 6





#define psi      parameters[0]
#define gama     parameters[1]
#define beta     parameters[2]
#define ln_P_ss  parameters[3]
#define phi_pi   parameters[4]
#define phi_y    parameters[5]
#define rho_R    parameters[6]
#define epsilon  parameters[7]
#define theta    parameters[8]
#define alpha    parameters[9]
#define zeta     parameters[10]
#define delta    parameters[11]
#define ln_A_ss  parameters[12]
#define rho_A    parameters[13]
#define sigma_A  parameters[14]
#define ln_sg_ss parameters[15]
#define rho_sg   parameters[16]
#define sigma_sg parameters[17]
#define sigma_xi parameters[18]

#define sigma_me_y parameters[19]
#define sigma_me_c parameters[20]
#define sigma_me_i parameters[21]
#define sigma_me_h parameters[22]
#define sigma_me_P parameters[23]


#undef psi
#undef gama
#undef beta
#undef ln_P_ss
#undef phi_pi
#undef phi_y
#undef rho_R
#undef epsilon
#undef theta
#undef alpha
#undef zeta
#undef delta
#undef ln_A_ss
#undef rho_A
#undef sigma_A
#undef ln_sg_ss
#undef rho_sg
#undef sigma_sg
#undef sigma_xi











#include "gpu_pf.h"
#include "gpu_pf_utilities.h"


#include "measurement_equation.cu"
 

/*------------------------------------------------------------------------------
 * This is a function to re-weight the particles based on the measurement equa-
 * tion. We will make measurement equations for:
 *
 *  - Interest rates
 *  - Output
 *  - Hours
 *  - Inflation
 *  - Capital
 *  
 *----------------------------------------------------------------------------*/
__global__ void measurement_equation
(const double* parameters, const int* i_cheb_indxs, const double* i_cheb_cxffs,
 int num_terms)
{
    // Lagged investment; requires the use of Chebyshev approximation.
    // @ ln_v_tm2
    // @ ln_R_tm2
    // @ ln_k_tm2
    // @ ln_A_tm1
    // @ ln_sg_tm1
    // @ ln_xi_tm1
    double ln_i_tm1;
    chebyshev_approximation(NUM_STATE, num_terms, x_tm1+6*id, &ln_i_tm1,
			    i_cheb_indxs, i_cheb_cxffs);
	

}

/*------------------------------------------------------------------------------
 * This is a function to estimate the likelihood of a model given a set of par-
 * ameters.
 *
 * Inputs:
 *    double** data ......... data[t][n] references series n at time t
 *    
 *  
 *----------------------------------------------------------------------------*/
extern "C" int gpu_particle_filter
(double** data, gpu_model* mod, gpu_options* opts, double* LL_out)
{
    //---------------
    // useful scalars

    int num_pxcle = 1 << 10;
    int num_state = 6;
    int num_obsxn = 
        
    int i, j, t;
    int verbose = 0;
    double alpha, beta;
    int resample = 1;
    *LL_out = 0.0;

    
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

    


    //-----------------------------------
    // allocating memory for the function
    
    double* X_devc;
    double* Xp_devc;
    cudaMalloc((void**)&X_devc,  N*nx*sizeof(double));
    cudaMalloc((void**)&Xp_devc, N*nx*sizeof(double));
    
    double* XX_devc;
    cudaMalloc((void**)&XX_devc, N*nx*nx*sizeof(double));

    double* YmMu_devc;
    double* HYmMu_devc;
    cudaMalloc((void**)&YmMu_devc, N*ny*sizeof(double));
    cudaMalloc((void**)&HYmMu_devc, N*ny*sizeof(double));

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
    double* norms_devc;
    cudaMalloc((void**)&norms_devc, N * ne * sizeof(double));

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);

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

    double* X0_host = (double*) malloc (nx * sizeof(double));
    for (i = 0; i < nx; i++)
	X0_host[i] = mod->X0[i];

    double* X0_devc;
    cudaMalloc((void**)&X0_devc, nx * sizeof(double));
    cudaMemcpy(X0_devc, X0_host, nx*sizeof(double), cudaMemcpyHostToDevice);

    double* P0_L_host = (double*) malloc (nx * nx * sizeof(double));
    for (i = 0; i < nx; i++)
	for (j = 0; j < nx; j++)
	    P0_L_host[nx*i + j] = mod->P0_L[nx*j + i];

    double* P0_L_devc;
    cudaMalloc((void**)&P0_L_devc, nx*nx*sizeof(double));
    cudaMemcpy(P0_L_devc, P0_L_host, nx*nx*sizeof(double), cudaMemcpyHostToDevice);

    // draw them, put them in Xp, multiply them into X
    curandGenerateNormalDouble(gen, Xp_devc, N*nx, 0.0, 1.0);

    alpha = 1.0;
    beta  = 0.0;
    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nx,N,nx,&alpha,
		P0_L_devc,nx, Xp_devc,nx, &beta, X_devc,nx);
    
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
	
	//----------------------------------------------------------------------
	// Generate shocks

	curandGenerateNormalDouble(gen, norms_devc, N*ne, 0.0, 1.0);	

	//----------------------------------------------------------------------
	// Input shocks + previous states to obtain current states

	//----------------------------------------------------------------------
	// Use current states to calculate observation errors

	//----------------------------------------------------------------------
	// Use observation errors to calculate reweightings


	
	alpha = 1.0;
	beta  = 0.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nx,N,nx,&alpha,
		    hx_devc,nx, X_devc,nx, &beta, Xp_devc,nx);

	
	// get standard normal draws (Big matrix multiplication)
	
	alpha = 1.0;
	beta  = 1.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nx,N,ne,&alpha,
		    RQ_L_devc,nx, norms_devc,ne, &beta, Xp_devc,nx);

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
    cudaFree(RQ_L_devc);

    cudaFree(g_devc);
    cudaFree(gx_devc);
    cudaFree(gxx_devc);
        
    cudaFree(X_devc);
    cudaFree(Xp_devc);
    cudaFree(XX_devc);

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
