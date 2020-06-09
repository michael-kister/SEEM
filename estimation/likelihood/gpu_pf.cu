
#include "gpu_pf.h"
#include "gpu_pf_utilities.h"


extern "C" gpu_options gpu_options_(int T, int npart, int seed)
{
    gpu_options opt;

    opt.T     = T;
    opt.npart = npart;
    opt.seed  = seed;

    return opt;
}


extern "C" gpu_model gpu_model_
(int nx, int ny, int ne, double* g_lev_2, double* gx_lev_2, double* gxx_lev_2, double* h_lev_1, double* hx_lev_1,
 double* RQ_L, double* HmatL, double* det_Hmat,double* X0, double* P0_L)
{
    gpu_model mod;

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

    return mod;
}

/*

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
*/

extern "C" int gpu_particle_filter(double* data, gpu_model* mod, gpu_options* opts, double* LL_out)
{
    //---------------
    // useful scalars
        
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

//printmat(mod->hx_lev_1, nx, nx);
    
    double* hx_devc;
    cudaMalloc((void**)&hx_devc, nx * nx * sizeof(double));
    cudaMemcpy(hx_devc, hx_host, nx*nx*sizeof(double), cudaMemcpyHostToDevice);

    free(hx_host);
    
    // fill in lower cholesky decomposition of RQR' matrix
    // (technically, R * chol(Q), but it's what scales the draws)
    double* RQ_L_host = (double*) calloc (nx * ne, sizeof(double));
    for (i = 0; i < ne; i++)
	for (j = i; j < nx; j++)
	    RQ_L_host[nx*i + j] = mod->RQ_L[ne*j + i];

    double* RQ_L_devc;
    cudaMalloc((void**)&RQ_L_devc, nx * nx * sizeof(double));
    cudaMemcpy(RQ_L_devc, RQ_L_host, nx*ne*sizeof(double), cudaMemcpyHostToDevice);

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
	// sample: x(i) ~ A*x(i)

	alpha = 1.0;
	beta  = 0.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nx,N,nx,&alpha,
		    hx_devc,nx, X_devc,nx, &beta, Xp_devc,nx);

	
	// get standard normal draws (Big matrix multiplication)
	curandGenerateNormalDouble(gen, norms_devc, N*ne, 0.0, 1.0);	
	
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
