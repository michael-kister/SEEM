
#include "gpu_pf_utilities.h"


__global__ void set_states(int N, char* s_val, double* s_draw, double p_s0)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id < N)
	
	s_val[id] = (s_draw[id] <= p_s0);
}

__global__ void set_states(int N, char* s_val, double* s_draw, double p_s0_gs0, double p_s0_gs1)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    double p_s0_loc;
    
    if (id < N) {

	p_s0_loc = (s_val[id] == 0) ? p_s0_gs0 : p_s0_gs1;
	
	s_val[id] = (s_draw[id] <= p_s0_loc);
    }
}

/**
 * This function zeros out certain shocks if the test_vec doesn't match the test_value.
 * The calling function would probably claim that the shocks are in column-major, but since
 * the shocks are also in columns, they are still together for individual particles (i.e.,
 * you don't need to worry about stride).
 */
__global__ void zero_out_shocks(int N, int num_shock, double* shocks,
				const char* test_vec, char test_value)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int i, b;

    if (id < N) {

	b = test_vec[id] == test_value;

	for (i = 0; i < num_shock; i++)

	    shocks[num_shock*id + i] *= b;

    }
}


/**
 * Obtain the row-major (zero-based) index of the state, based on three states.
 * The standard formula is: n3 + N3*(n2 + N2*n1), where here we assume N1 = N2 = 2.
 */
__global__ void set_total_states(int N, char* state_all, const char* state_1,
				 const char* state_2, const char* state_3)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id < N)

	state_all[id] = state_3[id] + 2*(state_2[id] + 2*state_1[id]);
}


/**
 * This is a function to undo the mapping from indices to row-major index. So we have an array
 * with value = n3 + N3*(n2 + N2*n1), and we wish to recover n3,n2,n1. Note that we do assume
 * that each of the dimensions are 2. This function could probably be generalized into a recursive
 * function, but I don't want to do that with the GPUs. Plus, making it specific allows for some
 * optimization.
 */
__global__ void reassign_states(int N, char* state_all, char* state_1, char* state_2, char* state_3)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    char i, j;
    
    if (id < N) {

	i = state_all[id];  // i = n3 + 2*(n2 + 2*n1)

	j = i / 2;          // j = n2 + 2*n1

	state_3[id] = i - 2*j;

	state_1[id] = j / 2;
	
	state_2[id] = j - 2*state_1[id];
    }
}




__global__ void set_X_kron_X(int N, double* X_devc, double* XX_devc, int nx)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int i, j;
    
    if (id < N) {

	// perform column-major "vertical kronecker product"
	for (i = 0; i < nx; i++)
	    for (j = 0; j < nx; j++)
		XX_devc[nx*nx*id + nx*i + j] = X_devc[nx*id + i] * X_devc[nx*id + j];
	
    }
}


__global__ void add_mean_vector(int N, int n, double* X_devc, double* mu_devc, double alpha)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int i;

    if (id < N) {

	for (i = 0; i < n; i++) {
	    
	    X_devc[n*id + i] += (alpha * mu_devc[i]);
	}
    }
}


__global__ void all_dot_products(int N, double* Y_mu_devc, double* HY_mu_devc, double*W_devc,
				 double y_coeff, int ny)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int i;

    double log_prob = y_coeff;

    if (id < N) {

	for (i = 0; i < ny; i++)
	    log_prob -= (0.5 * Y_mu_devc[ny*id + i] * HY_mu_devc[ny*id + i]);

	W_devc[id] *= exp( log_prob );
    }
}


__global__ void reassign_particles(int N, int nx, int* N_devc, double* X_devc, double* Xp_devc)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int i, n;

    if (id < N) {

	n = N_devc[id];

	for (i = 0; i < nx; i++) {

	    X_devc[nx*id + i] = Xp_devc[nx*n + i];
	    
	}
    }
}


__global__ void reassign_particles(int N, int nx, int* N_devc, char* X_devc, char* Xp_devc)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int i, n;

    if (id < N) {

	n = N_devc[id];

	for (i = 0; i < nx; i++) {

	    X_devc[nx*id + i] = Xp_devc[nx*n + i];
	    
	}
    }
}


__global__ void set_weights(int N, double* W_devc, double W_set)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < N) {

	W_devc[id] = W_set;
    }
}


// STUFF FOR PARALLELIZED CUMULATIVE SUM
// (not sure how useful -- but fun to have)


__global__ void pre_scan_ADD(double* data_devc)
{
    extern __shared__ double data_shar[];

    int i_thr = threadIdx.x;
    int i_dat = blockIdx.x*blockDim.x + threadIdx.x;

    // load from global to shared memory
    data_shar[i_thr] = data_devc[i_dat];
    __syncthreads();

    // perform up-sweep
    int stride;
    int index;
    for (stride = 1; stride < blockDim.x; stride *= 2) {
	index = blockDim.x - (2 * stride * i_thr) - 1;

	if (index > 0) {
	    data_shar[index] += data_shar[index - stride];
	}
	__syncthreads();
    }

    // set last value to 0
    if (i_thr == blockDim.x - 1)
	data_shar[i_thr] = 0.0;
    __syncthreads();

    // perform down-sweep
    double temp;
    for (stride = blockDim.x/2; stride >= 1; stride /= 2) {
	index = blockDim.x - (2 * stride * i_thr) - 1;

	if (index < blockDim.x) {
	    temp = data_shar[index];
	    data_shar[index] += data_shar[index - stride];
	    data_shar[index - stride] = temp;
	}
	__syncthreads();
    }

    // transfer back to global memory
    if (i_thr < blockDim.x - 1)
	data_devc[i_dat] = data_shar[i_thr+1];
    else
	data_devc[i_dat] += data_shar[i_thr];
}


__global__ void add_block_sum(double* data_devc, double* block_data_devc)
{
    int i_dat = blockIdx.x*blockDim.x + threadIdx.x;

    data_devc[i_dat] += block_data_devc[blockIdx.x];
}


int isPow2(int N)
{
    int t = 2;

    while (t <= N)
	if (t == N)
	    return 1;
	else
	    t <<= 1;
	
    return 0;
}


int gpu_cumsum(int N, double* data_devc)
{
    // setup pertaining to device properties
    int i;
    int info = 0;

    if (isPow2(N)) {

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	int nthread = prop.maxThreadsPerBlock;
	nthread = (N < nthread) ? N : nthread;
	int nblock  = (N % nthread == 0) ? N/nthread : (N/nthread) + 1;

	pre_scan_ADD <<<nblock, nthread, nthread*sizeof(double)>>> (data_devc);
    
	// allow for the possibility of multiple blocks, which requires recursion
	if (nblock > 1) {
	    
	    double* block_data_devc;
	    cudaMalloc((void**)&block_data_devc, (nblock+1)*sizeof(double));
	    for (i = 0; i < nblock; i++)
		cudaMemcpy(block_data_devc+i+1, data_devc+nthread*(i+1)-1,
			   sizeof(double), cudaMemcpyDeviceToDevice);
	    
	    info = gpu_cumsum(nblock, block_data_devc+1);
	
	    double zero = 0.0;
	    cudaMemcpy(block_data_devc, &zero, sizeof(double), cudaMemcpyHostToDevice);
	
	    add_block_sum <<<nblock, nthread>>> (data_devc, block_data_devc);

	    cudaFree(block_data_devc);

	}
	
    } else {

	printf("N = %d is not a power of 2; use sequential cumulative sum.\n", N);

	double* data_host = (double*) malloc (N*sizeof(double));
	cudaMemcpy(data_host, data_devc, N*sizeof(double), cudaMemcpyDeviceToHost);

	for (i = 1; i < N; i++)
	    data_host[i] += data_host[i-1];

	cudaMemcpy(data_devc, data_host, N*sizeof(double), cudaMemcpyHostToDevice);
	free(data_host);
	
    }
    
    return info;
}
