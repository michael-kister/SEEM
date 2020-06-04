#ifndef __GPU_PF_UTILS__
#define __GPU_PF_UTILS__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <cuda_runtime.h>
#include <cblas.h>
#include <cusolverDn.h>

#include <omp.h>


__global__ void set_X_kron_X(int N, double* X_devc, double* XX_devc, int nx);

__global__ void add_mean_vector(int N, int n, double* X_devc, double* mu_devc, double alpha);

__global__ void all_dot_products(int N, double* Y_mu_devc, double* HY_mu_devc, double*W_devc,
				 double y_coeff, int ny);

__global__ void reassign_particles(int N, int nx, int* N_devc, double* X_devc, double* Xp_devc);

__global__ void reassign_particles(int N, int nx, int* N_devc, char* X_devc, char* Xp_devc);

__global__ void set_weights(int N, double* W_devc, double W_set);



__global__ void pre_scan_ADD(double* data_devc);

__global__ void add_block_sum(double* data_devc, double* block_data_devc);

int isPow2(int N);

int gpu_cumsum(int N, double* data_devc);

// new list for MS functions

__global__ void set_states(int N, char* s_val, double* s_draw, double p_s0);

__global__ void set_states(int N, char* s_val, double* s_draw, double p_s0_gs0, double p_s0_gs1);

__global__ void zero_out_shocks(int N, int num_shock, double* shocks,
				const char* test_vec, char test_value);

__global__ void set_total_states(int N, char* state_all, const char* state_1,
				 const char* state_2, const char* state_3);

__global__ void reassign_states(int N, char* state_all, char* state_1, char* state_2, char* state_3);

#endif
