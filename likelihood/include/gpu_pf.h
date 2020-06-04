#ifndef __HOST_GPU_PF__
#define __HOST_GPU_PF__

typedef struct
{
    // constants/scalars
    int nx, ny, ne;

    // matrices as determined by perturbation methods
    double *g_lev_2, *gx_lev_2, *gxx_lev_2;
    double *h_lev_1, *hx_lev_1;

    // useful matrices/scalars which would otherwise be a pain to produce
    // from within the CUDA code.
    double *RQ_L, *HmatL, *det_Hmat, *X0, *P0_L;
        
} gpu_model;

extern "C" gpu_model gpu_model_
(int nx, int ny, int ne, double* g_lev_2, double* gx_lev_2, double* gxx_lev_2, double* h_lev_1, double* hx_lev_1,
 double* RQ_L, double* HmatL, double* det_Hmat, double* X0, double* P0_L);




typedef struct
{
    // constants/scalars
    int nx, ny, ne;

    // matrices as determined by perturbation methods
    double *g_lev_2, *gx_lev_2, *gxx_lev_2;
    double *h_lev_1, *hx_lev_1;

    // useful matrices/scalars which would otherwise be a pain to produce
    // from within the CUDA code.
    double **RQ_L, *HmatL, *det_Hmat, *X0, **P0_L, **Pr_ss;
        
} ms3_gpu_model;

extern "C" ms3_gpu_model ms3_gpu_model_
(int nx, int ny, int ne, double* g_lev_2, double* gx_lev_2, double* gxx_lev_2, double* h_lev_1, double* hx_lev_1,
 double** RQ_L, double* HmatL, double* det_Hmat, double* X0, double** P0_L, double** Pr_ss);




typedef struct
{
    int T;
    int npart;
    int seed;
    
} gpu_options;

extern "C" gpu_options gpu_options_(int T, int npart, int seed);



extern "C" int gpu_particle_filter(double* data, gpu_model* mod, gpu_options* opts, double* ll_out);

extern "C" int ms3_gpu_particle_filter(double* data, ms3_gpu_model* mod, gpu_options* opts, double* LL_out);


#endif
