#ifndef PERTURBATION_HPP
#define PERTURBATION_HPP

void solve_gx_hx
(double* gx, double* hx, double*** dF, int num_control, int num_state);

void solve_gxx_hxx
(Tensor& gxx_T, Tensor& hxx_T, Tensor (&dF_T)[5][5], int ny, int nx,
 const Tensor& gx_T, const Tensor& hx_T);

void solve_gss_hss
(Tensor& gss_T, Tensor& hss_T, Tensor (&dF_T)[5][5], int num_control, int num_state, int neps,
 const Tensor& gx_T, const Tensor& gxx_T, const Tensor& eta_T);

#endif
