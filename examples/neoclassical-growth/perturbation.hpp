#ifndef PERTURBATION_HPP
#define PERTURBATION_HPP

void solve_gx_hx
(double* gx, double* hx, double*** tensor, int num_control, int num_state);

void solve_gxx_hxx
(Tensor& gxx_T, Tensor& hxx_T, double*** tensor, int ny, int nx,
 const Tensor& gx_T, const Tensor& hx_T);

void solve_gss_hss
(Tensor& gss_T, Tensor& hss_T, double*** tensor, int num_control, int num_state, int neps,
 const Tensor& gx_T, const Tensor& gxx_T, const Tensor& eta_T);

#endif
