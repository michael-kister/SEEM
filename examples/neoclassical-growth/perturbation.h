#ifndef PERTURBATION_H
#define PERTURBATION_H

void solve_gx_hx
(double* gx, double* hx, double*** tensor, int num_control, int num_state);

void solve_gxx_hxx
(double* gxx, double* hxx, double*** tensor, int ny, int nx,
 const double* gx, const double* hx);

void solve_gss_hss
(double* gss, double* hss, double*** tensor, int num_control, int num_state, int neps,
 const double* gx, const double* gxx, const double* eta);

#endif
