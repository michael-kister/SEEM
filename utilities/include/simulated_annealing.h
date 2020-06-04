#ifndef __SIM_ANNEAL_H__
#define __SIM_ANNEAL_H__

#include <functional>

#include <iostream>

#include <deque>

#include <cmath>   /* fabs */

#include <cstdlib> /* srand, rand */

#define UNIF_RAND ((double)rand() / RAND_MAX)

#define UNIF_RAND_2 ((2.0 * UNIF_RAND) - 1.0)

int val_converged
(const std::deque<double>& val_check, double fp, double tol);

int simulated_anneal
(std::function<double(const double*)> &fun, int dim, double* x_out,
 const double* lb, const double* ub, int verbose = 0);

#endif
