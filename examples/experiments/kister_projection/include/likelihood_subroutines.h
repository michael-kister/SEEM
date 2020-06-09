#ifndef __LIKELIHOOD_SUBROUTINES__
#define __LIKELIHOOD_SUBROUTINES__

#include "solution_subroutines.h"

using AppVec = std::vector<Approximation>;
using DblDbl = std::function<void(const double*, double*)>;

void likelihood_setup
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 std::vector<AppVec>& J, std::vector<AppVec>& Z, std::vector<AppVec>& A,
 std::function<double(const double*)>& Jacobian,
 DblDbl& measr_eqn, DblDbl& trans_eqn, DblDbl& invrs_eqn, DblDbl& print_Zmat, DblDbl& print_Amat,
 Approximation** decxn, Approximation** measr, Approximation** endog, Approximation** invrs,
 std::function<void(ArgVec<SMOLYAK>&)>& para_to_cube,
 std::function<void(ArgVec<SMOLYAK>&)>& cube_to_para);

#endif
