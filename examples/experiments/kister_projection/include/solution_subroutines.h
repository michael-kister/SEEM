#ifndef __PROJECTION_SUBROUTINES__
#define __PROJECTION_SUBROUTINES__

// standard libraries
#include <functional>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <deque> // for fixing inverse solutions

// personal, external libraries
#include "miscellaneous.h" // for the determinant
#include "stroud.hpp"
#include "simulated_annealing.h"
#include "nonlinear_solver.h"
#include "smolyak.h"
#include "kister_perturbation.h"

// local, necessary stuff
#include "helper_structs.h"


using Graph = std::map<Point, double>;

using VecVecSmo = std::vector<ArgVec<SMOLYAK>>;

using namespace std::chrono;

#define CAP_EQN(P,ln_it,ln_ktm1) \
    (log(exp(ln_it) + (1.0-P.delta)*exp(ln_ktm1)))

#define PST_EQN(P,ln_Pt) \
    ((log(1.0-P.theta*exp((P.epsilon-1.0)*ln_Pt))-log(1.0-P.theta))/(1.0-P.epsilon))

#define SPR_EQN(P,ln_Pt,ln_Pst,ln_vtm1) \
    log((1.0-P.theta)*exp(-1.0*P.epsilon*ln_Pst) + P.theta*exp(P.epsilon*ln_Pt + ln_vtm1))

#define CON_EQN(P,ln_sgt,ln_At,ln_kt,ln_lt,ln_vt,ln_it) \
    log((1.0-exp(ln_sgt))*exp(ln_At + P.zeta*ln_kt + (1.0-P.zeta)*ln_lt - ln_vt) - exp(ln_it))

#define WAG_EQN(P,ln_ct,ln_lt) \
    (log((1.0-P.alpha)/P.alpha) + ln_ct - log(1.0-exp(ln_lt)))

#define MAR_EQN(P,ln_wt,ln_At,ln_kt,ln_lt,ln_vt) \
    (ln_wt - log(1.0-P.zeta) - ln_At - P.zeta*(ln_kt-ln_lt) + ln_vt)

#define RAT_EQN(P,ln_mct,ln_At,ln_kt,ln_lt,ln_vt) \
    (ln_mct + log(P.zeta) + ln_At + (P.zeta-1.0)*(ln_kt-ln_lt) - ln_vt)

#define OME_EQN(P,ln_ct,ln_lt) \
    ((1.0-P.psi)*(P.alpha*ln_ct + (1.0-P.alpha)*log(1.0-exp(ln_lt))) - ln_ct)

#define OUT_EQN(P,ln_ct,ln_it,ln_sgt) \
    (log(exp(ln_ct)+exp(ln_it)) - log(1.0-exp(ln_sgt)))

#define TAY_EQN(P,SS,ln_Rtm1,ln_Pt,ln_yt,ln_xit) \
    (P.rho_R*ln_Rtm1 + ln_xit + \
     (1.0-P.rho_R)*(SS.ln_R_ss + P.phi_pi*(ln_Pt-SS.ln_P_ss) + P.phi_y*(ln_yt-SS.ln_y_ss)))

#define INT_EQN(P,ln_Zt) \
    ((ln_Zt > 0.0) ? ln_Zt : 0.0)


// we really care about v, R, k, and then c & y

#define CAP_EQN_LONG(P,SS,v,R,k,A,s,xi,V,i,l,Pi,x) \
    CAP_EQN(P,i,k)

#define SPR_EQN_LONG(P,SS,v,R,k,A,s,xi,V,i,l,Pi,x) \
    SPR_EQN(P,Pi,PST_EQN(P,Pi),v)

#define CON_EQN_LONG(P,SS,v,R,k,A,s,xi,V,i,l,Pi,x) \
    CON_EQN(P,s,A,CAP_EQN(P,i,k),l,SPR_EQN(P,Pi,PST_EQN(P,Pi),v),i)

#define OUT_EQN_LONG(P,SS,v,R,k,A,s,xi,V,i,l,Pi,x) \
    OUT_EQN(P,CON_EQN(P,s,A,CAP_EQN(P,i,k),l,SPR_EQN(P,Pi,PST_EQN(P,Pi),v),i),i,s)

#define INT_EQN_LONG(P,SS,v,R,k,A,s,xi,V,i,l,Pi,x) \
    INT_EQN(P,TAY_EQN(P,SS,R,Pi,OUT_EQN(P,CON_EQN(P,s,A,CAP_EQN(P,i,k),l,SPR_EQN(P,Pi,PST_EQN(P,Pi),v),i),i,s),xi))



/*
#define SOL_CALL(x) do { if ((x) != 0) {				\
	    printf("ERROR: %s : %d (%s)\n", __FILE__, __LINE__, __func__); \
	    return 1;}} while(0)

#define VAL_CHECK_PAR(x) do { if(!isnormal(x) && x != 0.0) {    \
	    printf("ERROR: %s : %d (Thread %d)\n",              \
		   __FILE__, __LINE__, omp_get_thread_num());	\
	    goto closure_failure; }} while(0)

#define VAL_CHECK_PAR2(x) do { if(!isnormal(x) && x != 0.0) {   \
	    printf("ERROR: %s : %d (Thread %d)\n",              \
		   __FILE__, __LINE__, omp_get_thread_num());	\
	    goto inverse_closure_failure; }} while(0)
*/


//#ifndef SOL_CALL
//#define SOL_CALL(x) do { if ((x) != 0) {				\
//	    return 1;}} while(0)
//#endif


#define VAL_CHECK_PAR(x) do { if(!isnormal(x) && x != 0.0) {    \
	    goto closure_failure; }} while(0)

#define VAL_CHECK_PAR2(x) do { if(!isnormal(x) && x != 0.0) {   \
	    goto inverse_closure_failure; }} while(0)

#define MAP_SMO(x,a,b) ((2.0*(x-a)/(b-a))-1.0)

#define UNMAP(x,a,b) (((b-a)*(x+1.0)/2.0)+a)

#define SOLVERARGS (const Parameters&, const SteadyStates&, const double*, const double*, \
		    const ArgVec<SMOLYAK>&, ArgVec<TILDE>&,		                  \
		    Approximation**, int, int, int, int)




int is_converged
(const double* prev, const double* curr, int dim);

int solve_model
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB, 
 Approximation*** funs_all, Graph** graphs,
 int num_state, int num_decxn, int num_snglr, int num_meas);

int set_inverse_domain_cheby
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 Approximation** funs, int num_state, int num_snglr, int smolyak_q,
 Approximation& bound_fun, VecVecSmo& cube_points, VecVecSmo& para_points);

void transformation_setup
(const VecVecSmo& cube_points, const VecVecSmo& para_points,
 std::function<void(ArgVec<SMOLYAK>&)>& cube_to_para,
 std::function<void(ArgVec<SMOLYAK>&)>& para_to_cube);

int solve_inverse
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 Approximation** funs, Approximation** invs, Graph** graphs,
 std::function<void(ArgVec<SMOLYAK>&)>& cube_to_para,
 int num_state, int num_snglr, int smolyak_q);

int solve_at_point SOLVERARGS;
//(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
// std::vector<double>& collocation, std::vector<double>& solution,
// Approximation** funs_in, int num_state, int num_decxn, int integration_level, int verbose = 0);

int iterate_decision_rules
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 Approximation** funs_in, Approximation** funs_out, Graph** graphs, Grid& G, double*** coeffs,
 int num_state, int num_decxn, int smolyak_q, int for_R, int which_plot, int integration_level,
 int* num_iter_out);



#endif
