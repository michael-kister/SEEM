
#ifndef __NONLINSOLVE__
#define __NONLINSOLVE__

#ifndef SOLVE_CALL
//#define SOLVE_CALL(x) do { if ((x) != 0) { \
//	    printf("ERROR: %s : %d (%s)\n", __FILE__, __LINE__, __func__); \
//	    return 1;}} while(0)
#define SOLVE_CALL(x) do { if ((x) != 0) { return 1; }} while(0)
#endif


#include <cmath>
#include <cstdio>
#include <cstdarg>

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <mkl.h>

// logit: [0,1] -> [-Inf,+Inf]
// invlogit: [-Inf,+Inf] -> [0,1]
#define LOGIT(x) (log( x / (1.0 - x)))
#define INVLOGIT(x) (exp(x) / (1.0 + exp(x)))

// rebound: [a,b] -> [c,d]
// bound: [-Inf,+Inf] -> [a,b]
// unbound: [a,b] -> [-Inf,+Inf]
#define REBOUND(x,a,b,c,d) (((d-c)*(x-a)/(b-a)) + c)
#define UNBOUND(x,a,b) (LOGIT(REBOUND(x,a,b,0.0,1.0)))
#define BOUND(x,a,b) (REBOUND(INVLOGIT(x),0.0,1.0,a,b))



void set_newton_parts
(std::function<void(const double*,double*)>& fun, double* x, double* y, double* J, int dim);

int newtons_method
(std::function<void(const double*,double*)>& fun, double* x, int dim, int max_iter);

int set_broyden_parts
(std::function<int(const double*,double*)>& fun, double* x, double* y, double* A, int dim, double* norm,
 double tol, int verbose = 0);

int broydens_method
(std::function<int(const double*,double*)>& fun, double* x, int dim, int max_iter,
 double tol = 1.0e-10, int verbose = 0);




#endif
