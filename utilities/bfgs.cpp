
#include "bfgs.h"

void perturb_direction(double* dir, int n)
{
    printf("    *** Perturbing search direction.\n");
    
    for (int i = 0; i < n; i++) {

	dir[i] *= (1.0 + 0.2 * (((double)rand() / RAND_MAX) - 0.5));
    }
}

int bfgs_optimize(fun_ref &fun, double* x0, double* f0, double* Hmat, int n)
{
    /*
       The main idea at each loop is to obtain a direction, and perform a line
       search. This can be done with only a gradient and an inverse hessian.

       If the line search obtains the same value as before, then we've converged,
       and there is no reason to move onwards. If the line search gives us a
       better location, then let's try again. In that case, we want to update
       our inverse hessian.

       Additionally, there are some circumstances in which we might want to go
       back to the previous location; namely, if the new hessian is not positive
       definite.
          1.  If we arrived somewhere non-PD, then we try going straight down
              the hill (opposite of gradient). Comparable to only using first-
              order taylor expansion.
	  2.  If that doesn't work, then we do the same thing, but less far.
	      This can be iterated.
    */

    printf("========================================================\n");
    printf(" Starting Broyden-Fletcher-Goldfarb-Shanno Optimization\n");
    
    // constants
    int    max_iter = 1000;
    double tol      = .0001;   // tolerance with convergence

    // checks
    int fin     = 0; // check for completion
    int info    = 0; // testing for PD
    int attempt = 1; // attempt for iteration of BFGS algorithm
    
    srand(9252742);
    
    // function values
    double f_prev = *f0;
    double f_curr;

    // parameter containers
    double x_prev[n];
    for (int i = 0; i < n; i++)
	x_prev[i] = x0[i];
    double x_curr[n];
    
    // initialize the gradient
    double dx_prev[n];
    double dx_curr[n];
    double dx_temp[n];

    if (BFGS_VERBOSE >= 2)
	printf("\n  Perform initial gradient calculation\n");

    // grab the stuff for the 0th iteration
    numerical_gradient_1(fun, x_prev, f_prev, dx_prev, n);
    
    // initialize the direction in which to perform
    double dir[n];
    double norm_dir;

    // useful containers for BFGS algorithm
    double u_vec[n];
    double v_vec[n];
    double h_val;
    
    int iter = 0;

    while (fin == 0) {

	//----------------------------------------------------------------------
	// is this 
	if (attempt > 2) {
	    
	    // same direction, less ambitious
	    perturb_direction(dir, n);
	    
	} else if (attempt > 1) {

	    // use first-order direction
	    printf("    *** attempting to use first-order direction.\n");
	    
	    norm_dir = vector_norm(dx_prev, n);
	    norm_dir = (norm_dir > 1.0) ? norm_dir : 1.0;
	    for (int i = 0; i < n; i++)
		dir[i] = -1.0 * dx_prev[i] / norm_dir;

	} else if (attempt == 1) {

	    // (STANDARD OPTION) use second-order direction
	    cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, n,1,
			-1.0,Hmat,n, dx_prev,1, 0.0,dir,1);
	    
	    norm_dir = vector_norm(dir, n);
	    if (norm_dir > 1.0)
		for (int i = 0; i < n; i++)
		    dir[i] /= norm_dir;
	    
	} else {

	    printf("ERROR: 'attempt' must be a positive integer. (%s : %d)\n",
		   __FILE__, __LINE__);
	}
	attempt++;
	
	// set up the initial spots
	for (int i = 0; i < n; i++) {
	    dx_curr[i] = dx_prev[i];
	    x_curr[i] = x_prev[i];
	}
	f_curr = f_prev;
	
	// perform the line search in the above-obtained direction
	info = line_search(fun, x_curr, &f_curr, dir, dx_curr, n);

	for (int i = 0; i < n; i++)
	    printf("theta[%d] = %f\n", i, x_curr[i]);

	if (info != 0) {
	    printf("Line search unsuccessful: ");
	    continue;
	}
	printf("Iteration: %03d.......... Value: %+012.6f\n", iter, -1.0 * f_curr);
	
	// decide whether we're done
	if (iter++ > max_iter) {
	    printf("Reached max iterations.\n");
	    fin = 1;
	} else if (iter > n) {
	    fin = check_convergence(f_prev, f_curr, tol);
	}
	
	
	if (fin == 1) {
	    printf("\nConvergence reached after %d iterations.\n", iter);
	    printf("Final minimum value: %12.6f\n", f_curr);
	    printf("========================================================\n");
	}
	
	// if we're not done, we have much work to do!
	if (fin == 0) {

	    if (BFGS_VERBOSE >= 2)
		printf("\n  Perform subsequent gradient calculation\n");
	    numerical_gradient_1(fun, x_curr, f_curr, dx_curr, n);
	    
	    // calculate objects for BFGS update
	    h_val = 0.0;
	    for (int i = 0; i < n; i++) {
		u_vec[i] = dx_curr[i] - dx_prev[i];
		v_vec[i] = x_curr[i]  - x_prev[i];
		h_val   += u_vec[i]   * v_vec[i];
	    }

	    if (h_val < 0)
		printf("!!! Negative inner product : expect bad updated matrix.\n");
	    
	    // we're really more interested in its inverse
	    h_val = 1.0 / h_val;
	    
	    info = update_Hmat(Hmat, u_vec, v_vec, h_val, n);

	    if (info == 0) {

		// store the old gradient and calculate the new one
		for (int i = 0; i < n; i++) {
		    dx_prev[i] = dx_curr[i];
		    x_prev[i] = x_curr[i];
		}

		f_prev = f_curr;

		attempt = 1;

	    }
	}
    }

    *f0 = f_curr;
    for (int i = 0; i < n; i++)
	x0[i] = x_curr[i];
    
    printmat4(Hmat, n,n,1,1);
    return 0;
}




/**
 * Output key for line search routine:
 *
 *  0 - the execution is successful
 *  1 - reached maximum iterations without satisfactory gain
 *  2 - couldn't even get a hold on a reasonable value
 */

#define SET_XP for (int i = 0; i < n; i++) { xp[i] = x0[i] + (lam1 * dir[i]); }
#define CHECK_FP if ((fp1 - (*f0)) < (alpha*lam1*gdp)) { goto SUCCESS; }

int line_search(fun_ref &fun, double* x0, double* f0, double* dir, double* grad, int n)
{
    // perform a line search for a sufficiently small value of a function,
    // looking in the direction "dir," by a scale of "lambda."
    if (BFGS_VERBOSE >= 1)
	printf("  Starting line search\n");
    
    int max_iter = 5;
    int iter     = 0;
    int fin      = 0;
    double alpha = 0.0001;
    
    double gdp = 0.0; // gradient(g) dot(d) direction(p)
    for (int i = 0; i < n; i++)
	gdp += grad[i] * dir[i];

    if (gdp > 0.0)
	printf("!!! Strange direction... line %d\n", __LINE__);
    
    // scaling factors
    double lam1 = 10.0; // scaling factor
    double lam2 = 1.0; // scaling factor

    // attempts
    double xp[n];  // x-prime : possible value of x
    double fp1;    // f-prime : possible value of f (most recent)
    double fp2;    // f-prime : possible value of f (second most recent)    

    // scalars for use in cubic approximation
    double g1, g2, a_cub, b_cub;

    
    do {

	if (iter % 2 == 0)
	    lam1 /= 10.0;
	else
	    perturb_direction(dir, n);
	
	if (BFGS_VERBOSE >= 1)
	    printf("    > Linear approx...... λ = %8.6f : ", lam1);

	SET_XP;
	
	fp1 = fun(xp);
	if (BFGS_VERBOSE >= 1)
	    printf("%12.6f\n", fp1);

	if (iter++ > max_iter)
	    return 2;

    } while (fp1 >= TOO_BIG);
    
    CHECK_FP;
    
    // attempt quadratic approximation using new information
    fp2  = fp1;
    lam1 = -1.0 * gdp / (2.0*(fp2 - *f0 - gdp));
    
    if (BFGS_VERBOSE >= 1)
	printf("    > Quadratic approx... λ = %8.6f : ", lam1);

    SET_XP;

    fp1 = fun(xp);
    if (BFGS_VERBOSE >= 1)
	printf("%12.6f\n", fp1);
    
    CHECK_FP;

    // attempt to use cubic approximation
    while (fin == 0) {

	g1 = fp1 - *f0 - gdp*lam1;
	g2 = fp2 - *f0 - gdp*lam2;

	a_cub = ((g1/(lam1*lam1)) - (g2/(lam2*lam2))) / (lam1-lam2);
	b_cub = ((-1.0*g1*lam2/(lam1*lam1)) + (g2*lam1/(lam2*lam2))) / (lam1-lam2);

	lam2 = lam1;
	lam1 = (-1.0*b_cub + sqrt(b_cub*b_cub - 3.0*a_cub*gdp)) / (3.0*a_cub);

	if (BFGS_VERBOSE >= 1)
	    printf("    > Cubic approx....... λ = %8.6f : ", lam1);
	
	LAM_CHECK(lam1);
	
	fp2 = fp1;

	SET_XP;
	
	fp1 = fun(xp);
	if (BFGS_VERBOSE >= 1)
	    printf("%12.6f\n", fp1);

	CHECK_FP;

	// if we're at max iterations, we'll settle for any ol' decrease
	if (iter++ > max_iter)
	    if (fp1 < *f0)
		goto SUCCESS;
	    else
		return 1;
	
    }

    SUCCESS:
    for (int i = 0; i < n; i++)
	x0[i] = xp[i];
    
    *f0 = fp1;
    
    return 0;
}



double vector_norm(const double* vec, int n)
{
    double norm = 0.0;
    for (int i = 0; i < n; i++)
	norm += (vec[i] * vec[i]);

    return sqrt( norm );
}


int update_Hmat(double* Hmat, double* u_vec, double* v_vec, double inv_h, int n)
{
    if (BFGS_VERBOSE >= 1)
	printf("  Updating Hessian matrix\n");
    
    // duplicate of H matrix
    double Hloc[n * n];
    double Htmp[n * n];
    for (int i = 0; i < n*n; i++)
	Hloc[i] = Hmat[i];
    
    // useful matrix
    double uvp[n * n];
    double Huvp[n * n];
    
    // make u*v'
    for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	    uvp[n*i + j] = u_vec[i] * v_vec[j];
	    
    // make H*u*v'/h
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, n,n,
		inv_h,Hloc,n, uvp,n, 0.0,Huvp,n);

    // H += (u*v')'*(H*u*v'/h)/h
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n,n,n,
		inv_h,uvp,n, Huvp,n, 1.0,Hloc,n);

    // H -= ( H*u*v'/h + v*u'*H'/h )
    for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	    Hloc[n*i + j] -= (Huvp[n*i + j] + Huvp[n*j + i]);
	    
    // H += v*v'/h
    for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	    Hloc[n*i + j] += (v_vec[i] * v_vec[j]) * inv_h;

    // test for positive definite using temporary copy
    for (int i = 0; i < n*n; i++)
	Htmp[i] = Hloc[i];
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, Htmp, n);

    if (info == 0) {
	
	// if it worked, give us back the good one
	for (int i = 0; i < n*n; i++)
	    Hmat[i] = Hloc[i];
	
	return 0;

    } else {

	printf("!!! Update non-positive definite... discarding.\n");

	return 1;
    }
}


int check_convergence(double prev, double curr, double tol)
{
    // test whether the size of the change is small, relative to the average
    // magnitude of the values themselves.

    double eps   = 1.0e-16;
    double delta = fabs(curr - prev);
    double mu    = (fabs(prev) + fabs(curr) + eps) / 2;

    delta /= mu;
    
    if ( delta < tol )
	return 1;
    else
	return 0;
}


void numerical_gradient_2(fun_ref &fun, double* x0, double* dx, int n)
{
    double eps = 1.0e-5;
    double f_left;
    double f_right;

    for (int i = 0; i < n; i++) {

	// evaluate 'left' of initial spot
	x0[i]  -= eps;
	f_left  = fun(x0);

	// evaluate 'right' of initial spot
	x0[i]  += eps + eps;
	f_right = fun(x0);

	// set vector back to original location
	x0[i]  -= eps;

	// calculate slope
	dx[i] = (f_right - f_left) / (2 * eps);

	if (BFGS_VERBOSE >= 2)
	    printf("    Gradient (type 2) : %03d : %12.6f\n", i, dx[i]);
    }
    if (BFGS_VERBOSE >= 2)
	printf("\n");
}


void numerical_hessian_2(fun_ref &fun, double* x0, double* Amat, int n)
{
    // the matrix is going to enter as the inverse of the hessian, but is going
    // to leave as the hessian itself...

    printf("Starting numerical hessian computation (two-sided)...\n");

    double eps = 1.0e-5;
    double f_11, f_12, f_21, f_22;
    
    for (int i = 0; i < n; i++) {

	for (int j = 0; j <= i; j++) {

	    // it's not very elegant, but the computational effort is minimal,
	    // and this makes it very clear that there are no errors.

	    x0[i] -= eps; x0[j] -= eps;
	    f_11 = fun(x0);
	    x0[i] += eps; x0[j] += eps;

	    x0[i] -= eps; x0[j] += eps;
	    f_12 = fun(x0);
	    x0[i] += eps; x0[j] -= eps;

	    x0[i] += eps; x0[j] -= eps;
	    f_21 = fun(x0);
	    x0[i] -= eps; x0[j] += eps;

	    x0[i] -= eps; x0[j] -= eps;
	    f_22 = fun(x0);
	    x0[i] -= eps; x0[j] -= eps;

	    Amat[n*i + j] = (f_22 - f_21 - f_12 + f_11) / (4.0 * eps * eps);

	    if (BFGS_VERBOSE >= 2)
		printf("  (%2d,%2d) : %12.6f\n", i, j, Amat[n*i + j]);

	    if (i != j)
		Amat[n*j + i] = Amat[n*i + j];
	}
    }
}


void numerical_gradient_1(fun_ref &fun, double* x0, double f0, double* dx, int n)
{
    double eps = 1.0e-5;
    double f_right;

    for (int i = 0; i < n; i++) {

	// evaluate 'right' of initial spot
	x0[i]  += eps;
	f_right = fun(x0);

	// set vector back to original location
	x0[i]  -= eps;

	// calculate slope
	dx[i] = (f_right - f0) / eps;

	if (BFGS_VERBOSE >= 2)
	    printf("    Gradient (type 1) : %03d : %12.6f\n", i, dx[i]);
    }
    if (BFGS_VERBOSE >= 2)
	printf("\n");    
}
