
#include "nonlinear_solver.h"
#include "matrices.h"



/**
 * Citation: "Numerical Methods for Solving Systems of Nonlinear Equations,"
 *           Courtney Remani, 2012
 *
 * This is a function to implement Broyden's method, which avoids taking numer-
 * ical derivatives (and solving the linear system of equations) at each iter-
 * ation. Instead, we directly iterate over J using the following equation:
 *
 *     J1 = J0 + (y1 - J0*s1)*s'/(|s|*|s|),
 *
 * where s(i) = x(i) - x(i-1). We write A = inv(J), whereupon the Sherman-
 * Morrisson formula gives us that:
 *
 *     A1 = inv( J0 + (y1 - J0*s1)*s1'/(|s1|*|s1|) )
 *
 *        = A0 + (s - A0*y)*s'*A0/(s'*A0*y),
 *
 * whereupon we obtain:
 *
 *     x1 = x0 - A*y1.
 */
int set_broyden_parts
(std::function<int(const double*,double*)>& fun, double* x, double* y, double* A, int dim, double* norm,
 double tol, int verbose)
{
    lapack_int ipiv[dim];
    
    double yp[dim];
    
    double eps = 1.0e-4;

    SOLVE_CALL(fun(x, y));

    // check for any NaN
    for (int i = 0; i < dim; i++)
	if (!isnormal(y[i]) && y[i] != 0.0)
	    return 1;

    // obtain the norm
    norm[0] = 0.0;
    for (int i = 0; i < dim; i++)
	norm[0] += (y[i]*y[i]);

    // check if norm is small
    if (norm[0] < tol)
	return 0;

    for (int i = 0; i < dim; i++) {

	x[i] += eps;

	SOLVE_CALL(fun(x, yp));

	for (int j = 0; j < dim; j++)

	    A[dim*j+i] = (yp[j] - y[j]) / eps;
	
	x[i] -= eps;
    }

if (verbose)
    printmat4(A, dim, dim, 1, 1);

    SOLVE_CALL(LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, A, dim, ipiv));

    SOLVE_CALL(LAPACKE_dgetri(LAPACK_ROW_MAJOR, dim, A, dim, ipiv));

if (verbose)
    printmat4(A, dim, dim, 1, 1);

    //for (int i = 0; i < dim*dim; i++)
	//if (A[i] > 100.0 || A[i] < -100.0)
	    //return 1;
    
    return 0;
}

int broydens_method
(std::function<int(const double*,double*)>& fun, double* x0, int dim, int max_iter,
 double tol, int verbose)
{
    // for stopping criteria
    double norm;

    double y[dim];
    double yl[dim];
    double dy[dim];
    double y_tmp[dim];

    double A[dim * dim];

    double x[dim];
    double xl[dim];
    for (int i = 0; i < dim; i++)
	xl[i] = x[i] = x0[i];
    double dx[dim];
    double dxp[dim];

    double dxAdy = 0.0;

    // set "previous" F(X) and obtain inverse Jacobian
    SOLVE_CALL(set_broyden_parts(fun, xl, yl, A, dim, &norm, tol, verbose));
    
    if (norm < tol) {
	for (int i = 0; i < dim; i++)
	    x0[i] = x[i];
	return 0;
    }
    
if (verbose)
    printmat4(yl, dim,1,1,1);
if (verbose)
    printmat4(x, dim,1,1,1);

    // obtain updated X using Newton's formula
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, 1, dim,
		-1.0, A, dim, yl, 1, 1.0, x, 1);

if (verbose)
    printmat4(x, dim,1,1,1);

    for (int iter = 0; iter < max_iter; iter++) {

	// compute the most recent F(X)
	SOLVE_CALL(fun(x,y_tmp));

	// if it didn't fail, pass it back
	for (int i = 0; i < dim; i++)
	    y[i] = y_tmp[i];
	
	// check whether we've solved the model
	norm = 0.0;
	for (int i = 0; i < dim; i++)
	    norm += (y[i]*y[i]);
	if (norm < tol) {
	    for (int i = 0; i < dim; i++)
		x0[i] = x[i];
	    return 0;
	}
	
	// obtain differences
	for (int i = 0; i < dim; i++) {
	    dx[i] = x[i] - xl[i];
	    dy[i] = y[i] - yl[i];
	}
	
	// send the lagged values up to the current position
	for (int i = 0; i < dim; i++) {
	    xl[i] = x[i];
	    yl[i] = y[i];
	}
	
	// dxAdy = dx'*A0*dy
	dxAdy = 0.0;
	for (int i = 0; i < dim; i++)
	    for (int j = 0; j < dim; j++)
		dxAdy += dx[i] * A[dim*i+j] * dy[j];

	// dxp = dx'*A
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, dim, dim,
		    1.0, dx, dim, A, dim, 0.0, dxp, dim);

	// dx -= A*dy
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, 1, dim,
		    -1.0, A, dim, dy, 1, 1.0, dx, 1);
	
	// A += dx*dxp/(dxAdy)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, 1,
		    1.0/dxAdy, dx, 1, dxp, dim, 1.0, A, dim);
	
	// x -= A*y
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, 1, dim,
		    -1.0, A, dim, y, 1, 1.0, x, 1);
	
    }

    /*
    printf("Reached maximum iterations. Obtained: [");
    for (int i = 0; i < dim; i++)
	printf("%10.3e ", y[i]);
    printf("\b]\n");
    */
    return 1;
}


/**
 * Citation: "Numerical Methods for Solving Systems of Nonlinear Equations,"
 *           Courtney Remani, 2012
 * 
 * This is a function to implement Newton's method. Namely, if in one dimension:
 *
 *     f(x) ~ f(a) + (x - a)*∇f + .5*(x - a)'*H*(x - a)
 *
 *          = .5*x'*H*x + b'*x + c
 *
 * where:
 *
 *     b = ∇f - H*a, and
 *
 *     c = f(a) - a*∇f + 0.5*a'*H*a,
 *
 * then:
 *  
 *     f'(x) ~ H*x + b,
 *
 * which implies that f'(x) = 0 when x = -H\b. Therefore,
 *
 *     x = -H\(∇f - H*a) = a - H\∇f.
 *
 * We can generalize this to functions F(X) in more than one dimension with the
 * following equation:
 *
 *     X1 = X0 - J\F(X0),
 *
 * where J represents the Jacobian matrix:
 *
 *     J = [ dF/dx1 ... dF/dxn ],
 *
 *     dF/dxi = [ df1/dxi ... dfn/dxi ]'.
 *
 * We use this iterative method until we obtain some form of convergence in X.
 */
void set_newton_parts(std::function<void(const double*,double*)>& fun, double* x, double* y, double* J, int dim)
{
    double yp[dim];

    double eps = 1.0e-5;

    fun(x, y);
    
    for (int i = 0; i < dim; i++) {

	x[i] += eps;

	fun(x, yp);
	
	for (int j = 0; j < dim; j++)

	    J[dim*j+i] = (yp[j] - y[j]) / eps;
	
	x[i] -= eps;
    }
}

int newtons_method(std::function<void(const double*,double*)>& fun, double* x, int dim, int max_iter)
{
    // for stopping criteria
    double tol = 1.0e-10;
    double norm;

    int out = 0;

    double y[dim];

    double J[dim * dim];

    double xp[dim];
    
    lapack_int ipiv[dim];

    for (int iter = 0; iter < max_iter; iter++) {

	// obtain the Jacobian matrix and the function evaluation
	set_newton_parts(fun, x, y, J, dim);
	
	// obtain y <- J\y
	out = LAPACKE_dgesv(LAPACK_ROW_MAJOR, dim, 1, J, dim, ipiv, y, 1);
	if (out != 0) {
	    printf("Error: %d\n", out);
	    return out;
	}

	// update x -= y
	for (int i = 0; i < dim; i++)
	    xp[i] = x[i] - y[i];

	// are we done yet?
	norm = 0.0;
	for (int i = 0; i < dim; i++)
	    norm += (y[i]*y[i]);
	if (norm < tol)
	    return 0;
    
	for (int i = 0; i < dim; i++)
	    x[i] = xp[i];

    }
    
    for (int i = 0; i < dim; i++)
	x[i] = xp[i];
    
    return out;
}


