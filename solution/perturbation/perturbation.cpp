#include "perturbation.h"


/**
 * First we have to deal with a bunch of junk pertaining to the base class that stores all
 * the members of the perturbation class. This actually does have some methods of its own,
 * namely it requires some means to determine how to allocate space.
 */

SGU_BASE::SGU_BASE(int nx_, int ny_, int neps_, int npar_, int deg_, short int tag_) :
    nx(nx_), ny(ny_), neps(neps_), npar(npar_), deg(deg_), tag(tag_)
{
    printf("\t\t\t\t\tConstructing 'SGU_BASE' (%p)\n\n", this);

    n_all = nx + nx + ny + ny;
    
    theta = new double [npar];
    
    x_ss = new double [nx]();
    y_ss = new double [ny]();
    eta  = new double [nx * neps]();
    X_ss = new double [n_all];
    
    gx  = new double [ny * nx]();
    hx  = new double [nx * nx]();
    gxx = new double [ny * nx * nx]();
    hxx = new double [nx * nx * nx]();
    gss = new double [ny]();
    hss = new double [nx]();

    g_lev_1   = new double [ny]();
    gx_lev_1  = new double [ny * nx]();
    h_lev_1   = new double [nx]();
    hx_lev_1  = new double [nx * nx]();
    
    g_lev_2   = new double [ny]();
    gx_lev_2  = new double [ny * nx]();
    gxx_lev_2 = new double [ny * nx * nx]();
    h_lev_2   = new double [nx]();
    hx_lev_2  = new double [nx * nx]();
    hxx_lev_2 = new double [nx * nx * nx]();

    int widths[] = {1, nx, nx, ny, ny};
    block_sizes = new int [nblock];
    int block = 0;
    for (int v = 0; v < 5; v++) {
	block_size_recursion(block_sizes, &block, widths, widths[v], deg, 1, v);
    }
    
    derivatives    = new double* [nblock];
    derivative_map = new int*    [nblock];
    for (int i = 0; i < nblock; i++) {
	derivatives[i]    = new double [n_all*block_sizes[i]];
	derivative_map[i] = new int    [block_sizes[i]];
    }
	
    block = 0;
    int* vars  = new int [deg];
    int* vars2 = new int [deg];
    int* sizes = new int [deg];
    for (int v = 0; v < 5; v++) {
	vars[0] = v;
	sizes[0] = widths[v];
	derivative_map_recursion1(derivative_map, &block, widths, sizes, deg, 1, vars, vars2);
    }
    
    tensor_length = binomi(n_all + deg, deg);
    adolc_tensor = myalloc2(nx+ny, tensor_length);
    
    delete[] vars;
    delete[] vars2;
    delete[] sizes;

    printf("\t\t\t\t\tDone.\n\n");
}







SGU_BASE::SGU_BASE(const SGU_BASE& that):
    nx(that.nx), ny(that.ny), neps(that.neps), npar(that.npar), deg(that.deg), tag(that.tag),
    n_all(that.n_all), tensor_length(that.tensor_length)

{
    printf("\t\t\t\t\tCopy constructor 'SGU_BASE' (%p receiving %p)\n\n", this, &that);
    
    theta = new double [npar];

    x_ss = new double [nx]();
    y_ss = new double [ny]();
    eta  = new double [nx * neps]();
    X_ss = new double [n_all];
    
    gx  = new double [ny * nx]();
    hx  = new double [nx * nx]();
    gxx = new double [ny * nx * nx]();
    hxx = new double [nx * nx * nx]();
    gss = new double [ny]();
    hss = new double [nx]();

    g_lev_1   = new double [ny]();
    gx_lev_1  = new double [ny * nx]();
    h_lev_1   = new double [nx]();
    hx_lev_1  = new double [nx * nx]();
    
    g_lev_2   = new double [ny]();
    gx_lev_2  = new double [ny * nx]();
    gxx_lev_2 = new double [ny * nx * nx]();
    h_lev_2   = new double [nx]();
    hx_lev_2  = new double [nx * nx]();
    hxx_lev_2 = new double [nx * nx * nx]();

    block_sizes = new int [nblock];

    derivatives    = new double* [nblock];
    derivative_map = new int*    [nblock];
    
    memcpy(block_sizes, that.block_sizes, nblock*sizeof(int));

    for (int i = 0; i < nblock; i++) {
	derivatives[i]    = new double [n_all*block_sizes[i]];
	derivative_map[i] = new int    [block_sizes[i]];
    }

    adolc_tensor = myalloc2(nx+ny, tensor_length);

    // start the main copying step
    
    memcpy(theta, that.theta, npar*sizeof(double));
    
    memcpy(x_ss, that.x_ss, nx*sizeof(double));
    memcpy(y_ss, that.y_ss, ny*sizeof(double));
    memcpy(eta,  that.eta,  nx*neps*sizeof(double));
    memcpy(X_ss, that.X_ss, n_all*sizeof(double));
    
    memcpy(gx,  that.gx,  ny*nx*sizeof(double));
    memcpy(hx,  that.hx,  nx*nx*sizeof(double));
    memcpy(gxx, that.gxx, ny*nx*nx*sizeof(double));
    memcpy(hxx, that.hxx, nx*nx*nx*sizeof(double));
    memcpy(gss, that.gss, ny*sizeof(double));
    memcpy(hss, that.hss, nx*sizeof(double));

    memcpy(g_lev_1,  that.g_lev_1,  ny*sizeof(double));
    memcpy(gx_lev_1, that.gx_lev_1, ny*nx*sizeof(double));
    memcpy(h_lev_1,  that.h_lev_1,  nx*sizeof(double));
    memcpy(hx_lev_1, that.hx_lev_1, nx*nx*sizeof(double));
    
    memcpy(g_lev_2,   that.g_lev_2,   ny*sizeof(double));
    memcpy(gx_lev_2,  that.gx_lev_2,  ny*nx*sizeof(double));
    memcpy(gxx_lev_2, that.gxx_lev_2, ny*nx*nx*sizeof(double));
    memcpy(h_lev_2,   that.h_lev_2,   nx*sizeof(double));
    memcpy(hx_lev_2,  that.hx_lev_2,  nx*nx*sizeof(double));
    memcpy(hxx_lev_2, that.hxx_lev_2, nx*nx*nx*sizeof(double));

    for (int i = 0; i < nblock; i++) {
	memcpy(derivatives[i], that.derivatives[i], n_all*block_sizes[i]*sizeof(double));
	memcpy(derivative_map[i], that.derivative_map[i], block_sizes[i]*sizeof(int));
    }
    
    printf("\t\t\t\t\tDone.\n\n");
}

void swap(SGU_BASE& first, SGU_BASE& second)
{
    printf("\t\t\t\t\t*Swapping 'SGU_BASE' (%p <--> %p)\n\n", &first, &second);

    std::swap(first.theta, second.theta);
    
    std::swap(first.x_ss, second.x_ss);
    std::swap(first.y_ss, second.y_ss);
    std::swap(first.eta,  second.eta);
    std::swap(first.X_ss, second.X_ss);

    std::swap(first.gx,  second.gx);
    std::swap(first.hx,  second.hx);
    std::swap(first.gxx, second.gxx);
    std::swap(first.hxx, second.hxx);
    std::swap(first.gss, second.gss);
    std::swap(first.hss, second.hss);

    std::swap(first.g_lev_1,   second.g_lev_1);
    std::swap(first.gx_lev_1,  second.gx_lev_1);
    std::swap(first.h_lev_1,   second.h_lev_1);
    std::swap(first.hx_lev_1,  second.hx_lev_1);

    std::swap(first.g_lev_2,    second.g_lev_2);
    std::swap(first.gx_lev_2,   second.gx_lev_2);
    std::swap(first.gxx_lev_2,  second.gxx_lev_2);
    std::swap(first.h_lev_2,    second.h_lev_2);
    std::swap(first.hx_lev_2,   second.hx_lev_2);
    std::swap(first.hxx_lev_2,  second.hxx_lev_2);

    std::swap(first.derivatives,  second.derivatives);
    
    printf("\t\t\t\t\t*Done.\n\n");
}

SGU_BASE::~SGU_BASE()
{
    printf("\t\t\t\t\t-Destructing 'SGU_BASE' (%p)... ", this);

    delete[] theta;
    
    delete[] x_ss;
    delete[] y_ss;
    delete[] eta;
    delete[] X_ss;
    
    delete[] gx;
    delete[] hx;
    delete[] gxx;
    delete[] hxx;
    delete[] gss;
    delete[] hss;
    
    delete[] g_lev_1;
    delete[] gx_lev_1;
    delete[] h_lev_1;
    delete[] hx_lev_1;
    
    delete[] g_lev_2;
    delete[] gx_lev_2;
    delete[] gxx_lev_2;
    delete[] h_lev_2;
    delete[] hx_lev_2;
    delete[] hxx_lev_2;

    delete[] block_sizes;
    for (int i = 0; i < nblock; i++) {
	delete[] derivatives[i];
	delete[] derivative_map[i];
    }
    delete[] derivatives;
    delete[] derivative_map;

    printf("Done.\n\n");
}


void SGU_BASE::block_size_recursion(int* block_sizes, int* block, int* widths, int size,
				    int deg, int deg_in, int var_in)
{
    //for (int v = 0; v <= var_in; v++) {
    for (int v = 0; v < 5; v++) {
	
	if (deg_in < deg-1) {
	    block_size_recursion(block_sizes, block, widths, size*widths[v], deg, deg_in+1, v);

	} else {
	    //printf("Block %2d  receiving %d\n", *block, size*widths[v]);
	    block_sizes[*block] = size*widths[v];
	    ++*block;

	}
    }
}

void SGU_BASE::derivative_map_recursion2(int** derivative_map, int* block, int* m, int* widths, int* sizes,
					 int deg, int deg_in, int* vars, int* vars2)
{
    // setting the index of the variable; 
    vars2[deg_in] = 0;
    for (int i = 0; i < vars[deg_in]; i++)
	vars2[deg_in] += widths[i];

    for (int v2 = 0; v2 < widths[vars[deg_in]]; v2++) {

	if (deg_in < deg-1) {

	    derivative_map_recursion2(derivative_map, block, m, widths, sizes, deg, deg_in+1, vars, vars2);

	} else {

	    int* vars22 = new int [deg];
	    for (int i = 0; i < deg; i++)
		vars22[i] = vars2[i];
	    insert_sort_int(vars22,deg);
	    
	    derivative_map[*block][*m] = tensor_address(deg, vars22);
	    //printf("   Spot: %d  Index: %2d    [%d,%d]    Vars: (%d,%d)\n",
	    //       *m,derivative_map[*block][*m],vars2[0],vars2[1],vars[0],vars[1]);
	    
	    ++*m;
	}
	vars2[deg_in]++;
    }
}


void SGU_BASE::derivative_map_recursion1(int** derivative_map, int* block, int* widths, int* sizes,
					 int deg, int deg_in, int* vars, int* vars2)
{
    //for (int v = 0; v <= vars[deg_in-1]; v++) {
    for (int v = 0; v < 5; v++) {

	vars[deg_in] = v;
	sizes[deg_in] = widths[v];

	if (deg_in < deg-1) {
	    derivative_map_recursion1(derivative_map, block, widths, sizes, deg, deg_in+1, vars, vars2);
	    
	} else {

	    int m = 0;

	    vars2[0] = 0;
	    for (int i = 0; i < vars[0]; i++)
		vars2[0] += widths[i];
	    
	    for (int i = 0; i < widths[vars[0]]; i++) {
		derivative_map_recursion2(derivative_map, block, &m, widths, sizes, deg, 1, vars, vars2);
		vars2[0]++;
	    }
	    ++*block;
	}
    }
}




/**
 * Now we have the real meat and potatoes of the perturbation methods; here is where we
 * can execute the automatic differentiation, solve the model, and display the values.
 */


//========================================//
//                                        //
//         PUBLIC MODEL METHODS           //
//                                        //
//========================================//


void perturbation::display(int d)
{
    if (d > 0 && d < 10) {

	printf("\nSigma: (at %p)\n", &sigma); printmat4(&sigma, 1,1,1,1);
	printf("y_ss: (at %p)\n", y_ss);    printmat4(y_ss, ny,1,1,1);
	printf("x_ss: (at %p)\n", x_ss);    printmat4(x_ss, nx,1,1,1);

	if (d > 1) {
	
	    printf("\ngx: (at %p)\n", gx);  printmat4(gx,  ny, nx, 1, 1);
	    printf("gxx: (at %p)\n", gxx); printmat4(gxx, ny, nx*nx, 1, 1);
	    printf("gss: (at %p)\n", gss); printmat4(gss, ny, 1, 1, 1);

	    printf("\nhx: (at %p)\n", hx);  printmat4(hx,  nx, nx, 1, 1);
	    printf("hxx: (at %p)\n", hxx); printmat4(hxx, nx, nx*nx, 1, 1);
	    printf("hss: (at %p)\n", hss); printmat4(hss, nx, 1, 1, 1);

	    if (d > 2) {

		printf("\nh_lev_1:\n"); printmat4(h_lev_1,   nx, 1,     1,1);
		printf("hx_lev_1:\n");  printmat4(hx_lev_1 , nx, nx,    1,1);

		printf("\ng_lev_1:\n"); printmat4(g_lev_1,   ny, 1,     1,1);
		printf("gx_lev_1:\n");  printmat4(gx_lev_1 , ny, nx,    1,1);

		printf("\nh_lev_2:\n"); printmat4(h_lev_2,   nx, 1,     1,1);
		printf("hx_lev_2:\n");  printmat4(hx_lev_2 , nx, nx,    1,1);
		printf("hxx_lev_2:\n"); printmat4(hxx_lev_2, nx, nx*nx, 1,1);

		printf("\ng_lev_2:\n"); printmat4(g_lev_2,   ny, 1,     1,1);
		printf("gx_lev_2:\n");  printmat4(gx_lev_2 , ny, nx,    1,1);
		printf("gxx_lev_2:\n"); printmat4(gxx_lev_2, ny, nx*nx, 1,1);

	    }
	}
    } else if (d > 10) {

	printf("\nSigma: (at %p)\n", &sigma); //printmat4(&sigma, 1,1,1,1);
	printf("y_ss:  (at %p)\n", y_ss);    //printmat4(y_ss, ny,1,1,1);
	printf("x_ss:  (at %p)\n", x_ss);    //printmat4(x_ss, nx,1,1,1);

	if (d > 11) {
	
	    printf("\ngx:    (at %p)\n", gx);  //printmat4(gx,  ny, nx, 1, 1);
	    printf("gxx:   (at %p)\n", gxx); //printmat4(gxx, ny, nx*nx, 1, 1);
	    printf("gss:   (at %p)\n", gss); //printmat4(gss, ny, 1, 1, 1);

	    printf("\nhx:    (at %p)\n", hx);  //printmat4(hx,  nx, nx, 1, 1);
	    printf("hxx:   (at %p)\n", hxx); //printmat4(hxx, nx, nx*nx, 1, 1);
	    printf("hss:   (at %p)\n", hss); //printmat4(hss, nx, 1, 1, 1);

	}
    }
}

int perturbation::load_parameters(double* theta_in, int verbose)
{
    for (int i = 0; i < npar; i++)
	theta[i] = theta_in[i];

    set_param_vec(tag, npar, theta);

    set_steady_state();

    differentiate_tag();

    map_tetrahedral_to_tensors();

    SGU_CALL(solve_gx_hx());
    SGU_CALL(solve_gxx_hxx());
    SGU_CALL(solve_gss_hss());

    set_level_solution();

    display(verbose);
    
    return 0;
}



//========================================//
//                                        //
//           ADOL-C FUNCTIONS             //
//                                        //
//========================================//



void perturbation::differentiate_tag()
{
    double** S = new double* [n_all];
    for (int i = 0; i < n_all; i++) {
	S[i] = new double [n_all]();
	
	S[i][i] = 1.0;
    }
    
    tensor_eval(tag, nx+ny, n_all, deg, n_all, X_ss, adolc_tensor, S);

    for (int i = 0; i < n_all; i++)
	delete[] S[i];
    delete[] S;
}


void perturbation::map_tetrahedral_to_tensors(void)
{
    for (int j_fvar = 0; j_fvar < nx+ny; j_fvar++)
    for (int i_block = 0; i_block < nblock; i++)
	  
            for (int k_diff = 0; k_diff < block_sizes[i_block]; k_diff++) {
	      
	        derivatives[i_block][k_diff + j_fvar * block_sizes[i_block]] =
		  
	            adolc_tensor[j_fvar][derivative_map[i_block][k_diff]];
      }
}






//========================================//
//                                        //
//           SOLUTION FUNCTIONS           //
//                                        //
//========================================//


int perturbation::solve_gx_hx()
{
    /*------------------------------------------------------------*/
    int n = nx + ny;

    double A [n * n]; // A = [fxp fyp] (Blocks 10 & 20)
    double B [n * n]; // B = [-fx -fy] (Blocks  5 & 15)


    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	    if (j < nx) {
		A[n*i+j] = derivatives[10][nx*i+j];
		B[n*i+j] = -1.0*derivatives[5][nx*i+j];
	    } else {
		A[n*i+j] = derivatives[20][ny*i+j-nx];
		B[n*i+j] = -1.0*derivatives[15][ny*i+j-nx];
	    }
	}
    }

    double* Q;         // unused
    double  Z [n * n]; // definitely used

    double rconde[2];
    double rcondv[2];
    
    double ar[n];
    double ai[n];
    double be[n];
    
    lapack_int sdim;
    
    /*------------------------------------------------------------*/
    SGU_CALL(LAPACKE_dggesx(LAPACK_ROW_MAJOR,'N','V','S',&mod_gt_one,'N',
			    n, A, n, B, n, &sdim, ar, ai, be,
			    Q, n, Z, n, rconde, rcondv));

    /*------------------------------------------------------------*/
    // solve for gx

    double mz22p [ny * ny];
    lapack_int ipiv [n];
    
    for (int i = nx; i < n; i++)
	for (int j = nx; j < n; j++)
	    mz22p[ny*(j-nx)+(i-nx)] = -1.0 * Z[n*i+j];
    
    for (int i = 0; i < nx; i++)
	for (int j = nx; j < n; j++)
	    gx[nx*(j-nx)+i] = Z[n*i+j];

    SGU_CALL(LAPACKE_dgesv(LAPACK_ROW_MAJOR, ny, nx, mz22p, ny, ipiv, gx, nx));

    /*------------------------------------------------------------*/
    // solve for hx

    double z11p [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    z11p[nx*j+i] = Z[n*i+j];

    double b11p [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    b11p[nx*j+i] = B[n*i+j];
    
    SGU_CALL(LAPACKE_dgesv(LAPACK_ROW_MAJOR, nx, nx, z11p, nx, ipiv, b11p, nx));
    
    double b11_z11 [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    b11_z11[nx*j+i] = b11p[nx*i+j];

    double a11 [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    a11[nx*i+j] = A[n*i+j];
    
    SGU_CALL(LAPACKE_dgesv(LAPACK_ROW_MAJOR, nx, nx, a11, nx, ipiv, b11_z11, nx));
    
    double z11 [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    z11[nx*i+j] = Z[n*i+j];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx, nx, nx,
		1.0, z11, nx, b11_z11, nx, 0.0, hx, nx);

    return 0;
}


void perturbation::ghxx_fun(double* F, double* df, double* kron1, double* kron2,
		    int r1, int c1, int r2, int c2)
{
    double kron[r1*r2*c1*c2];
    kronecker_product(kron, kron1, kron2, r1, c1, r2, c2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx+ny, c1*c2, r1*r2,
		1.0, df, r1*r2, kron, c1*c2, 1.0, F, c1*c2);
}
int perturbation::solve_gxx_hxx()
{
    //----------------------------------------------------------------------
    // first the fun stuff

    // use value-initialization to ensure it starts at zero
    double* F  = new double [(nx+ny)*nx*nx]();
    
    int widths[] = {nx,nx,ny,ny};
    
    double** K = new double* [4];
    for (int i = 0; i < 4; i++)
	K[i] = new double [widths[i]*nx];

    // set the matrices that are eligible to enter kronecker product
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    K[0][nx*i+j] = (i==j) ? 1.0 : 0.0;
    for (int i = 0; i < nx*nx; i++)
	K[1][i] = hx[i];
    for (int i = 0; i < ny*nx; i++)
	K[2][i] = gx[i];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ny, nx, nx,
		1.0, gx, nx, hx, nx, 0.0, K[3], nx);

    // obtain the sum of 16 combinations of
    double* flat_derivative;
    if (nx >= ny)
	flat_derivative = new double [(nx+ny)*nx*nx];
    else
	flat_derivative = new double [(nx+ny)*ny*ny];
	
    for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	    flatten_tensor(derivatives[5*(i+1)+(j+1)], flat_derivative, nx+ny, widths[i], 1, widths[j]);
	    ghxx_fun(F, flat_derivative, K[j],K[i], widths[j],nx, widths[i],nx);
	} 
    }

    // Ft is the negative transpose of F, since we'll want vec(F)
    double* Ft = new double [(nx+ny)*nx*nx];
    for (int i = 0; i < nx+ny; i++)
	for (int j = 0; j < nx*nx; j++)
	    Ft[(nx+ny)*j+i] = -1.0*F[nx*nx*i+j];
    
    //----------------------------------------------------------------------
    // now the painful stuff... 

    int n = nx+ny;
    int xxn = nx*nx*n;
    int xxx = nx*nx*nx;
    int xxy = nx*nx*ny;

    double* Ixx = new double [nx*nx*nx*nx];
    for (int i = 0; i < nx*nx; i++)
	for (int j = 0; j < nx*nx; j++)
	    Ixx[nx*nx*i+j] = (i == j) ? 1.0 : 0.0;
    
    double* A = new double [n*ny];
    flatten_tensor(derivatives[20], A, n,ny,1,1);

    double* B = new double [nx*nx*nx*nx];
    kronecker_product(B, hx,hx, nx,nx, nx,nx);
    square_transpose(B, nx*nx);
    double* BA = new double [xxn*xxy];
    kronecker_product(BA, B, A, nx*nx,nx*nx, n,ny); 
	
    double* C = new double [n*ny];
    flatten_tensor(derivatives[15], C, n,ny,1,1);
    
    double* IC = new double [xxn*xxy];
    kronecker_product(IC, Ixx, C, nx*nx,nx*nx, n,ny);
    
    double* D = new double [n*nx];
    flatten_tensor(derivatives[10], D, n,nx,1,1);
    double* ID = new double [xxn*xxx];
    kronecker_product(ID, Ixx, D, nx*nx,nx*nx, n,nx);

    double* E = new double [n*nx];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx, ny,
		1.0, derivatives[20], ny, gx, nx, 0.0, E, nx);
    double* IE = new double [xxn*xxx];
    kronecker_product(IE, Ixx, E, nx*nx,nx*nx, n,nx);

    double* G = new double [xxn*xxn];
    for (int i = 0; i < xxn; i++)
	for (int j = 0; j < xxn; j++)
	    G[xxn*i+j] = (j < xxy) ? BA[xxy*i+j] + IC[xxy*i+j] : ID[xxx*i+j-xxy] + IE[xxx*i+j-xxy];
    
    lapack_int ipiv[xxn];
    SGU_CALL(LAPACKE_dgesv(LAPACK_ROW_MAJOR, xxn, 1, G, xxn, ipiv, Ft, 1));
    

    //----------------------------------------------------------------------
    // allocate the top xxy to gxx, and the bottom xxx to hxx

    for (int i = 0; i < ny; i++)
	for (int j = 0; j < nx*nx; j++)
	    gxx[nx*nx*i+j] = Ft[ny*j+i];
    
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx*nx; j++)
	    hxx[nx*nx*i+j] = Ft[nx*j+i+ny*nx*nx];

    return 0;
}


void perturbation::ghss_fun(double* F, double* df, double* vec1, double* vec2,
		     int r1, int c1, int r2, int c2)
{
    double vec[r1*r2];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, r2, r1, c2,
		1.0, vec2, c2, vec1, c1, 0.0, vec, r1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx+ny, 1, r1*r2,
		1.0, df, r1*r2, vec, 1, 1.0, F, 1);
    
}
int perturbation::solve_gss_hss()
{
    //----------------------------------------------------------------------
    double* F = new double [nx+ny]();
    
    int n = nx+ny;
    int widths[] = {nx,ny};

    double** V = new double* [2];
    for (int i = 0; i < 2; i++)
	V[i] = new double [widths[i]*neps];

    // set the matrices that can enter the vec
    for (int i = 0; i < nx*neps; i++)
	V[0][i] = eta[i];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ny, neps, nx,
		1.0, gx, nx, eta, neps, 0.0, V[1], neps);

    int nm = (nx >= ny) ? nx : ny;
    double flat_derivative[n*nm*nm];
    
    for (int i = 0; i < 2; i++) {
	for (int j = 0; j < 2; j++) {
	    flatten_tensor(derivatives[5*(2*i+2)+(2*j+2)], flat_derivative, n, widths[i], 1, widths[j]);
	    ghss_fun(F, flat_derivative, V[i],V[j], widths[i],neps, widths[j],neps);
	}
    }

    double etaeta[nx*nx];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nx, nx, neps,
		1.0, eta, neps, eta, neps, 0.0, etaeta, nx);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx*nx, ny,
		1.0, derivatives[20], ny, gxx, nx*nx, 0.0, flat_derivative, nx*nx);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, nx*nx,
		-1.0, flat_derivative, nx*nx, etaeta, 1, -1.0, F, 1);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx, ny,
		1.0, derivatives[20], ny, gx, nx, 0.0, flat_derivative, nx);

    double G[n*n];
    for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	    G[n*i+j] = (j < ny) ?
		derivatives[20][ny*i+j] + derivatives[15][ny*i+j] :
		flat_derivative[nx*i+j-ny] + derivatives[10][nx*i+j-ny];

    lapack_int ipiv[n];
    SGU_CALL(LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, G, n, ipiv, F, 1));

    for (int i = 0; i < ny; i++)
	gss[i] = F[i];

    for (int i = 0; i < nx; i++)
	hss[i] = F[i+ny];

    return 0;
}


void perturbation::set_level_solution()
{
    /**-------------------------------------------------------------------------
     * It may be the case that we would prefer to work in levels, rather than in
     * differences from the steady-state. Therefore, we offer the matrices in
     * this form as well. If xb is the steady-state, then we have that:
     *
     * (x - xb) = HX*(x - xb) + HXX*(x - xb)^2 + HSS*(sigma)^2
     *
     * The intercept is fairly straightforward; we note that it does include the
     * second-order contribution from sigma. The second-order matrix for x is
     * also fairly straightforward; note that we [do/do not] pre-include the 0.5
     * scaling factor.
     *
     * The slope matrix is the only tricky one, since we must include portions
     * from the second-degree matrix:
     *
     * (x - xb) @ (x - xb) = (x @ x) - (x @ xb) - (xb @ x) + (xb @ xb)
     *
     * The code below should be correct. It's somewhat of a pain to write out,
     * but basically it makes use of symmetry.
     */
    double x_kron_x[nx*nx]; // temporarily used to store identity matrix
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    x_kron_x[nx*i + j] = (i == j) ? 1.0 : 0.0;
    double x_kron_I[nx*nx*nx];
    kronecker_product(x_kron_I, x_ss, x_kron_x, nx, 1, nx, nx);
    kronecker_product(x_kron_x, x_ss, x_ss, nx, 1, nx, 1);
    
    //--------------------------------------------------------------------------
    // X : degree 0
    //--------------
    for (int i = 0; i < nx; i++)
	h_lev_1[i] = x_ss[i];
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx, 1, nx,
		-1.0, hx, nx, x_ss, 1, 1.0, h_lev_1, 1);

    for (int i = 0; i < nx; i++)
	h_lev_2[i] = h_lev_1[i];
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx, 1, nx*nx,
		0.5, hxx, nx*nx, x_kron_x, 1, 1.0, h_lev_2, 1);
    
    for (int i = 0; i < nx; i++)
    	h_lev_2[i] += (0.5 * hss[i] * sigma * sigma);

    //--------------
    // X : degree 1
    //--------------
    for (int i = 0; i < nx*nx; i++)
	hx_lev_1[i] = hx_lev_2[i] = hx[i];
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx, nx, nx*nx,
		-1.0, hxx, nx*nx, x_kron_I, nx, 1.0, hx_lev_2, nx);
    
    //--------------
    // X : degree 2
    //--------------
    for (int i = 0; i < nx*nx*nx; i++)
	hxx_lev_2[i] = hxx[i];

    //--------------------------------------------------------------------------
    // Y : degree 0
    //--------------
    for (int i = 0; i < ny; i++)
	g_lev_1[i] = y_ss[i];
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ny, 1, nx,
		-1.0, gx, nx, x_ss, 1, 1.0, g_lev_1, 1);

    for (int i = 0; i < ny; i++)
	g_lev_2[i] = g_lev_1[i];
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ny, 1, nx*nx,
		0.5, gxx, nx*nx, x_kron_x, 1, 1.0, g_lev_2, 1);
    
    for (int i = 0; i < ny; i++)
    	g_lev_2[i] += (0.5 * gss[i] * sigma * sigma);

    //--------------
    // Y : degree 1
    //--------------
    for (int i = 0; i < ny*nx; i++)
	gx_lev_1[i] = gx_lev_2[i] = gx[i];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ny, nx, nx*nx,
		-1.0, gxx, nx*nx, x_kron_I, nx, 1.0, gx_lev_2, nx);
    
    //--------------
    // Y : degree 2
    //--------------
    for (int i = 0; i < ny*nx*nx; i++)
	gxx_lev_2[i] = gxx[i];

}


//========================================//
//                                        //
//           UTILITY FUNCTIONS            //
//                                        //
//========================================//



lapack_logical mod_gt_one(const double* ar, const double* ai, const double* be) {
    // check that the modulus of (ar+ai)/b is less than one
    if (sqrt(((*ar)*(*ar)+(*ai)*(*ai))/((*be)*(*be))) > 1)
	return 1;
    else
	return 0;
}


void insert_sort_int(int* X, int N)
{
    int i = 1;
    int j = 1;
    while (i < N) {
	j = i;
	while (j > 0 && X[j-1] < X[j]) {
	    std::swap(X[j], X[j-1]);
	    j--;
	}
	i++;
    }
}


void flatten_tensor(double* T_in, double* T_out, int d1, int d2, int d3, int d4)
{
    for (int i = 0; i < d1; i++)
	for (int j = 0; j < d2; j++)
	    for (int p = 0; p < d3; p++)
		for (int q = 0; q < d4; q++)
		    T_out[p*d1*d2*d4 + i*d2*d4 + q*d2 + j] =
			T_in[i*d2*d3*d4 + j*d3*d4 + p*d4 + q];
}
