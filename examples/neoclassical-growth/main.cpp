
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <adolc/adolc.h>

#include <lapacke.h>
#include <cblas.h>

void solve_gx_hx
(double* gx, double* hx, double*** tensor, int num_control, int num_state);

void solve_gxx_hxx
(double* gxx, double* hxx, double*** tensor, int ny, int nx,
 const double* gx, const double* hx);

void solve_gss_hss
(double* gss, double* hss, double*** tensor, int num_control, int num_state, int neps,
 const double* gx, const double* gxx, const double* eta);

void kronecker_product
(double* M_out, const double* A, const double* B, int r1, int c1, int r2, int c2)
{
    // note: this assumes row major layout
    for (int i = 0; i < r1; i++)
	for (int j = 0; j < c1; j++)
	    for (int p = 0; p < r2; p++)
		for (int q = 0; q < c2; q++)
		    M_out[r2*c1*c2*i+c1*c2*p+c2*j+q] = A[c1*i+j] * B[c2*p+q];
}

void square_transpose
(double* M, int n)
{
    for (int i = 0; i < n; i++)
	for (int j = 0; j < i; j++)
	    std::swap(M[n*i+j], M[n*j+i]);
}

void flatten_tensor
(double* T_in, double* T_out, int d1, int d2, int d3, int d4)
{
    for (int i = 0; i < d1; i++)
	for (int j = 0; j < d2; j++)
	    for (int p = 0; p < d3; p++)
		for (int q = 0; q < d4; q++)
		    T_out[p*d1*d2*d4 + i*d2*d4 + q*d2 + j] =
			T_in[i*d2*d3*d4 + j*d3*d4 + p*d4 + q];
}


void PrintMatrix(const double* M, int nrow, int ncol) {
    for (int r = 0; r < nrow; ++r) {
	printf("[ ");
	for (int c = 0; c < ncol; ++c) {
	    if (M[r*ncol+c] > 0.00001 ||
		M[r*ncol+c] < -.00001)
		printf("%10.6f ", M[r*ncol+c]);
	    else
		printf("     -     ");
	}
	printf("]\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {

    /*****************************************************************************
     * SET UP ENVIRONMENT
     ****************************************************************************/

    // constants
    int num_state   = 2; // number of "non-jump" variables
    int num_control = 2; // number of "jump" variables
    int num_param   = 7; // number of parameters




    // Don't touch
    //==========================
    // useful
    int num_variable = num_state + num_control;
    int degree = 2;
    int tensor_length = binomi(2*num_variable + degree, degree);

    // initilize independent and dependent variables
    double  *theta = new double  [num_param];
    double  *X_ss  = new double  [2*num_variable]; // steady states for location

    // storage for ADOL-C
    adouble *X     = new adouble [2*num_variable]; // includes two times
    adouble *Y     = new adouble [1*num_variable]; // half as many equations
    locint  *para  = new locint  [num_param];
    double  *pxy   = new double  [1*num_variable]; // TODO: verify use!!

    // seed matrix
    double **S = new double* [2*num_variable];
    for (int i = 0; i < 2*num_variable; ++i) {
	S[i] = new double [2*num_variable](); // set them all to zero
	S[i][i] = 1.0;
    }

    // tensor containing auto-diff results
    double **adolc_tensor = new double* [1*num_variable];
    for (int i = 0; i < num_variable; ++i) {
	adolc_tensor[i] = new double [tensor_length];
    }
  

    /*****************************************************************************
     * OBTAIN STEADY STATE
     ****************************************************************************/

    // set parameter values
    double beta  = 0.9896;
    double tau   = 2.0;
    double thxta = 0.357;
    double alpha = 0.4;
    double delta = 0.0196;
    double rho   = 0.95;
    double sigma = 0.007;
  
    // store them all together
    theta[0] = beta;
    theta[1] = tau;
    theta[2] = thxta;
    theta[3] = alpha;
    theta[4] = delta;
    theta[5] = rho;
    theta[6] = sigma;

    // obtain steady states
    double Phi = pow(((1/beta)-1+delta)/alpha,1/(alpha-1));
    double Psi = (thxta/(1-thxta))*(1-alpha)*pow(Phi,alpha);

    double l_ss = Psi/(Psi-(delta*Phi)+pow(Phi,alpha));
    double c_ss = (1-l_ss)*Psi;
    double k_ss = Phi*l_ss;
    double z_ss = 0;
  
    // store them all together
    X_ss[0] = k_ss; // current states
    X_ss[1] = z_ss;
    X_ss[2] = k_ss; // future states
    X_ss[3] = z_ss;
    X_ss[4] = l_ss; // current policy
    X_ss[5] = c_ss;
    X_ss[6] = l_ss; // future policy
    X_ss[7] = c_ss;

  
    /*****************************************************************************
     * AUTOMATIC DIFFERENTIATION
     ****************************************************************************/

    // Step 1 ....................................................... turn on trace
    trace_on(0);

    // Step 2 : load parameters
    for (int i = 0; i < num_param; ++i) {
	para[i] = mkparam_idx(theta[i]);
    }
  
    // Step 3 ....................................... make shortcuts for parameters
#define BETA_X getparam(para[0])
#define TAU__X getparam(para[1])
#define THETAX getparam(para[2])
#define ALPHAX getparam(para[3])
#define DELTAX getparam(para[4])
#define RHO__X getparam(para[5])
#define SIGMAX getparam(para[6])
  
    // Step 4 ...................................... register independent variables
    X[0] <<= k_ss; // current states
    X[1] <<= z_ss;
    X[2] <<= k_ss; // future states
    X[3] <<= z_ss;
    X[4] <<= l_ss; // current policy
    X[5] <<= c_ss;
    X[6] <<= l_ss; // future policy
    X[7] <<= c_ss;

    // Step 5 ...................................... make shortcuts for ind. values
    adouble k_t    = X[0]; // current states
    adouble z_t    = X[1];
    adouble k_tp1  = X[2]; // future states
    adouble z_tp1  = X[3];
    adouble l_t    = X[4]; // current policy
    adouble c_t    = X[5];
    adouble l_tp1  = X[6]; // future policy
    adouble c_tp1  = X[7];
  
    // Step 6 ..................................... construct some helper variables

    adouble eq1_lhs   = pow(pow(c_t  ,THETAX)*pow(1-l_t  ,1-THETAX),1-TAU__X)*THETAX/c_t  ;
    adouble eq1_rhs_1 = pow(pow(c_tp1,THETAX)*pow(1-l_tp1,1-THETAX),1-TAU__X)*THETAX/c_tp1;
    adouble eq1_rhs_2 = (1-DELTAX) + ALPHAX*exp(z_tp1)*pow(k_tp1/l_tp1,ALPHAX-1);

    adouble eq2_lhs = (c_t*(1-THETAX))/(THETAX*(1-l_t));
    adouble eq2_rhs = (1-ALPHAX)*exp(z_t)*pow(k_t/l_t,ALPHAX);
  
    adouble eq3_lhs = c_t + k_tp1;
    adouble eq3_rhs = (1-DELTAX)*k_t + exp(z_t)*pow(k_t,ALPHAX)*pow(l_t,1-ALPHAX);
  
    adouble eq4_lhs = z_tp1;
    adouble eq4_rhs = RHO__X*z_t;
  
    // Step 6 ...................................... write out our target equations

    Y[0] = eq1_lhs - BETA_X * eq1_rhs_1 * eq1_rhs_2;
    Y[1] = eq2_lhs - eq2_rhs;
    Y[2] = eq3_lhs - eq3_rhs;
    Y[3] = eq4_lhs - eq4_rhs;
  
    // Step 7 .................................... store evaluated for use later???
    for (int i = 0; i < num_variable; ++i)
	Y[i] >>= pxy[i];

    // Step 8 ...................................................... turn off trace
    trace_off();
  
    // Step 9 ................................... perform automatic differentiation
    tensor_eval(0, num_variable, 2*num_variable, 2,
		2*num_variable, X_ss, adolc_tensor, S);

  
    /*****************************************************************************
     * ANALYSIS
     ****************************************************************************/

    printf("\nl_ss : %7.4f\n", l_ss);
    printf("c_ss : %7.4f\n", c_ss);
    printf("k_ss : %7.4f\n", k_ss);
    printf("z_ss : %7.4f\n\n", z_ss);
  

    /*
    for (int v = 0; v < num_variable; ++v) {
	printf("f(%d) : \n", v+1);
	for (int i = 0; i <= 2*num_variable; ++i) {
	    printf("   ");
	    for (int j = 0; j <= i; ++j) {
		if (adolc_tensor[v][(i*(i+1)/2)+j] > 0.00001 ||
		    adolc_tensor[v][(i*(i+1)/2)+j] < -.00001)
		    printf("%10.7f  ", adolc_tensor[v][(i*(i+1)/2)+j]);
		else
		    printf("    -       ");
	    }
	    printf("\n");
	}
	printf("\n");
    }
    */

    /*****************************************************************************
     * REARRANGING DATA
     ****************************************************************************/

    // first we want a mapping to the indices we want

    int group_size[] = {1,num_state,num_state,num_control,num_control};

    int r, c, ii, jj;

    int ***index_map = new int** [5];
    for (int i = 0; i < 5; ++i) {
	index_map[i] = new int* [5];
	for (int j = 0; j < 5; ++j) {
	    index_map[i][j] = new int [group_size[i] * group_size[j]];
	    for (int u = 0; u < group_size[i]; ++u) {
		r = u;
		for (int k = 0; k < i; ++k) { r += group_size[k]; }
		for (int v = 0; v < group_size[j]; ++v) {
		    c = v;
		    for (int k = 0; k < j; ++k) { c += group_size[k]; }
		    ii = (r > c) ? r : c;
		    jj = (r > c) ? c : r;
		    index_map[i][j][group_size[j]*u+v] = (ii*(ii+1)/2)+jj;
		}
	    }
	}
    }

    // then we actually want to construct our necessary arrays

    int block_size;
  
    double ***tensor = new double** [5];
    for (int i = 0; i < 5; ++i) {
	tensor[i] = new double* [5];
	for (int j = 0; j < 5; ++j) {
	    block_size = group_size[i] * group_size[j];
	    tensor[i][j] = new double [num_variable * block_size];
	    for (int u = 0; u < num_variable; ++u) {
		for (int v = 0; v < block_size; ++v) {
		    tensor[i][j][block_size*u+v] =
			adolc_tensor[u][index_map[i][j][v]];
		}
	    }
	}
    }
  
  
  
    /*****************************************************************************
     * SOLVE GX HX
     ****************************************************************************/
    //int n = num_variable;
    int nx = num_state;
    int ny = num_control;

    double *gx  = new double [ny * nx]();
    double *hx  = new double [nx * nx]();
  
    solve_gx_hx(gx, hx, tensor, num_control, num_state);

  
  
    printf("hx: (kt, zt) (transition)\n");
    PrintMatrix(hx, num_state, num_state);
  
    printf("gx: (lt, ct) (measurement)\n");
    PrintMatrix(gx, num_control, num_state);
  
    /*****************************************************************************
     * SOLVE GXX, HXX, GSS, HSS.
     ****************************************************************************/

    double *gxx = new double [ny * nx * nx]();
    double *hxx = new double [nx * nx * nx]();
    //double *gss = new double [ny]();
    //double *hss = new double [nx]();

    solve_gxx_hxx(gxx, hxx, tensor, num_control, num_state, gx, hx);
  
    printf("hxx: (kt, zt) (transition)\n");
    PrintMatrix(hxx, num_state*num_state, num_state);
  
    printf("gxx: (lt, ct) (measurement)\n");
    PrintMatrix(gxx, num_control*num_state, num_state);

    int num_shock = 1;
    double eta[num_state*num_shock];
    eta[0] = 0;
    eta[1] = 1;

    double *gss = new double [ny]();
    double *hss = new double [nx]();
    solve_gss_hss(gss, hss, tensor, num_control, num_state, num_shock, gx, gxx, eta);

    printf("hss: (kt, zt) (transition)\n");
    PrintMatrix(hss, num_state, 1);
  
    printf("gss: (lt, ct) (measurement)\n");
    PrintMatrix(gss, num_control, 1);
    
    
  
    /*****************************************************************************
     * CLEAN UP ENVIRONMENT
     ****************************************************************************/

    for (int i = 0; i < 5; ++i) {
	for (int j = 0; j < 5; ++j) {
	    delete[] index_map[i][j];
	}
	delete[] index_map[i];
    }
    delete[] index_map;
  
    for (int i = 0; i < 5; ++i) {
	for (int j = 0; j < 5; ++j) {
	    delete[] tensor[i][j];
	}
	delete[] tensor[i];
    }
    delete[] tensor;




  
    for (int i = 0; i < num_variable; ++i)
	delete[] adolc_tensor[i];
    delete[] adolc_tensor;

    for (int i = 0; i < 2*num_variable; ++i)
	delete[] S[i];
    delete[] S;

    delete[] pxy;
    delete[] para;
    delete[] Y;
    delete[] X;

    delete[] X_ss;
    delete[] theta;

    delete[] gx;
    delete[] hx;
    delete[] gxx;
    delete[] hxx;
  
    return 0;
}




void solve_gx_hx
(double* gx, double* hx, double*** tensor, int num_control, int num_state) {

    int num_variable = num_state + num_control;

    int n = num_variable;
    int nx = num_state;
    int ny = num_control;

    double A [num_variable * num_variable]; // A = [fxp fyp] (Blocks [2][0] & [4][0])
    double B [num_variable * num_variable]; // B = [-fx -fy] (Blocks [1][0] & [3][0])
  
  
    for (int i = 0; i < num_variable; ++i) {
	for (int j = 0; j < num_variable; ++j) {
	    if (j < num_state) {
		A[num_variable*i+j] =      tensor[2][0][num_state*i+j];
		B[num_variable*i+j] = -1.0*tensor[1][0][num_state*i+j];
	    } else {
		A[num_variable*i+j] =      tensor[4][0][num_control*i+j-num_state];
		B[num_variable*i+j] = -1.0*tensor[3][0][num_control*i+j-num_state];
	    }
	}
    }
    //printf("A:\n");
    //PrintMatrix(A, num_variable, num_variable);
  
    //printf("B:\n");
    //PrintMatrix(B, num_variable, num_variable);
  
    double Q [] = {0}; // unused
    double Z [n * n];  // definitely used
  
    double rconde[2];
    double rcondv[2];
  
    double ar[n];
    double ai[n];
    double be[n];
  
    lapack_int sdim;
  
    auto mod_gt_one_lam =
	[](const double* ar, const double* ai, const double* be) -> lapack_logical {
	    // check that the modulus of (ar+ai)/b is less than one
	    if (sqrt(((*ar)*(*ar)+(*ai)*(*ai))/((*be)*(*be))) > 1)
		return 1;
	    else
		return 0;
	};
  
    /*------------------------------------------------------------*/
    LAPACKE_dggesx(LAPACK_ROW_MAJOR,'N','V','S', mod_gt_one_lam,'N',
		   n, A, n, B, n, &sdim, ar, ai, be,
		   Q, n, Z, n, rconde, rcondv);

    printf("There were %d explosive eigenvalues.\n", sdim);
    if (sdim != num_control) {
	printf("Uh oh...\n");
	abort();
    }
  
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
  
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, ny, nx, mz22p, ny, ipiv, gx, nx);
  
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
  
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, nx, nx, z11p, nx, ipiv, b11p, nx);
  
    double b11_z11 [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    b11_z11[nx*j+i] = b11p[nx*i+j];
  
    double a11 [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    a11[nx*i+j] = A[n*i+j];
  
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, nx, nx, a11, nx, ipiv, b11_z11, nx);
  
    double z11 [nx * nx];
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    z11[nx*i+j] = Z[n*i+j];
  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx, nx, nx,
		1.0, z11, nx, b11_z11, nx, 0.0, hx, nx);
}


void solve_gxx_hxx
(double* gxx, double* hxx, double*** tensor, int ny, int nx,
 const double* gx, const double* hx) {

    auto ghxx_fun =
	[nx,ny](double* F, double* df, double* kron1, double* kron2,
		int r1, int c1, int r2, int c2) -> void
	{
	    double kron[r1*r2*c1*c2];
	    kronecker_product(kron, kron1, kron2, r1, c1, r2, c2);
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx+ny, c1*c2, r1*r2,
			1.0, df, r1*r2, kron, c1*c2, 1.0, F, c1*c2);
	};

    
    //----------------------------------------------------------------------
    // first the fun stuff

    // use value-initialization to ensure it starts at zero
    double F[(nx+ny)*nx*nx] = {0};
    
    int widths[] = {nx,nx,ny,ny};
    
    double** K = new double* [4];
    for (int i = 0; i < 4; i++)
	K[i] = new double [widths[i]*nx];

    // set the matrices that are eligible to enter kronecker product

    // K[0] = I
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    K[0][nx*i+j] = (i==j) ? 1.0 : 0.0;

    // K[1] = hx
    for (int i = 0; i < nx*nx; i++)
	K[1][i] = hx[i];

    // K[2] = gx
    for (int i = 0; i < ny*nx; i++)
	K[2][i] = gx[i];

    // K[3] = gx * hx
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ny, nx, nx,
		1.0, gx, nx, hx, nx, 0.0, K[3], nx);

    // obtain the sum of 16 combinations of
    int n_tmp = nx > ny ? nx : ny;
    double flat_derivative[(nx+ny)*n_tmp*n_tmp];
	
    for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	    //flatten_tensor(derivatives[5*(i+1)+(j+1)], flat_derivative, nx+ny, widths[i], 1, widths[j]);
	    flatten_tensor(tensor[i+1][j+1], flat_derivative, nx+ny, widths[i], 1, widths[j]);
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

    double Ixx[nx*nx*nx*nx];
    for (int i = 0; i < nx*nx; i++)
	for (int j = 0; j < nx*nx; j++)
	    Ixx[nx*nx*i+j] = (i == j) ? 1.0 : 0.0;
    
    //--------------------------------------------------------------------------
    // A = F_{x}
    double A[n*ny];
    //flatten_tensor(derivatives[20], A, n,ny,1,1);
    flatten_tensor(tensor[4][0], A, n,ny,1,1);

    // B = ~(hx << hx)
    double B[nx*nx*nx*nx];
    kronecker_product(B, hx,hx, nx,nx, nx,nx);
    square_transpose(B, nx*nx);

    // BA = B << A
    double BA[xxn*xxy];
    kronecker_product(BA, B, A, nx*nx,nx*nx, n,ny); 
	
    //--------------------------------------------------------------------------
    // C = F_{x'}
    double C[n*ny];
    //flatten_tensor(derivatives[15], C, n,ny,1,1);
    flatten_tensor(tensor[3][0], C, n,ny,1,1);

    // IC = I << C
    double IC[xxn*xxy];
    kronecker_product(IC, Ixx, C, nx*nx,nx*nx, n,ny);

    //--------------------------------------------------------------------------
    // D = F_{y}
    double D[n*nx];
    //flatten_tensor(derivatives[10], D, n,nx,1,1);
    flatten_tensor(tensor[2][0], D, n,nx,1,1);

    // ID = I << D
    double ID[xxn*xxx];
    kronecker_product(ID, Ixx, D, nx*nx,nx*nx, n,nx);

    //--------------------------------------------------------------------------
    
    double E[n*nx];
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx, ny,
    //		1.0, derivatives[20], ny, gx, nx, 0.0, E, nx);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx, ny,
		1.0, tensor[4][0], ny, gx, nx, 0.0, E, nx);
    double IE[xxn*xxx];
    kronecker_product(IE, Ixx, E, nx*nx,nx*nx, n,nx);

    double G[xxn*xxn];
    for (int i = 0; i < xxn; i++)
	for (int j = 0; j < xxn; j++)
	    G[xxn*i+j] = (j < xxy) ? BA[xxy*i+j] + IC[xxy*i+j] : ID[xxx*i+j-xxy] + IE[xxx*i+j-xxy];
    
    lapack_int ipiv[xxn];
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, xxn, 1, G, xxn, ipiv, Ft, 1);
    

    //----------------------------------------------------------------------
    // allocate the top xxy to gxx, and the bottom xxx to hxx

    for (int i = 0; i < ny; i++)
	for (int j = 0; j < nx*nx; j++)
	    gxx[nx*nx*i+j] = Ft[ny*j+i];
    
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx*nx; j++)
	    hxx[nx*nx*i+j] = Ft[nx*j+i+ny*nx*nx];

    
}


void solve_gss_hss
(double* gss, double* hss, double*** tensor, int num_control, int num_state, int neps,
 const double* gx, const double* gxx, const double* eta)
{


    int nx = num_state;
    int ny = num_control;


    auto ghss_fun =
	[nx,ny](double* F, double* df, double* vec1, double* vec2,
		int r1, int c1, int r2, int c2) -> void
	{
	    double vec[r1*r2];
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, r2, r1, c2,
			1.0, vec2, c2, vec1, c1, 0.0, vec, r1);
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nx+ny, 1, r1*r2,
			1.0, df, r1*r2, vec, 1, 1.0, F, 1);
	};

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
	    //flatten_tensor(derivatives[5*(2*i+2)+(2*j+2)], flat_derivative, n, widths[i], 1, widths[j]);
	    flatten_tensor(tensor[2*i+2][2*j+2], flat_derivative, n, widths[i], 1, widths[j]);
	    ghss_fun(F, flat_derivative, V[i],V[j], widths[i],neps, widths[j],neps);
	}
    }

    double etaeta[nx*nx];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nx, nx, neps,
		1.0, eta, neps, eta, neps, 0.0, etaeta, nx);
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx*nx, ny,
    //		1.0, derivatives[20], ny, gxx, nx*nx, 0.0, flat_derivative, nx*nx);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx*nx, ny,
		1.0, tensor[4][0], ny, gxx, nx*nx, 0.0, flat_derivative, nx*nx);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, nx*nx,
		-1.0, flat_derivative, nx*nx, etaeta, 1, -1.0, F, 1);
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx, ny,
    //		1.0, derivatives[20], ny, gx, nx, 0.0, flat_derivative, nx);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nx, ny,
		1.0, tensor[4][0], ny, gx, nx, 0.0, flat_derivative, nx);

    double G[n*n];
    for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	    G[n*i+j] = (j < ny) ?
		tensor[4][0][ny*i+j] + tensor[3][0][ny*i+j] :
		flat_derivative[nx*i+j-ny] + tensor[2][0][nx*i+j-ny];
                //derivatives[20][ny*i+j] + derivatives[15][ny*i+j] :
                //flat_derivative[nx*i+j-ny] + derivatives[10][nx*i+j-ny];

    lapack_int ipiv[n];
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, G, n, ipiv, F, 1);

    for (int i = 0; i < ny; i++)
	gss[i] = F[i];

    for (int i = 0; i < nx; i++)
	hss[i] = F[i+ny];

}
