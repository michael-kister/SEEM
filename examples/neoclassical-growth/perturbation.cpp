#include "perturbation.hpp"

void solve_gx_hx
(double* gx, double* hx, double*** dF, int num_control, int num_state)
{
    int num_variable = num_state + num_control;

    int n = num_variable;
    int nx = num_state;
    int ny = num_control;

    double A [num_variable * num_variable]; // A = [fxp fyp] (Blocks [2][0] & [4][0])
    double B [num_variable * num_variable]; // B = [-fx -fy] (Blocks [1][0] & [3][0])
  
  
    for (int i = 0; i < num_variable; ++i) {
	for (int j = 0; j < num_variable; ++j) {
	    if (j < num_state) {
		A[num_variable*i+j] =      dF[2][0][num_state*i+j];
		B[num_variable*i+j] = -1.0*dF[1][0][num_state*i+j];
	    } else {
		A[num_variable*i+j] =      dF[4][0][num_control*i+j-num_state];
		B[num_variable*i+j] = -1.0*dF[3][0][num_control*i+j-num_state];
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


    if (sdim != num_control) {
	printf("There were %d explosive eigenvalues, ", sdim);
	printf("while there should have been %d.\n", num_control);
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
(Tensor& gxx_T, Tensor& hxx_T, Tensor (&dF_T)[5][5], int ny, int nx,
 const Tensor& gx_T, const Tensor& hx_T) {

    // save size parameters
    //int widths[] = {1,nx,nx,ny,ny};

    // create an identity matrix
    Tensor I_T({nx,nx});
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < nx; j++)
	    I_T.X[nx*i+j] = (i==j) ? 1.0 : 0.0;
    Tensor Ixx_T = ~(I_T << I_T);

    // useful container for looping over tensor products
    Tensor K_T[4] = {I_T, hx_T, gx_T, gx_T*hx_T};
    
    //=====================================
    // this needs to be moved outside
    //Tensor dF_T[5][5];
    //for (int i = 0; i < 5; ++i)
    //	for (int j = 0; j < 5; ++j)
    //	    dF_T[i][j] = Tensor({nx+ny,widths[i],1,widths[j]}, tensor[i][j]);
    //=====================================

    // first the fun stuff
    // TODO: can i and j be switched?
    Tensor F_T({nx+ny,nx,1,nx});
    for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	    F_T += dF_T[j+1][i+1] * (K_T[j] << K_T[i]);
    
    ((~F_T) ^= {1,0}) *= -1;
    F_T.sizes[0] *= F_T.sizes[1];
    F_T.sizes[1] = 1;

    // then the painful stuff
    Tensor B_T = hx_T << hx_T;
    ~B_T ^= {1,0};
    Tensor BA_T = B_T   << dF_T[4][0];
    Tensor IC_T = Ixx_T << dF_T[3][0];
    Tensor ID_T = Ixx_T << dF_T[2][0];
    Tensor IE_T = Ixx_T << (dF_T[4][0] * gx_T);

    BA_T += IC_T;
    ID_T += IE_T;

    ~BA_T ^= {1,0};
    ~ID_T ^= {1,0};
    
    BA_T.X.insert(BA_T.X.end(), ID_T.X.begin(), ID_T.X.end());
    BA_T.sizes[0] += ID_T.sizes[0];
    BA_T ^= {1,0};

    // now combine together
    BA_T |= F_T;

    // now assign results
    gxx_T = Tensor({nx,1,nx,ny},&BA_T.X[0]);
    gxx_T ^= {3,0,1,2};
    //gxx_T.print();

    hxx_T = Tensor({nx,1,nx,nx},&BA_T.X[ny*nx*nx]);
    hxx_T ^= {3,0,1,2};
    //hxx_T.print();
}

void solve_gss_hss
(Tensor& gss_T, Tensor& hss_T, Tensor (&dF_T)[5][5], int num_control, int num_state, int neps,
 const Tensor& gx_T, const Tensor& gxx_T, const Tensor& eta_T)
{
    int nx = num_state;
    int ny = num_control;

    auto ghss_fun_T =
	[nx,ny](Tensor& F_T, Tensor& df_T, Tensor v1_T, Tensor v2_T) -> void
	{
	    v2_T ^= {1,0};
	    v1_T *= v2_T;
	    (v1_T++++) ^= {1,2,0,3};
	    F_T += df_T * v1_T;
	};
    Tensor F_T({nx+ny,1});
    //int n = nx+ny;
    //int widths[] = {nx,ny};
    
    //=====================================
    //int widths0[] = {1,nx,nx,ny,ny};
    // this needs to be moved outside
    //Tensor dF_T[5][5];
    //for (int i = 0; i < 5; ++i)
    //for (int j = 0; j < 5; ++j)
    //    dF_T[i][j] = Tensor({nx+ny,widths0[i],1,widths0[j]}, tensor[i][j]);
    //=====================================

    //Tensor eta_T({nx,neps},eta);
    //Tensor gx_T({ny,nx},gx);
    Tensor V_T[] = {eta_T, gx_T*eta_T};
    
    for (int i = 0; i < 2; i++) // dx', dy'
	for (int j = 0; j < 2; j++) // dx', dy'
	    ghss_fun_T(F_T, dF_T[2*j+2][2*i+2], V_T[i], V_T[j]);

    //Tensor gxx_T({ny,nx,1,nx},gxx);
    Tensor ee_T = eta_T * (eta_T ^ intvec({1,0}));
    (ee_T++++) ^= {0,2,1,3};
    F_T += dF_T[4][0] * gxx_T * ee_T;
    F_T *= -1;

    Tensor AB_T = dF_T[4][0] + dF_T[3][0];
    Tensor CD_T = (dF_T[4][0]*gx_T) + dF_T[2][0];
    ~AB_T ^= {1,0};
    ~CD_T ^= {1,0};
    
    AB_T.X.insert(AB_T.X.end(), CD_T.X.begin(), CD_T.X.end());
    AB_T.sizes[0] += CD_T.sizes[0];
    AB_T ^= {1,0};

    AB_T |= F_T;

    gss_T = Tensor({1,1,1,ny},&AB_T.X[0]);
    gss_T ^= {3,0,1,2};
    //gss_T.print();
    
    hss_T = Tensor({1,1,1,nx},&AB_T.X[ny*1*1]);
    hss_T ^= {3,0,1,2};
    //hss_T.print();
}

