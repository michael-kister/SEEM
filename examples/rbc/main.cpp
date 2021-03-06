
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <adolc/adolc.h>

#include <lapacke.h>
#include <cblas.h>


int main(int argc, char **argv) {

  /*****************************************************************************
   * SET UP ENVIRONMENT
   ****************************************************************************/

  // constants
  int num_state   = 2; // number of "non-jump" variables
  int num_control = 1; // number of "jump" variables
  int num_param   = 5; // number of parameters

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
  double alpha = 0.33;
  double beta  = 0.99;
  double delta = 0.025;
  double rho   = 0.95;
  double sigma = 0.01;

  // store them all together
  theta[0] = alpha;
  theta[1] = beta;
  theta[2] = delta;
  theta[3] = rho;
  theta[4] = sigma;  

  // obtain steady states

  double koh =  (log((1.0/beta)+delta-1.0)-log(alpha))/(alpha-1.0);
  double w_ss = log(1.0-alpha) + (alpha*koh);
  double r_ss = log(1.0*alpha) + ((alpha-1.0)*koh);
  double h_ss = log(1.0-alpha) - log((1.0-alpha)+(1.0-delta*exp((1.0-alpha)*koh)));
  double y_ss = alpha*koh + h_ss;
  double k_ss = koh + h_ss;
  double i_ss = log(delta) + k_ss;
  double c_ss = log(exp(y_ss) - exp(i_ss));
  double z_ss = 0.0;
  
  // store them all together
  X_ss[0] = X_ss[2] = k_ss;
  X_ss[1] = X_ss[3] = z_ss;
  X_ss[4] = X_ss[5] = h_ss;

  
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
  #define ALPHAX getparam(para[0])
  #define BETAX  getparam(para[1])
  #define DELTAX getparam(para[2])
  #define RHOX   getparam(para[3])
  #define SIGMAX getparam(para[4])
  
  // Step 4 ...................................... register independent variables
  X[0] <<= k_ss;
  X[1] <<= z_ss;
  X[2] <<= k_ss;
  X[3] <<= z_ss;
  X[4] <<= h_ss;
  X[5] <<= h_ss;

  // Step 5 ...................................... make shortcuts for ind. values
  adouble k_t   = X[0]; // Current states
  adouble z_t   = X[1];
  adouble k_tp1 = X[2]; // Future states
  adouble z_tp1 = X[3];
  adouble h_t   = X[4]; // Current policy
  adouble h_tp1 = X[5]; // Future policy

  // Step 6 ..................................... construct some helper variables
  adouble y_t   = (ALPHAX*k_t)   + ((1-ALPHAX)*(h_t  +z_t));
  adouble y_tp1 = (ALPHAX*k_tp1) + ((1-ALPHAX)*(h_tp1+z_tp1));
  adouble c_t   = log(1.0-exp(h_t))  +log(1.0-ALPHAX)+y_t  -h_t;
  adouble c_tp1 = log(1.0-exp(h_tp1))+log(1.0-ALPHAX)+y_tp1-h_tp1;
  adouble i_t   = log(exp(y_t) - exp(c_t));
  adouble r_tp1 = log(ALPHAX) + y_tp1 - k_tp1;

  // Step 6 ...................................... write out our target equations
  Y[0] = exp(k_tp1) - ((1-DELTAX)*exp(k_t)) - exp(i_t); // lom capital
  Y[1] = (BETAX*exp(c_t - c_tp1)*(1.0+exp(r_tp1)-DELTAX)) - 1.0; // euler
  Y[2] = z_tp1 - (RHOX * z_t); // lom productivity
	  
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

  printf("\nh_ss : %7.4f\n", h_ss);
  printf("r_ss : %7.4f\n", r_ss);
  printf("k_ss : %7.4f\n", k_ss);
  printf("y_ss : %7.4f\n", y_ss);
  printf("w_ss : %7.4f\n", w_ss);
  printf("i_ss : %7.4f\n", i_ss);
  printf("c_ss : %7.4f\n\n\n", c_ss);
  
  
  for (int v = 0; v < num_variable; ++v) {
    printf("f(%d) : \n", v+1);
    for (int i = 0; i <= 2*num_variable; ++i) {
      printf("   ");
      for (int j = 0; j <= i; ++j) {
	printf("%6.3f  ", adolc_tensor[v][(i*(i+1)/2)+j]);
      }
      printf("\n");
    }
    printf("\n");
  }



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

  int n = num_variable;
  int nx = num_state;
  int ny = num_control;

  double *gx  = new double [ny * nx]();
  double *hx  = new double [nx * nx]();
  
  
  
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
  
  double* Q;         // unused
  double  Z [n * n]; // definitely used
  
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

  printf("sdim = %d\n", sdim);
  
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
  
  
  
  printf("gx:\n");
  printf("[ %8.4f  %8.4f ]\n\n\n", gx[0], gx[1]);
  
  printf("hx:\n");
  printf("[ %8.4f  %8.4f ]\n[ %8.4f  %8.4f ]\n\n\n",
	 hx[0], hx[1], hx[2], hx[3]);
  
  
  
  
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

  
  return 0;
}
