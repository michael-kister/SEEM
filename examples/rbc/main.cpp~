
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <adolc/adolc.h>

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
  double** S = new double* [2*num_variable];
  for (int i = 0; i < 2*num_variable; ++i) {
    S[i] = new double [2*num_variable](); // set them all to zero
    S[i][i] = 1.0;
  }

  // tensor containing auto-diff results
  double** adolc_tensor = new double* [1*num_variable];
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

  // first we want a mapping from the location of our array to the indices of 


  
  /*****************************************************************************
   * CLEAN UP ENVIRONMENT
   ****************************************************************************/

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
