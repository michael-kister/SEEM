
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <adolc/adolc.h>

#include "tensor.hpp"

/*
class Perturbation {

    // Only called from public "Solve" method
    void solve_gx_hx(void);
    void solve_gxx_hxx(void);
    void solve_gss_hss(void);
    
public:
    
    // specification parameters
    int num_state, num_control, num_shock, num_param;
    int approximation_level;

    // Member tensors
    Tensor derivatives;
    Tensor gx, hx, gxx, hxx, gss, hss;

    // Solve the model
    void Solve(void);    
};
*/

#include "perturbation.hpp"
#include "perturbation.cpp"

int main(int argc, char **argv) {

    /***************************************************************************
     * SET UP ENVIRONMENT
     **************************************************************************/

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
  

    /***************************************************************************
     * OBTAIN STEADY STATE
     **************************************************************************/

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
  
    /***************************************************************************
     * AUTOMATIC DIFFERENTIATION
     **************************************************************************/

    // Step 1 .................................................... turn on trace
    trace_on(0);

    // Step 2 : load parameters
    for (int i = 0; i < num_param; ++i) {
	para[i] = mkparam_idx(theta[i]);
    }
  
    // Step 3 .................................... make shortcuts for parameters
#define BETA_X getparam(para[0])
#define TAU__X getparam(para[1])
#define THETAX getparam(para[2])
#define ALPHAX getparam(para[3])
#define DELTAX getparam(para[4])
#define RHO__X getparam(para[5])
#define SIGMAX getparam(para[6])
  
    // Step 4 ................................... register independent variables
    X[0] <<= k_ss; // current states
    X[1] <<= z_ss;
    X[2] <<= k_ss; // future states
    X[3] <<= z_ss;
    X[4] <<= l_ss; // current policy
    X[5] <<= c_ss;
    X[6] <<= l_ss; // future policy
    X[7] <<= c_ss;

    // Step 5 ................................... make shortcuts for ind. values
    adouble k_t    = X[0]; // current states
    adouble z_t    = X[1];
    adouble k_tp1  = X[2]; // future states
    adouble z_tp1  = X[3];
    adouble l_t    = X[4]; // current policy
    adouble c_t    = X[5];
    adouble l_tp1  = X[6]; // future policy
    adouble c_tp1  = X[7];
  
    // Step 6 .................................. construct some helper variables

    adouble eq1_lhs   = pow(pow(c_t  ,THETAX)*pow(1-l_t  ,1-THETAX),1-TAU__X)*THETAX/c_t  ;
    adouble eq1_rhs_1 = pow(pow(c_tp1,THETAX)*pow(1-l_tp1,1-THETAX),1-TAU__X)*THETAX/c_tp1;
    adouble eq1_rhs_2 = (1-DELTAX) + ALPHAX*exp(z_tp1)*pow(k_tp1/l_tp1,ALPHAX-1);

    adouble eq2_lhs = (c_t*(1-THETAX))/(THETAX*(1-l_t));
    adouble eq2_rhs = (1-ALPHAX)*exp(z_t)*pow(k_t/l_t,ALPHAX);
  
    adouble eq3_lhs = c_t + k_tp1;
    adouble eq3_rhs = (1-DELTAX)*k_t + exp(z_t)*pow(k_t,ALPHAX)*pow(l_t,1-ALPHAX);
  
    adouble eq4_lhs = z_tp1;
    adouble eq4_rhs = RHO__X*z_t;
  
    // Step 6 ................................... write out our target equations

    Y[0] = eq1_lhs - BETA_X * eq1_rhs_1 * eq1_rhs_2;
    Y[1] = eq2_lhs - eq2_rhs;
    Y[2] = eq3_lhs - eq3_rhs;
    Y[3] = eq4_lhs - eq4_rhs;
  
    // Step 7 ................................. store evaluated for use later???
    for (int i = 0; i < num_variable; ++i)
	Y[i] >>= pxy[i];

    // Step 8 ................................................... turn off trace
    trace_off();
    
    // Step 9 ................................ perform automatic differentiation
    tensor_eval(0, num_variable, 2*num_variable, 2,
		2*num_variable, X_ss, adolc_tensor, S);

  
    /***************************************************************************
     * ANALYSIS
     **************************************************************************/

    printf("\nSteady-States:\n");
    printf("l_ss : %7.4f\n", l_ss);
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

    /***************************************************************************
     * REARRANGING DATA
     **************************************************************************/

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
  
    double ***derivatives = new double** [5];
    for (int i = 0; i < 5; ++i) {
	derivatives[i] = new double* [5];
	for (int j = 0; j < 5; ++j) {
	    block_size = group_size[i] * group_size[j];
	    derivatives[i][j] = new double [num_variable * block_size];
	    for (int u = 0; u < num_variable; ++u) {
		for (int v = 0; v < block_size; ++v) {
		    derivatives[i][j][block_size*u+v] =
			adolc_tensor[u][index_map[i][j][v]];
		}
	    }
	}
    }

    int nx = num_state;
    int ny = num_control;
    int widths[] = {1,nx,nx,ny,ny};
    Tensor derivatives_T[5][5];
    for (int i = 0; i < 5; ++i)
	for (int j = 0; j < 5; ++j)
	    derivatives_T[i][j] = Tensor
		({num_state+num_control,widths[i],1,widths[j]},derivatives[i][j]);

  
  
    /***************************************************************************
     * SOLVE GX HX
     **************************************************************************/


    double *gx  = new double [ny * nx]();
    double *hx  = new double [nx * nx]();
    solve_gx_hx(gx, hx, derivatives, num_control, num_state);
  
    Tensor gx_T({ny,nx},gx);
    Tensor hx_T({nx,nx},hx);

    printf("gx: (lt, ct) (measurement)\n");
    gx_T.print(); 

    printf("hx: (lt, ct) (measurement)\n");
    hx_T.print(); 

    //printf("\n");
  
    //printf("hx: (kt, zt) (transition)\n");
    //Tensor({nx,nx},hx).print();
    //printf("\n");
    
    /***************************************************************************
     * SOLVE GXX, HXX, GSS, HSS.
     **************************************************************************/

    Tensor gxx_T({ny,nx,1,nx});
    Tensor hxx_T({nx,nx,1,nx});
    solve_gxx_hxx(gxx_T, hxx_T, derivatives_T, num_control, num_state, gx_T, hx_T);

    printf("gxx: (lt, ct) (measurement)\n");
    gxx_T.print(); 

    printf("hxx: (lt, ct) (measurement)\n");
    hxx_T.print(); 



    //printf("gxx: (lt, ct) (measurement)\n");
    //gxx_T.print();
    //printf("\n");
    //printf("hxx: (kt, zt) (transition)\n");
    //hxx_T.print();
    //printf("\n");
    
    //double gxx[ny*nx*nx];
    //double hxx[ny*nx*nx];
    //solve_gxx_hxx(gxx, hxx, tensor, num_control, num_state, gx, hx);

    //printf("gxx: (lt, ct) (measurement)\n");
    //Tensor({ny,nx,1,nx},gxx).print();
    //printf("\n");
    
    //printf("hxx: (kt, zt) (transition)\n");
    //Tensor({nx,nx,1,nx},hxx).print();
    //printf("\n");
    


    
    int num_shock = 1;
    //double eta[] = {0,1};
    //double *gss = new double [ny]();
    //double *hss = new double [nx]();

    int neps = 1;
    
    Tensor gss_T({ny,1,1,1});
    Tensor hss_T({ny,1,1,1});
    Tensor eta_T({nx,neps});
    eta_T.X[0] = 0;
    eta_T.X[1] = 1;

    solve_gss_hss(gss_T, hss_T, derivatives_T, num_control, num_state, num_shock,
		  gx_T, gxx_T, eta_T);
    
    printf("gss: (lt, ct) (measurement)\n");
    gss_T.print(); 

    printf("hss: (lt, ct) (measurement)\n");
    hss_T.print(); 

    //printf("hss: (kt, zt) (transition)\n");
    //Tensor({nx,1},hss).print(); 
    
  
    
    /***************************************************************************
     * Test Steady State
     **************************************************************************/

    /*
    printf("\n\n\n");
    double xt[] = {0.05, 0.05};

    double yt[] = {l_ss, c_ss};
    Tensor yss_T({ny,1},yt);
    //Tensor({1,1,ny,1},yt).print();
    // looping over output variable
    for (int i = 0; i < ny; ++i) {
	// looping over first multiplication
	for (int j = 0; j < nx; ++j) {
	    yt[i] += gx[nx*i+j]*xt[j];
	    for (int k = 0; k < nx; ++k) {
		yt[i] += 0.5*gxx[nx*nx*i+nx*j+k]*xt[j]*xt[k];
	    }
	}
    }
    printf("yt from arrays:\n");
    Tensor({1,1,ny,1},yt).print();


    double xtp1[] = {k_ss, z_ss};
    //Tensor({1,1,nx,1},xtp1).print();
    Tensor xss_T({nx,1},xtp1);
    // looping over output variable
    for (int i = 0; i < nx; ++i) {
	// looping over first multiplication
	for (int j = 0; j < nx; ++j) {
	    xtp1[i] += hx[nx*i+j]*xt[j];
	    for (int k = 0; k < nx; ++k) {
		xtp1[i] += 0.5*hxx[nx*nx*i+nx*j+k]*xt[j]*xt[k];
	    }
	}
    }
    printf("xtp1 from arrays:\n");
    Tensor({1,1,nx,1},xtp1).print();

    printf("\n\n\n");

    Tensor dx_T({nx,1},xt);

    Tensor yt_T = yss_T + (gx_T * dx_T) + (0.5*((gxx_T)*(dx_T << dx_T)));
    printf("yt from Tensor:\n");
    yt_T.print();
    
    Tensor xtp1_T = xss_T + (hx_T * dx_T) + (0.5*((hxx_T)*(dx_T << dx_T)));
    printf("xtp1 from Tensor:\n");
    xtp1_T.print();
    */
    
  
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
	    delete[] derivatives[i][j];
	}
	delete[] derivatives[i];
    }
    delete[] derivatives;




  
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
    //delete[] gxx;
    //delete[] hxx;
  
    return 0;
}
