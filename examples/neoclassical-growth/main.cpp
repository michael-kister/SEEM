
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <adolc/adolc.h>

#include "tensor.hpp"

// currently holds all the solution methods
#include "perturbation.hpp"
#include "perturbation.cpp"

#include "perturbation_manager.cpp"
#include "perturbation_class.cpp"
#include "neoclassical_perturbation.cpp"

int main(int argc, char **argv) {

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
  
    double parameters[7];
    // store them all together
    parameters[0] = beta;
    parameters[1] = tau;
    parameters[2] = thxta;
    parameters[3] = alpha;
    parameters[4] = delta;
    parameters[5] = rho;
    parameters[6] = sigma;

    // instantiate model
    NeoclassicalPerturbation model(parameters);

  
    /*
    printf("\nSteady-States:\n");
    printf("l_ss : %7.4f\n", l_ss);
    printf("c_ss : %7.4f\n", c_ss);
    printf("k_ss : %7.4f\n", k_ss);
    printf("z_ss : %7.4f\n\n", z_ss);
    */
    
    
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
    
  



    return 0;
}
