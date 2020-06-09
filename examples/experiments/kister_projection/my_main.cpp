// g++ -Wall -O -g -o run_code workspace.cpp
 
#include "smolyak.h"
#include "nonlinear_solver.h"
#include "stroud.hpp"

#include <functional>
#include <iostream>
#include <ctime>

// some stuff for the perturbation method
#include <kister_perturbation.h>

#include <omp.h>


// some stuff for the backup method of solving nonlinear equations
#include "newuoa.h"

using my_fun = std::function<double(const double*, double*)>;
using pn_fun = std::function<double(long n, const double *)>;
NewuoaClosure make_closure(pn_fun& function) {
    struct Wrap {
        static double call(void *data, long n_newuoa, const double* dec_newuoa) {
            return reinterpret_cast<pn_fun *>(data)->operator()(n_newuoa, dec_newuoa);
        }
    };
    return NewuoaClosure {&function, &Wrap::call};
}



// helper macros

// change this from "abort" to something friendlier
#define VAL_CHECK(x) do { if(!isnormal(x) && x != 0.0) {  \
	    printf("x = %f at %d\n",x,__LINE__); abort(); \
	}} while(0)
#define LN_UTILITY(ln_ct, ln_lt) (alpha*ln_ct + (1.0-alpha)*log(1.0-exp(ln_lt)))
#define LN_PI_STAR(ln_Pt) ((log(1.0-theta*exp((epsilon-1.0)*ln_Pt))-log(1.0-theta))/(1.0-epsilon))
#define MAP_SMO(x,a,b) ((2.0*(x-a)/(b-a))-1.0)
#define UNMAP(x,a,b) (((b-a)*(x+1.0)/2.0)+a)

// global constants
int q             = 1;
int for_R         = 0;
int which_plot    = 2;
int say_who       = 0;
int verbose       = 0;
int max_proj_iter = 200;

// helper function to check for convergence
int is_converged(double* prev, double* curr, int dim)
{
    double tol = 1.0e-2;
    for (int i = 0; i < dim; i++)
	if (2.0*fabs(curr[i] - prev[i])/(fabs(prev[i])+fabs(curr[i])+(1.0e-4)) > tol) {
	    if (say_who)
		printf("Element-%d ==> ", i);
	    return 0;
	}
    return 1;
}

// parameters
double psi      = 2.0;
double gama     = 5.0;
double beta     = 0.991;
double ln_P_ss  = log(1.005);
double phi_pi   = 1.5;
double phi_y    = 0.25;
double rho_R    = 0.100;
double epsilon  = 6.0;
double theta    = 0.75;
double alpha    = 0.357;
double zeta     = 0.3;
double delta    = 0.0196;
double ln_A_ss  = 0.0;
double rho_A    = 0.9;
double sigma_A  = 0.0025;
double ln_sg_ss = log(0.2);
double rho_sg   = 0.8;
double sigma_sg = 0.0025;
double sigma_xi = 0.0025;

// also store the parameters as an array
double theta_array[] = {psi,      gama,   beta,     ln_P_ss, phi_pi,
			phi_y,    rho_R,  epsilon,  theta,   alpha,
			zeta,     delta,  ln_A_ss,  rho_A,   sigma_A,
			ln_sg_ss, rho_sg, sigma_sg, sigma_xi};

// steady states
double ln_Ps_ss = LN_PI_STAR(ln_P_ss);
double ln_R_ss = ln_P_ss - log(beta);
double ln_v_ss = log(1.0-theta) - epsilon*ln_Ps_ss - log(1.0-theta*exp(epsilon*ln_P_ss));
double ln_m_ss = log(beta);
double ln_mc_ss= log(1.0-beta*theta*exp(epsilon*ln_P_ss)) - log(1.0-beta*theta*exp((epsilon-1.0)*ln_P_ss)) + log((epsilon-1.0)/epsilon) + ln_Ps_ss + ln_v_ss;
double ln_rk_ss= log((1.0/beta) - 1.0 + delta);
double ln_Th_ss= (ln_rk_ss - ln_A_ss - log(zeta) - ln_mc_ss)/(zeta-1.0);
double ln_Ph_ss= log(alpha*(1.0-zeta)/(1.0-alpha)) + ln_mc_ss + ln_A_ss + zeta*ln_Th_ss - ln_v_ss;
double ln_l_ss = ln_Ph_ss - log((exp(ln_A_ss-ln_v_ss)*(1.0-exp(ln_sg_ss))*exp(zeta*ln_Th_ss)) - (delta*exp(ln_Th_ss)) + exp(ln_Ph_ss));
double ln_k_ss = ln_Th_ss + ln_l_ss;
double ln_i_ss = log(delta) + ln_k_ss;
double ln_c_ss = ln_Ph_ss + log(1.0-exp(ln_l_ss));
double ln_V_ss = LN_UTILITY(ln_c_ss, ln_l_ss);
double ln_y_ss = log(exp(ln_c_ss)+exp(ln_i_ss)) - log(1.0-exp(ln_sg_ss));
//double ln_w_ss = log(1.0-alpha) + ln_c_ss - log(alpha*(1.0-exp(ln_l_ss)));
double ln_x_ss = ln_mc_ss + ln_y_ss - ln_v_ss - log(1.0-beta*theta*exp(epsilon*ln_P_ss));

// declare this function here an specify it later
void solve_at_point
(std::vector<double>& collocation, std::vector<double>& solution,
 Approximation& Vfun, Approximation& ifun, Approximation& lfun, Approximation& Pfun, Approximation& xfun,
 double* lower_bounds, double* upper_bounds, int verbose = 0, double tol = 1.0e-14);

int main ()
{
    int i, j;

    char decision_names[] = "VilPx";
    char state_names[]    = "vRkAgx"; 
    
    printf("\n");

    printf("v  = %13.10f\n", ln_v_ss);
    printf("R  = %13.10f\n", ln_R_ss);
    printf("k  = %13.10f\n", ln_k_ss);
    printf("A  = %13.10f\n", ln_A_ss);
    printf("sg = %13.10f\n", ln_sg_ss);
    printf("xi = %13.10f\n\n", 0.0);
    
    printf("V  = %13.10f\n", ln_V_ss);
    printf("i  = %13.10f\n", ln_i_ss);
    printf("l  = %13.10f\n", ln_l_ss);
    printf("P  = %13.10f\n", ln_P_ss);
    printf("x  = %13.10f\n\n", ln_x_ss);

    // constants
    int dim = 6;
    int mu  = q - dim;

    // a grid is a set of points

    Grid G  = smolyak_grid(dim, mu);
    int num_collocation = G.size();
    int num_states = 6;
    int num_decisions = 5;
    
    // instantiate each policy function
    Approximation Vfun(dim, mu);
    Approximation ifun(dim, mu);
    Approximation lfun(dim, mu);
    Approximation Pfun(dim, mu);
    Approximation xfun(dim, mu);

    // instantiate a graph to contain values
    std::map<Point, double> Vfun_graph;
    std::map<Point, double> ifun_graph;
    std::map<Point, double> lfun_graph;
    std::map<Point, double> Pfun_graph;
    std::map<Point, double> xfun_graph;

    // vectors for states and controls
    std::vector<double> state_vec (6, 0.0);
    std::vector<double> decxn_vec (5, 0.0);
    
    // containers for examining projection (double for comparing them)
    double*** coeffs = new double**[num_decisions];
    for (int i = 0; i < num_decisions; i++) {
	coeffs[i] = new double*[2];
	for (int j = 0; j < 2; j++)
	    coeffs[i][j] = new double[num_collocation];
    }
    int** orders = new int*[num_states];
    for (int i = 0; i < num_states; i++)
	orders[i] = new int[num_collocation];

    // a useful integer
    int i_coeff;

    // define the bounds of the states
    double v_lb_f  = 0.0; // = log(1.0)
    double v_ub_f  = log(1.005);
    double R_lb_f  = 0.0;
    double R_ub_f  = log(1.03);
    double k_lb_f  = log(exp(ln_k_ss) - 0.05);
    double k_ub_f  = log(exp(ln_k_ss) + 0.05);
    double A_lb_f  = ln_A_ss - 3.0*sigma_A;
    double A_ub_f  = ln_A_ss + 3.0*sigma_A;
    double sg_lb_f = ln_sg_ss - 3.0*sigma_sg;
    double sg_ub_f = ln_sg_ss + 3.0*sigma_sg;
    double xi_lb_f = -3.0*sigma_xi;
    double xi_ub_f = +3.0*sigma_xi;

    // combine them into arrays
    double lower_bounds[6] = {v_lb_f, R_lb_f, k_lb_f, A_lb_f, sg_lb_f, xi_lb_f};
    double upper_bounds[6] = {v_ub_f, R_ub_f, k_ub_f, A_ub_f, sg_ub_f, xi_ub_f};

    //--------------------------------------------//
    //                                            // 
    //  STEP 1:  SOLVE VIA PERTURBATION           //
    //                                            // 
    //--------------------------------------------//

    // set stuff up for the perturbation method
    MPK_SGU perturbation(1);
    perturbation.load_parameters(theta_array, 0);
    
    //--------------------------------------------//
    //                                            // 
    //  STEP 2:  SOLVE VIA LOW-ORDER PROJECTION   //
    //                                            // 
    //--------------------------------------------//
    
    // something for making the plots
    int ib = 1;
    
    // iterate over improving decision rules
    for (int i = 0; i < max_proj_iter; i++) {
	    
	// go to every collocation point
	for (const auto& it : G) {

	    // convert the collocation point to a vector
	    state_vec = static_cast<std::vector<double>>(it);
		
	    if (i == 0) {

		// This is where you obtain the decisions implied by the perturbation
		// method; you'll need to unmap the state vector from [-1,1] into the
		// true space, and then scale it back by the steady states.
		//
		// Once this is done, you simply call the Taylor approximation and let
		// that determine what the rule is at each point.
		    
		for (int j = 0; j < num_states; j++)
		    state_vec[j] = UNMAP(state_vec[j], lower_bounds[j], upper_bounds[j]);

		state_vec[0] -= ln_v_ss;
		state_vec[1] -= ln_R_ss;
		state_vec[2] -= ln_k_ss;
		state_vec[3] -= ln_A_ss;
		state_vec[4] -= ln_sg_ss;
		state_vec[5] -= 0.0;

		perturbation.decision(&state_vec[0], &decxn_vec[0]);
		    
	    } else {

		// Subsequent to the first iteration, we can use the projection method
		// obtained from the last loop. We must provide an initial guess for
		// this routine, which we set as the output from the last loop's decision
		// rules.
		
		decxn_vec[0] = Vfun(state_vec);
		decxn_vec[1] = ifun(state_vec);
		decxn_vec[2] = lfun(state_vec);
		decxn_vec[3] = Pfun(state_vec);
		decxn_vec[4] = xfun(state_vec);

		solve_at_point(state_vec, decxn_vec, Vfun, ifun, lfun, Pfun, xfun, lower_bounds, upper_bounds, verbose);
		    
	    }
	    
	    // Once we have obtained the decisions for a given point, we store it in a graph
	    // that will eventually be used to produce the chebyshev coefficients.
	    
	    Vfun_graph.insert(std::pair<Point,double>(it, decxn_vec[0]));
	    ifun_graph.insert(std::pair<Point,double>(it, decxn_vec[1]));
	    lfun_graph.insert(std::pair<Point,double>(it, decxn_vec[2]));
	    Pfun_graph.insert(std::pair<Point,double>(it, decxn_vec[3]));
	    xfun_graph.insert(std::pair<Point,double>(it, decxn_vec[4]));
	}

	// determine projection coefficients for Chebyshev projection
	Vfun.set_coefficients(Vfun_graph);
	ifun.set_coefficients(ifun_graph);
	lfun.set_coefficients(lfun_graph);
	Pfun.set_coefficients(Pfun_graph);
	xfun.set_coefficients(xfun_graph);
	    
	// clear the graphs to start fresh
	Vfun_graph.clear();
	ifun_graph.clear();
	lfun_graph.clear();
	Pfun_graph.clear();
	xfun_graph.clear();

	// obtain the coefficients (and store them in either of two places)
	i_coeff = i % 2;
	Vfun.print_receipt(coeffs[0][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
	ifun.print_receipt(coeffs[1][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
	lfun.print_receipt(coeffs[2][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
	Pfun.print_receipt(coeffs[3][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
	xfun.print_receipt(coeffs[4][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);

	// print the output to be plotted in R
	if (for_R == 1) {
	    
	    printf("bounds[[%d]] <- list(", ib);
	    for (int j = 0; j < 6; j++)
		printf("c(%13.10f,%13.10f), ", lower_bounds[j], upper_bounds[j]);
	    printf("\b\b)\n");
		
	    printf("coeffs[[%d]] <- c(", ib++);
	    for (int j = 0; j < num_collocation; j++)
		printf("%+11.4e, ", coeffs[which_plot][i_coeff][j]);
	    printf("\b\b)\n");
	}
	
	// check whether the approximation has converged
	if (i > 0) {
	    for (int j = 0; j < 5; j++) {
		if (!is_converged(coeffs[j][0], coeffs[j][1], num_collocation)) {
		    if (say_who)
			printf("%c says go again!\n", decision_names[j]);
		    goto loop_again;
		}
	    }
	    break;
	}
    loop_again: ;
	
    } // looping over decision

    printf("\n");

    for_R = 1;
    if (for_R)
	printf("# Switching to higher order now.\n\n");
    for_R = 0;
    // clean up containers of specific size
    // free up dynamically allocated memory
    for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 2; j++) {
	    delete[] coeffs[i][j];
	}
	delete[] coeffs[i];
    }
    //delete[] coeffs;
    
    for (int i = 0; i < 6; i++) {
	delete[] orders[i];
    }
    //delete[] orders;

    // this is just a little thing if you want to see what goes on at the last location
    /*
    verbose = 1;
    // if you want to see the last stuff
    for (const auto& it : G) {

	// convert the collocation point to a vector
	state_vec = static_cast<std::vector<double>>(it);

	decxn_vec[0] = Vfun(state_vec);
	decxn_vec[1] = ifun(state_vec);
	decxn_vec[2] = lfun(state_vec);
	decxn_vec[3] = Pfun(state_vec);
	decxn_vec[4] = xfun(state_vec);

	solve_at_point(state_vec, decxn_vec, Vfun, ifun, lfun, Pfun, xfun, lb_i, ub_i, lb_i, ub_i, verbose);
    }
    return 0;
    */
    
    //--------------------------------------------//
    //                                            // 
    //  STEP 3:  SOLVE VIA HIGH-ORDER PROJECTION  //
    //                                            // 
    //--------------------------------------------//

    // increment mu from 1 to 3
    mu += 1;
    G = smolyak_grid(dim, mu);
    num_collocation = G.size();

    // make some new approximations for the higher order
    Approximation Vfun2(dim, mu);
    Approximation ifun2(dim, mu);
    Approximation lfun2(dim, mu);
    Approximation Pfun2(dim, mu);
    Approximation xfun2(dim, mu);



    // make them shared, since master is the only one who is allowed to touch them
    // dynamically allocated memory for each CPU
    for (int i = 0; i < 5; i++) {
	coeffs[i] = new double*[2];
	for (int j = 0; j < 2; j++)
	    coeffs[i][j] = new double[num_collocation];
    }
    for (int i = 0; i < 6; i++)
	orders[i] = new int[num_collocation];

    int iter = 0;
    int stop = 0;
    
    #pragma omp parallel
    {

	Point it_par = *G.begin();
	std::vector<double> state_vec_par (6, 0.0);
	std::vector<double> decxn_vec_par (5, 0.0);

	
	// iterate over improving decision rules
	while (!stop) {

            #pragma omp for
	    for (int j = 0; j < num_collocation; j++) {

		it_par = G.element(j);
		
		state_vec_par = static_cast<std::vector<double>>(it_par);
		
		if (iter == 0) {

		    // If it's the first loop, just insert the values determined by the
		    // low-order projection approximation.

                    #pragma omp critical
		    {
			Vfun_graph.insert(std::pair<Point,double>(it_par, Vfun(state_vec_par)));
			ifun_graph.insert(std::pair<Point,double>(it_par, ifun(state_vec_par)));
			lfun_graph.insert(std::pair<Point,double>(it_par, lfun(state_vec_par)));
			Pfun_graph.insert(std::pair<Point,double>(it_par, Pfun(state_vec_par)));
			xfun_graph.insert(std::pair<Point,double>(it_par, xfun(state_vec_par)));
		    }
		
		} else {

		    // In subsequent interations, we do the regular stuff.
		    
		    decxn_vec_par[0] = Vfun2(state_vec_par);
		    decxn_vec_par[1] = ifun2(state_vec_par);
		    decxn_vec_par[2] = lfun2(state_vec_par);
		    decxn_vec_par[3] = Pfun2(state_vec_par);
		    decxn_vec_par[4] = xfun2(state_vec_par);
		    
		    solve_at_point(state_vec_par, decxn_vec_par, Vfun2, ifun2, lfun2, Pfun2, xfun2, lower_bounds, upper_bounds, verbose);
		    
                    #pragma omp critical
		    {
			Vfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[0]));
			ifun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[1]));
			lfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[2]));
			Pfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[3]));
			xfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[4]));
		    }
		}
	    } // parallel for loop
	    
	    #pragma omp master
	    {
		// determine the projection coefficients for Chebyshev projection
		Vfun2.set_coefficients(Vfun_graph);
		ifun2.set_coefficients(ifun_graph);
		lfun2.set_coefficients(lfun_graph);
		Pfun2.set_coefficients(Pfun_graph);
		xfun2.set_coefficients(xfun_graph);

		// clear the graphs to start fresh
		Vfun_graph.clear();
		ifun_graph.clear();
		lfun_graph.clear();
		Pfun_graph.clear();
		xfun_graph.clear();
		
		// obtain the coefficients (and store them in either of two places)
		i_coeff = iter % 2;
		Vfun2.print_receipt(coeffs[0][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		ifun2.print_receipt(coeffs[1][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		lfun2.print_receipt(coeffs[2][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		Pfun2.print_receipt(coeffs[3][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		xfun2.print_receipt(coeffs[4][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
	
		// print the output to be plotted in R
		if (for_R == 1) {
	    
		    printf("bounds[[%d]] <- list(", ib);
		    for (int j = 0; j < 6; j++)
			printf("c(%13.10f,%13.10f), ", lower_bounds[j], upper_bounds[j]);
		    printf("\b\b)\n");
		
		    printf("coeffs[[%d]] <- c(", ib++);
		    for (int j = 0; j < num_collocation; j++)
			printf("%+11.4e, ", coeffs[which_plot][i_coeff][j]);
		    printf("\b\b)\n");
		}
	
		// check whether the approximation has converged
		if (iter > 0)
		    if (is_converged(coeffs[0][0], coeffs[0][1], num_collocation) &&
			is_converged(coeffs[1][0], coeffs[1][1], num_collocation) &&
			is_converged(coeffs[2][0], coeffs[2][1], num_collocation) &&
			is_converged(coeffs[3][0], coeffs[3][1], num_collocation) &&
			is_converged(coeffs[4][0], coeffs[4][1], num_collocation))
			
			stop = 1;

		if (++iter > max_proj_iter)

		    stop = 1;
		
	    } // only master checks for convergence
	    
        #pragma omp barrier
	    
	} // looping over decision

    } // pragma omp parallel
    
    for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 2; j++) {
	    delete[] coeffs[i][j];
	}
	delete[] coeffs[i];
    }
    for (int i = 0; i < 6; i++) {
	delete[] orders[i];
    }
    
    for_R = 1;
    if (for_R)
	printf("# Switching to higher order now.\n\n");

    
    //----------------------------------------------//
    //                                              // 
    //  STEP 4:  SOLVE VIA HIGHER-ORDER PROJECTION  //
    //                                              // 
    //----------------------------------------------//

    // increment mu from 1 to 3
    mu += 1;
    G = smolyak_grid(dim, mu);
    num_collocation = G.size();

    // make some new approximations for the higher order
    Approximation Vfun22(dim, mu);
    Approximation ifun22(dim, mu);
    Approximation lfun22(dim, mu);
    Approximation Pfun22(dim, mu);
    Approximation xfun22(dim, mu);



    // make them shared, since master is the only one who is allowed to touch them
    // dynamically allocated memory for each CPU
    for (int i = 0; i < 5; i++) {
	coeffs[i] = new double*[2];
	for (int j = 0; j < 2; j++)
	    coeffs[i][j] = new double[num_collocation];
    }
    for (int i = 0; i < 6; i++)
	orders[i] = new int[num_collocation];

    iter = 0;
    stop = 0;
    
    #pragma omp parallel
    {

	Point it_par = *G.begin();
	std::vector<double> state_vec_par (6, 0.0);
	std::vector<double> decxn_vec_par (5, 0.0);

	
	// iterate over improving decision rules
	while (!stop) {

            #pragma omp for
	    for (int j = 0; j < num_collocation; j++) {

		it_par = G.element(j);
		
		state_vec_par = static_cast<std::vector<double>>(it_par);
		
		if (iter == 0) {

		    // If it's the first loop, just insert the values determined by the
		    // low-order projection approximation.

                    #pragma omp critical
		    {
			Vfun_graph.insert(std::pair<Point,double>(it_par, Vfun2(state_vec_par)));
			ifun_graph.insert(std::pair<Point,double>(it_par, ifun2(state_vec_par)));
			lfun_graph.insert(std::pair<Point,double>(it_par, lfun2(state_vec_par)));
			Pfun_graph.insert(std::pair<Point,double>(it_par, Pfun2(state_vec_par)));
			xfun_graph.insert(std::pair<Point,double>(it_par, xfun2(state_vec_par)));
		    }
		
		} else {

		    // In subsequent interations, we do the regular stuff.
		    
		    decxn_vec_par[0] = Vfun22(state_vec_par);
		    decxn_vec_par[1] = ifun22(state_vec_par);
		    decxn_vec_par[2] = lfun22(state_vec_par);
		    decxn_vec_par[3] = Pfun22(state_vec_par);
		    decxn_vec_par[4] = xfun22(state_vec_par);
		    
		    solve_at_point(state_vec_par, decxn_vec_par, Vfun22, ifun22, lfun22, Pfun22, xfun22, lower_bounds, upper_bounds, verbose);
		    
                    #pragma omp critical
		    {
			Vfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[0]));
			ifun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[1]));
			lfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[2]));
			Pfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[3]));
			xfun_graph.insert(std::pair<Point,double>(it_par, decxn_vec_par[4]));
		    }
		}
	    } // parallel for loop
	    
	    #pragma omp master
	    {
		// determine the projection coefficients for Chebyshev projection
		Vfun22.set_coefficients(Vfun_graph);
		ifun22.set_coefficients(ifun_graph);
		lfun22.set_coefficients(lfun_graph);
		Pfun22.set_coefficients(Pfun_graph);
		xfun22.set_coefficients(xfun_graph);

		// clear the graphs to start fresh
		Vfun_graph.clear();
		ifun_graph.clear();
		lfun_graph.clear();
		Pfun_graph.clear();
		xfun_graph.clear();
		
		// obtain the coefficients (and store them in either of two places)
		i_coeff = iter % 2;
		Vfun22.print_receipt(coeffs[0][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		ifun22.print_receipt(coeffs[1][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		lfun22.print_receipt(coeffs[2][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		Pfun22.print_receipt(coeffs[3][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
		xfun22.print_receipt(coeffs[4][i_coeff], orders[0], orders[1], orders[2], orders[3], orders[4], orders[5]);
	
		// print the output to be plotted in R
		if (for_R == 1) {
	    
		    printf("bounds[[%d]] <- list(", ib);
		    for (int j = 0; j < 6; j++)
			printf("c(%13.10f,%13.10f), ", lower_bounds[j], upper_bounds[j]);
		    printf("\b\b)\n");
		
		    printf("coeffs[[%d]] <- c(", ib++);
		    for (int j = 0; j < num_collocation; j++)
			printf("%+11.4e, ", coeffs[which_plot][i_coeff][j]);
		    printf("\b\b)\n");
		}
	
		// check whether the approximation has converged
		if (iter > 0)
		    if (is_converged(coeffs[0][0], coeffs[0][1], num_collocation) &&
			is_converged(coeffs[1][0], coeffs[1][1], num_collocation) &&
			is_converged(coeffs[2][0], coeffs[2][1], num_collocation) &&
			is_converged(coeffs[3][0], coeffs[3][1], num_collocation) &&
			is_converged(coeffs[4][0], coeffs[4][1], num_collocation))
			
			stop = 1;

		if (++iter > max_proj_iter)

		    stop = 1;
		
	    } // only master checks for convergence
	    
        #pragma omp barrier
	    
	} // looping over decision

    } // pragma omp parallel
    
    for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 2; j++) {
	    delete[] coeffs[i][j];
	}
	delete[] coeffs[i];
    }
    delete[] coeffs;
    for (int i = 0; i < 6; i++) {
	delete[] orders[i];
    }
    delete[] orders;
    
    return 0;

}


void solve_at_point
(std::vector<double>& collocation, std::vector<double>& solution,
 Approximation& Vfun, Approximation& ifun, Approximation& lfun, Approximation& Pfun, Approximation& xfun,
 double* lower_bounds, double* upper_bounds, int verbose, double tol)
{
    
    // constants
    int dim = 3;
    double pi  = 3.141592653589793E+00;
    double sq2 = 1.414213562373095E+00;
    double q_scale = pow(pi, -0.5*dim);

    // obtain quadrature
    int o = en_r2_11_1_size(dim);
    double* q_loc = new double [dim*o];
    double* w_val = new double [o];
    en_r2_11_1(3, 1, o, q_loc, w_val);
    
    // read in collocation point [-1,1] -> [lower_bounds,upper_bounds]
    double ln_vtm1 = UNMAP(collocation[0], lower_bounds[0], upper_bounds[0]);
    double ln_Rtm1 = UNMAP(collocation[1], lower_bounds[1], upper_bounds[1]);
    double ln_ktm1 = UNMAP(collocation[2], lower_bounds[2], upper_bounds[2]);
    double ln_At   = UNMAP(collocation[3], lower_bounds[3], upper_bounds[3]);
    double ln_sgt  = UNMAP(collocation[4], lower_bounds[4], upper_bounds[4]);
    double ln_xit  = UNMAP(collocation[5], lower_bounds[5], upper_bounds[5]);

    if (verbose) {
	printf("StdSt: %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f\n", ln_v_ss, ln_R_ss, ln_k_ss, ln_A_ss, ln_sg_ss, 0.0);
	printf("Point: %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f\n", ln_vtm1, ln_Rtm1, ln_ktm1, ln_At,   ln_sgt,   ln_xit);
    }
 
    // mean of log for expectations
    double mu_ln_A  = (1.0-rho_A)*ln_A_ss   + rho_A*ln_At;
    double mu_ln_sg = (1.0-rho_sg)*ln_sg_ss + rho_sg*ln_sgt;
    
    // set the lambda to be passed into nonlinear solver
    my_fun FUN = [&](const double* DEC, double* ERR) -> double {

	// vector for collocation
	std::vector<double> coltp1 (6, 0.0);
	
	// read in possible time-t values
	double ln_Vt = DEC[0] + ln_V_ss;
	double ln_it = DEC[1] + ln_i_ss;
	double ln_lt = DEC[2] + ln_l_ss;
	double ln_Pt = DEC[3] + ln_P_ss;
	double ln_xt = DEC[4] + ln_x_ss;
	
	// expectations/integrals
	double Int1 = 0.0;
	double Int2 = 0.0;
	double Int3 = 0.0;
	double Int4 = 0.0;
	double Int5 = 0.0;
	
	// obtain determined variables for solution
	//       it -> ct -> wt -> mct -> rkt
	// P* -> vt /     -> ut -> Ot
	//                -> yt -> Zt -> Rt
	double ln_kt = log(exp(ln_it) + (1.0-delta)*exp(ln_ktm1));
	double ln_Pst= LN_PI_STAR(ln_Pt);
	double ln_vt = log((1.0-theta)*exp(-1.0*epsilon*ln_Pst) + theta*exp(epsilon*ln_Pt + ln_vtm1));
	double ln_ct = log((1.0-exp(ln_sgt))*exp(ln_At + zeta*ln_kt + (1.0-zeta)*ln_lt - ln_vt) - exp(ln_it));
	double ln_wt = log((1.0-alpha)/alpha) + ln_ct - log(1.0-exp(ln_lt));
	double ln_mct= ln_wt - log(1.0-zeta) - ln_At - zeta*(ln_kt-ln_lt) + ln_vt;
	double ln_Ot = (1.0-psi)*LN_UTILITY(ln_ct, ln_lt) - ln_ct;
	double ln_yt = log(exp(ln_ct)+exp(ln_it)) - log(1.0-exp(ln_sgt));
	double ln_Zt = rho_R*ln_Rtm1 + (1.0-rho_R)*(ln_P_ss-log(beta) + phi_pi*(ln_Pt-ln_P_ss) + phi_y*(ln_yt-ln_y_ss)) + ln_xit;
	double ln_Rt = (ln_Zt > 0.0) ? ln_Zt : 0.0;
	
	// stochastic state variables
	double ln_Atp1, ln_sgtp1, ln_xitp1;

	// decision variables at time t+1
	double ln_Vtp1, ln_itp1, ln_ltp1, ln_Ptp1, ln_xtp1;

	// determined variables at time t+1
	double ln_ktp1, ln_Pstp1, ln_vtp1, ln_ctp1, ln_wtp1, ln_mctp1, ln_rktp1, ln_Otp1;
	double ln_v_g_p; // (\gamma-\psi)*log(V_{t+1})
	double ln_p_eps; // \eps * log(\Pi_{t})
	
	// set deterministic (or decision) states
	coltp1[0] = MAP_SMO(ln_vt, lower_bounds[0], upper_bounds[0]);
	coltp1[1] = MAP_SMO(ln_Rt, lower_bounds[1], upper_bounds[1]);
	coltp1[2] = MAP_SMO(ln_kt, lower_bounds[2], upper_bounds[2]);

	// iterate over each point in the quadrature
	for (int i = 0; i < o; i++) {
	    
	    // location at which to perform piece of quadrature
	    ln_Atp1  = (sigma_A /sq2) * q_loc[i*dim+0] + mu_ln_A;
	    ln_sgtp1 = (sigma_sg/sq2) * q_loc[i*dim+1] + mu_ln_sg;
	    ln_xitp1 = (sigma_xi/sq2) * q_loc[i*dim+2];

	    // map everything back into [-1,1] (for chebyshev polynomials)
	    coltp1[3] = MAP_SMO(ln_Atp1,  lower_bounds[3], upper_bounds[3]);
	    coltp1[4] = MAP_SMO(ln_sgtp1, lower_bounds[4], upper_bounds[4]);
	    coltp1[5] = MAP_SMO(ln_xitp1, lower_bounds[5], upper_bounds[5]);

	    // obtain some future decisions for integrals
	    ln_Vtp1 = Vfun(coltp1) + ln_V_ss;
	    ln_itp1 = ifun(coltp1) + ln_i_ss;
	    ln_ltp1 = lfun(coltp1) + ln_l_ss;
	    ln_Ptp1 = Pfun(coltp1) + ln_P_ss;
	    ln_xtp1 = xfun(coltp1) + ln_x_ss;

	    // obtain algebraic states
	    ln_ktp1 = log(exp(ln_itp1) + (1.0-delta)*exp(ln_kt));
	    ln_Pstp1= LN_PI_STAR(ln_Ptp1);
	    ln_vtp1 = log((1.0-theta)*exp(-1.0*epsilon*ln_Pstp1) + theta*exp(epsilon*ln_Ptp1 + ln_vt));
	    ln_ctp1 = log((1.0-exp(ln_sgtp1))*exp(ln_Atp1 + zeta*ln_ktp1 + (1.0-zeta)*ln_ltp1 - ln_vtp1) - exp(ln_itp1));
	    ln_wtp1 = log((1.0-alpha)/alpha) + ln_ctp1 - log(1.0-exp(ln_ltp1));
	    ln_mctp1= ln_wtp1 - log(1.0-zeta) - ln_Atp1 - zeta*(ln_ktp1-ln_ltp1) + ln_vtp1;
	    ln_rktp1= ln_mctp1 + log(zeta) + ln_Atp1 + (zeta-1.0)*(ln_ktp1-ln_ltp1) - ln_vtp1;
	    ln_Otp1 = (1.0-psi)*LN_UTILITY(ln_ctp1, ln_ltp1) - ln_ctp1;

	    // implement some powers
	    ln_v_g_p = (gama-psi)*ln_Vtp1;
	    ln_p_eps = epsilon*ln_Ptp1;
	    
	    // construction of integrals
	    Int1 += w_val[i] * exp((1.0-gama)*ln_Vtp1);
	    Int2 += w_val[i] * exp(ln_v_g_p + ln_Otp1 - ln_Ptp1);
	    Int3 += w_val[i] * exp(ln_v_g_p + ln_Otp1 + log(exp(ln_rktp1) + 1.0 - delta));
	    Int4 += w_val[i] * exp(ln_v_g_p + ln_Otp1 + ln_p_eps + ln_xtp1);
	    Int5 += w_val[i] * exp(ln_v_g_p + ln_Otp1 + ln_p_eps + ln_xtp1 - ln_Ptp1 - ln_Pstp1);
	}
	
	// scale integrals
	Int1 *= q_scale;
	Int2 *= q_scale;
	Int3 *= q_scale;
	Int4 *= q_scale;
	Int5 *= q_scale;
	
	// redefine helper variable
	ln_v_g_p = (gama-psi)*log(Int1)/(1.0-gama);
	
	// we would like to find where this is zero
	ERR[0] = (1.0-psi)*ln_Vt - log((1.0-beta)*exp(ln_Ot+ln_ct) + (beta*pow(Int1,(1.0-psi)/(1.0-gama))));
	ERR[1] = ln_v_g_p + ln_Ot - log(beta) - ln_Rt - log(Int2);
	ERR[2] = ln_v_g_p + ln_Ot - log(beta) - log(Int3);
	ERR[3] = ln_v_g_p + ln_Ot + log(exp(ln_xt)-exp(ln_mct+ln_yt-ln_vt)) - log(theta*beta*Int4);
	ERR[4] = ln_v_g_p + ln_Ot + log(exp(ln_xt)-((epsilon-1.0)/epsilon)*exp(ln_Pst+ln_yt)) - ln_Pst - log(theta*beta*Int5);

	// check if any of them are NaN-- don't do anything, but alert the user.
	VAL_CHECK(ERR[0]);
	VAL_CHECK(ERR[1]);
	VAL_CHECK(ERR[2]);
	VAL_CHECK(ERR[3]);
	VAL_CHECK(ERR[4]);
	
	// play nice in case we want to use a minimization routine
	double out_norm = 0.0;
	for (int i = 0; i < 5; i++)
	    out_norm += ERR[i]*ERR[i];
	if (!isnormal(out_norm) && out_norm != 0.0)
	    out_norm = 1.0e10;

	return out_norm;
    };

    // make decision an array to be friendly
    double* arr = &solution[0];
    
    // solve for the decision
    if (verbose) {
	printf("S: [ % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f ]\n",
	       ln_V_ss,          ln_i_ss,          ln_l_ss,          ln_P_ss,          ln_x_ss);
	printf("I: [ % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f ]\n",
	       arr[0] + ln_V_ss, arr[1] + ln_i_ss, arr[2] + ln_l_ss, arr[3] + ln_P_ss, arr[4] + ln_x_ss);
    }

    // first attempt to obtain the solution at the point
    int sol_err = broydens_method(FUN, arr, 5, 50, tol);

    // if it doesn't work, try another method
    if (sol_err != 0) {
	
	printf("# Broyden's method failed; try Powell's method. ");
	
	const long newuoa_num_vars = 5;
	const long newuoa_num_cond = (newuoa_num_vars+1)*(newuoa_num_vars+2)/2;
	const double newuoa_radius = 1.0e-6;
	const double newuoa_finrad = 1.0e-13;
	const long newuoa_max_iter = 100;
	const std::size_t newuoasz = NEWUOA_WORKING_SPACE_SIZE(newuoa_num_vars, newuoa_num_cond);
	double newuoa_workingspace[newuoasz];

	std::size_t newuoa_funcount = 0;

	double err_newuoa[5];
	
	pn_fun FUN2 = [&](long n_newuoa, const double* dec_newuoa) {
	    ++newuoa_funcount;
	    return FUN(dec_newuoa, err_newuoa);
	};
	
	NewuoaClosure closure = make_closure(FUN2);
	
	double result = newuoa_closure
	    (&closure,
	     newuoa_num_vars,
	     newuoa_num_cond,
	     arr,
	     newuoa_radius,
	     newuoa_finrad,
	     newuoa_max_iter,
	     newuoa_workingspace);
	
	double check_res = FUN(arr, err_newuoa);
	
	printf("Results: %.12f\n", check_res);
	
    }
    
    if (verbose)
	printf("O: [ % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f ]\n\n",
	       arr[0] + ln_V_ss, arr[1] + ln_i_ss, arr[2] + ln_l_ss, arr[3] + ln_P_ss, arr[4] + ln_x_ss);
    
    delete[] w_val;
    delete[] q_loc;
}
