#include "helper_structs.h"
#include "solution_subroutines.h"
#include "matrices.h"
#include <stdio.h>

/*
 *------------------------------------------------------------------------------
 * Let us define the following vectors of variables:
 *
 * S1(t) := { v(t-1), k(t-1) }
 *
 * S2(t) := { R(t-1), A(t), sg(t), xi(t) }
 *
 * Y1(t) := { i(t), l(t), Pi(t) }
 *
 * Y2(t) := { y(t), c(t) }
 *
 * In implementing the particle filter, we group the steps in a different way
 * compared to when we solve the model. Note that now the only one with a weird
 * index is S1, which suggests that we should change its indices, but recall
 * that k(t) explicitly requires i(t).
 *
 * Let us assume that this is our process. In that case, we can just say that we
 * observe c(t-1) and y(t-1), and call it a day. Subsequently, for a given part-
 * icle filter step, we:
 *   1. Move forward S1 (deterministically)
 *   2. Calculate lagged consumption & output
 *      - this could be considered "moving forward" these "states"
 *   3. Move the particles forward
 *      - productivity and govt share move stochastically
 *      - lagged interest rates are deterministic
 *   4. Calculate inflation, labor, 
 *      - this could be considered "moving forward" these "states"
 *   5. Calculate the reweightings
 *      - Y: normal with M.E.
 *      - C: normal with M.E.
 *      - I: normal with M.E.
 *      - H: normal with M.E.
 *      - P: normal with M.E.
 *      - R: truncated normal from xi
 *
 * In this way, we never condition on observables, because everything is cont-
 * ained in the state vector (we just happen to implement our law of motion in
 * four steps).
 *
 * Although how can you just offload xi onto the measurement equation, if it's
 * an input into the chebyshev polynomials? You would need to remove it from
 * there... which I think was doable...
 *
 * I think it's reasonable, since they're still receiving the lagged interest
 * rates.
 *
 * -----------------------------------------------------------------------------
 *
 *     { S1(t-1),          Y1(t-1)          } |--> S1(t)    |         ]
 *                                                          |         |
 *  $  { S1(t),   S2(t-1), Y1(t-1)          } |--> Y2(t-1)  ]--> t-1  |
 *                                                                    |
 *  $  {          S2(t-1), Y1(t-1), Y2(t-1) } |--> S2(t)    ]         |
 *                                                          |         |
 *  $  { S1(t),   S2(t),                    } |--> Y1(t)    |         ]--> t
 *                                                          |
 *                                                          |
 *                                                          |
 *     { S1(t),            Y1(t)            } |--> S1(t+1)  |         ]
 *                                                          |         |
 *  $  { S1(t+1), S2(t),   Y1(t+1)          } |--> Y2(t)    ]--> t    |
 *                                                                    |
 *  $  {          S2(t),   Y1(t),   Y2(t)   } |--> S2(t+1)  ]         |
 *                                                          |         |
 *  $  { S1(t+1), S2(t+1),                  } |--> Y1(t+1)  |         ]--> t+1
 *                                                          |
 *                                                          |
 *                                                          |
 *     { S1(t+1),          Y1(t+1)          } |--> S1(t+2)  |         ]
 *                                                          |         |
 *  $  { S1(t+2), S2(t+1), Y1(t+2)          } |--> Y2(t+1)  ]--> t+1  |
 *                                                                    |
 *  $  {          S2(t+1), Y1(t+1), Y2(t+1) } |--> S2(t+2)  ]         |
 *                                                          |         |
 *  $  { S1(t+2), S2(t+2),                  } |--> Y1(t+2)  |         ]--> t+2
 *     
 *------------------------------------------------------------------------------
 *
 *
 * @   { S1(t-1),          Y1(t-1)          } |--> S1(t)         \
 *     { S1(t),   S2(t-1), Y1(t-1)          } |--> Y2(t-1)        |
 * @   {          S2(t-1), Y1(t-1), Y2(t-1) } |--> S2(t)          |
 * @   { S1(t),   S2(t),                    } |--> Y1(t)         /
 * @@  { S1(t),            Y1(t)            } |--> S1(t+1)       \
 * @   { S1(t+1), S2(t),   Y1(t+1)          } |--> Y2(t)          |
 * @@  {          S2(t),   Y1(t),   Y2(t)   } |--> S2(t+1)        |
 * @@  { S1(t+1), S2(t+1),                  } |--> Y1(t+1)       /
 *
 * @   { S1(t-1),          Y1(t-1)          } |--> S1(t)         \
 *     { S1(t),   S2(t-1), Y1(t-1)          } |--> Y2(t-1)        |
 * @   {          S2(t-1), Y1(t-1), Y2(t-1) } |--> S2(t)          |
 * @   {          S2(t-1), Y1(t-1), Y2(t-1) } |--> S2(t)          |
 * @   { S1(t),   S2(t),                    } |--> Y1(t)         /
 * @@  { S1(t),            Y1(t)            } |--> S1(t+1)       \
 * @   { S1(t+1), S2(t),   Y1(t+1)          } |--> Y2(t)          |
 * @@  {          S2(t),   Y1(t),   Y2(t)   } |--> S2(t+1)        |
 * @@  {          S2(t),   Y1(t),   Y2(t)   } |--> Y3(t+1)        |
 * @@  { S1(t+1), S2(t+1),                  } |--> Y1(t+1)       /
 *
 *
 *
 * If you could go from:
 * 
 * @   { S1(t-1),          Y1(t-1)          } |--> S1(t)         \
 *     { S1(t),   S2(t-1), Y1(t-1)          } |--> Y2(t-1)        |
 * @   {          S2(t-1), Y1(t-1), Y2(t-1) } |--> S2(t)          |--> t
 * @   { S1(t),   S2(t),                    } |--> Y1(t)         /
 * @@  { S1(t),            Y1(t)            } |--> S1(t+1)  <--  \
 * @   { S1(t+1), S2(t),   Y1(t+1)          } |--> Y2(t)    <--   |
 * @@  {          S2(t),   Y1(t),   Y2(t)   } |--> S2(t+1)        |--> t+1
 * @@  { S1(t+1), S2(t+1),                  } |--> Y1(t+1)       /
 * @@@ { S1(t+1),          Y1(t+1)          } |--> S1(t+2)  <--  \
 * @@  { S1(t+2), S2(t+1), Y1(t+2)          } |--> Y2(t+1)  <--   |
 * @@@ {          S2(t+1), Y1(t+1), Y2(t+1) } |--> S2(t+2)        |--> t+2
 * @@@ { S1(t+2), S2(t+2),                  } |--> Y1(t+2)       /
 *     
 * To:
 * 
 *     { S1(t),   S2(t-1), Y1(t-1)          } |--> Y2(t-1)       /
 * @   { S1(t-1),          Y1(t-1)          } |--> S1(t)         \
 * @   {          S2(t-1), Y1(t-1), Y2(t-1) } |--> S2(t)          |
 * @   { S1(t),   S2(t),                    } |--> Y1(t)          |--> t
 * @   { S1(t+1), S2(t),   Y1(t+1)          } |--> Y2(t)    <--  /
 * @@  { S1(t),            Y1(t)            } |--> S1(t+1)  <--  \
 * @@  {          S2(t),   Y1(t),   Y2(t)   } |--> S2(t+1)        |
 * @@  { S1(t+1), S2(t+1),                  } |--> Y1(t+1)        |--> t+1
 * @@  { S1(t+2), S2(t+1), Y1(t+2)          } |--> Y2(t+1)  <--  /
 * @@@ { S1(t+1),          Y1(t+1)          } |--> S1(t+2)  <--  \
 * @@@ {          S2(t+1), Y1(t+1), Y2(t+1) } |--> S2(t+2)        |
 * @@@ { S1(t+2), S2(t+2),                  } |--> Y1(t+2)        |--> t+2
 *
 * Of course the S1 doesn't mind being moved to later, since later is always
 * easier. However, the Y2 struggles to get bumped up, since Y2 would like
 * to make use of k(t) and v(t). Is it possible that we could project them onto
 * k(t-1) and v(t-1)? Presumably yes... Although I'm hesitant to introduce two
 * additional decision rules... especially when we can avoid approximations
 * otherwise...
 *
 *  
 * 
 *------------------------------------------------------------------------------
 */


int main (int argc, char** argv)
{
    // important constants
    int num_state = 6; // number of states
    int num_decxn = 5; // number of decision rules
    int num_snglr = 3; // number of singular transitions
    int num_measr = 2; // number of measurement equations we need
    int num_fiter = 4; // number of times we iterate the decision rule

    // parameters
    double psi      = 2.0;
    double gama     = 5.0;
    double beta     = 0.991;
    double ln_P_ss  = log(1.005);
    double phi_pi   = 2.0;
    double phi_y    = 0.2;
    //double rho_R    = 0.20;
    double rho_R    = 0.80;
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
    
    double par_array[] = {psi, gama, beta, ln_P_ss, phi_pi, phi_y, rho_R,
			  epsilon, theta, alpha, zeta, delta, ln_A_ss, rho_A,
			  sigma_A, ln_sg_ss, rho_sg, sigma_sg, sigma_xi};
    
    Parameters   P(par_array);
    SteadyStates SS(P);

    SS.print();

    double v_lb  = log(1.0005);
    double v_ub  = log(1.005);
    
    double R_lb  = log(1.0005);
    //double R_ub  = log(1.05);
    double R_ub  = log(1.03);
    
    double k_lb  = log(exp(SS.ln_k_ss) - 0.05);
    double k_ub  = log(exp(SS.ln_k_ss) + 0.05);
    double A_lb  = P.ln_A_ss - 3.0*P.sigma_A;
    double A_ub  = P.ln_A_ss + 3.0*P.sigma_A;
    double sg_lb = P.ln_sg_ss - 3.0*P.sigma_sg;
    double sg_ub = P.ln_sg_ss + 3.0*P.sigma_sg;
    double xi_lb = -3.0*P.sigma_xi;
    double xi_ub = +3.0*P.sigma_xi;

    double LB[] = {v_lb, R_lb, k_lb, A_lb, sg_lb, xi_lb};
    double UB[] = {v_ub, R_ub, k_ub, A_ub, sg_ub, xi_ub};
    
    /*--------------------------------------------------------------------------
     * DECLARATIONS
     *------------------------------------------------------------------------*/
    
    // graph for obtaining approximations
    
    Graph Vfun_graph;
    Graph ifun_graph;
    Graph lfun_graph;
    Graph Pfun_graph;
    Graph xfun_graph;

    Graph* graphs[num_decxn];
    graphs[0] = &Vfun_graph;
    graphs[1] = &ifun_graph;
    graphs[2] = &lfun_graph;
    graphs[3] = &Pfun_graph;
    graphs[4] = &xfun_graph;

    // approximations to decision rules

    #define MAKEAPPROX1(x,i,j) x##fun_##i(num_state, j - num_state)
    #define MAKEAPPROX2(j,k) MAKEAPPROX1(V,j,k), \
	                     MAKEAPPROX1(i,j,k), \
	                     MAKEAPPROX1(l,j,k), \
	                     MAKEAPPROX1(P,j,k), \
	                     MAKEAPPROX1(x,j,k)
    
    Approximation MAKEAPPROX2(0,1), MAKEAPPROX2(1,1), MAKEAPPROX2(2,2), MAKEAPPROX2(3,3);

    #define PUTFUNS1(x,i,j) funs_##i[j] = &x##fun_##i
    #define PUTFUNS2(j) PUTFUNS1(V,j,0), \
	                PUTFUNS1(i,j,1), \
	                PUTFUNS1(l,j,2), \
	                PUTFUNS1(P,j,3), \
	                PUTFUNS1(x,j,4)
    
    Approximation* funs_0[num_decxn];
    Approximation* funs_1[num_decxn];
    Approximation* funs_2[num_decxn];
    Approximation* funs_3[num_decxn];
    
    PUTFUNS2(0), PUTFUNS2(1), PUTFUNS2(2), PUTFUNS2(3);

    Approximation** fun_prog[num_fiter];
    fun_prog[0] = funs_0;
    fun_prog[1] = funs_1;
    fun_prog[2] = funs_2;
    fun_prog[3] = funs_3;
    
    
    /*--------------------------------------------------------------------------
     * Solve the model
     *------------------------------------------------------------------------*/ 

    int sol_err;

    int do_solve = 1;

    if (do_solve) {

	sol_err = solve_model(P, SS, LB, UB, fun_prog,
			      graphs, num_state, num_decxn, num_snglr, num_measr);
    

	fun_prog[3][0]->save("V.appx3");
	fun_prog[3][1]->save("i.appx3");
	fun_prog[3][2]->save("l.appx3");
	fun_prog[3][3]->save("P.appx3");
	fun_prog[3][4]->save("x.appx3");

    } else {
	
	*fun_prog[3][0] = Approximation(num_state, 3-num_state, "V.appx3");
	*fun_prog[3][1] = Approximation(num_state, 3-num_state, "i.appx3");
	*fun_prog[3][2] = Approximation(num_state, 3-num_state, "l.appx3");
	*fun_prog[3][3] = Approximation(num_state, 3-num_state, "P.appx3");
	*fun_prog[3][4] = Approximation(num_state, 3-num_state, "x.appx3");
	
    }

    Point test_point{{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}};

    double tt = (*fun_prog[3][0])(test_point);
    
    printf("%f\n", tt);

    

  
    int num_terms = fun_prog[3][0]->receipt_length();

    printf("length: %d\n", num_terms);


    
    // take the solved model, and put it into a way that avoids OOP
    
    int***   decxn_indxs = new int**   [num_decxn];
    
    double** decxn_cxffs = new double* [num_decxn];
    
    for (int i = 0; i < num_decxn; ++i) {
	
	decxn_indxs[i] = new int* [num_terms];

	for (int j = 0; j < num_terms; ++j)

	    decxn_indxs[i][j] = new int [num_state];

	decxn_cxffs[i] = new double [num_terms];

	fun_prog[3][i]->print_receipt(decxn_indxs[i], decxn_cxffs[i]);

	break;
    }

    //fun_prog[0][0]->save("test_save.txt"); 

    
    for (int i = 0; i < num_terms; ++i) {

	printf("[");
	for (int j = 0; j < num_state; ++j) {
	    printf("%d,", decxn_indxs[0][i][j]);
	}
	printf("\b] %11.4e\n", decxn_cxffs[0][i]);
    }
    
    
    /*--------------------------------------------------------------------------
     * Calculate the likelihood
     *------------------------------------------------------------------------*/ 





    
    return 0;
}

    /*--------------------------------------------------------------------------
     * Write the output of the functions to a file that can be read by R
     *------------------------------------------------------------------------*/ 

/*

    int plot = 0;
    
    int num_rows = 3;
    int num_cols = 6;
    int num_gaps = 100;
    int num_points = num_gaps + 1;
    
    if (plot) {

	std::vector<double> point(6, 0.0);
	std::vector<double> stead(6, 0.0);

	stead[0] = SS.ln_v_ss;
	stead[1] = SS.ln_R_ss;
	stead[2] = SS.ln_k_ss;
	stead[3] = SS.ln_A_ss;
	stead[4] = SS.ln_sg_ss;
	stead[5] = 0.0;

	for (int i = 0; i < num_cols; i++)

	    stead[i] = MAP_SMO(stead[i], LB[i], UB[i]);

	// you have 3 inverses, each with 6 series

	double*** plot_array = new double**[num_rows];

	for (int i = 0; i < num_rows; i++) {

	    plot_array[i] = new double*[num_cols];

	    for (int j = 0; j < num_cols; j++) {

		plot_array[i][j] = new double[num_points];

		point = stead;

		point[j] = -1.0;
		
		for (int k = 0; k < num_points; k++) {

		    //printf("<%d,%d,%d>\n", i, j, k);
	    
		    point[j] += (2.0)/(num_gaps);
		    
		    plot_array[i][j][k] = (*invrs_3[i])(point);
		    
		}
	    }
	}

	FILE* fw = fopen("new_inverses_3.txt", "w");

	for (int i = 0; i < num_points; i++) {

	    for (int j = 0; j < num_rows; j++) {

		for (int k = 0; k < num_cols; k++) {

		    fprintf(fw, "% 10.6f,", plot_array[j][k][i]);
		}
	    }

	    fprintf(fw, "\b\n");

	    fflush(fw);
	}

	fclose(fw);
    }


*/


    /*
    // something to show what we took in/got out:
    printf("\n\n");    
    for (int i = 0; i < num_state; i++) {
	printf("[  ");
	for (int j = 0; j < num_state; j++) {
	    printf("%+4.1f  ", cube_points[j][i]);
	}
	printf("]\n");
    }
    printf("\n\n");
    for (int i = 0; i < num_state; i++) {
	printf("[  ");
	for (int j = 0; j < num_state; j++) {
	    printf("% 7.4f  ", para_points[j][i]);
	}
	printf("]\n");
    }
    printf("\n\n");
    */


/*------------------------------------------------------------------------------
 * OVERVIEW OF EFFICIENT IMPORTANCE SAMPLING:
 *
 * p(yt|ytm1) = ∫ p(yt|xt)*p(xt|xtm1)*p(xtm1|ytm1) dxt dxtm1
 *            = ∫ p(xt,xtm1|yt)*p(yt|ytm1)         dxt dxtm1
 *            ≈ ∑ p(yt|ytm1)
 *                                                  (drawn from p(xt,xtm1|yt))
 *            = ∑ p(yt|ytm1)*p(xt,xtm1|yt)/π(xt,xtm1|yt)
 *                                                  (drawn from π(xt,xtm1|yt))
 * 
 * So basically we would like to approximate p(xt,xtm1|yt). This is easily ob-
 * tained from the EKF, if we just include a lag of x in the state equation:
 *
 *                    [ xt   ] = [ A - ] [ xtm1 ]
 *                    [ xtm1 ]   [ I - ] [ xtm2 ]
 *
 * However, we actually want to draw from p(qt,pt,ptm1|yt), since qtm1 is det-
 * ermined by ψ(qt,pt,ptm1|yt). Therefore, our state space looks like: 
 *
 *                    [ qt   ]   [ A B - ] [ qtm1 ]
 *                    [ pt   ] = [ C D - ] [ ptm1 ]
 *                    [ ptm1 ]   [ - I - ] [ ptm2 ]
 *
 * where A, B, C, & D combined to produce our previously known transition mat-
 * rix. In our case, we can just use the first-order matrix from the pertur-
 * bation method.
 *
 * Additionally, for the EKF, we need a measurement equation. We will have
 * something that looks like:
 *
 *                    [ yt ] = [ Z - ] [ xt   ]
 *                                     [ xtm1 ]
 *
 * where Z is our previously known measurement equation. In our case, we can
 * differentiate our decision rules to obtain first-order approximations.
 *
 * Note that for the EKF we do not need to implement the measurement/trans-
 * ition equations; we will use the nonlinear equations for that. However, we
 * need these matrices in order to perform the updating steps.
 *
 * One question is how to obtain measurement equations from decision rules?
 * In our case, we would like to measure:
 *
 *   *1. Output        = (ct - it)/(1 - st)
 *   *2. Consumption   = ((1-st)/vt)*(At)*(kt^ζ)*(lt^(1-ζ)) - it
 *    3. Hours
 *    4. Inflation
 *    5. Interest rate
 *
 * Where the two marked with an asterisk(*) must be obtained.
 *
 * ADVANCED BOUNDARY CONSIDERATIONS:
 *
 * When considering the inverse mapping, we must come to terms with the fact
 * that the inverse is not defined for all values. Therefore, we must consider
 * a subset of the original region desired (and then presumably extrapolate
 * for value that lie outside of that region).
 *
 * It should be simple enough, if we have a parallelepiped, to map this into
 * a hypercube. The more interesting question is how to take an existing hyp-
 * recube, and obtain the largest possible parallelepiped within it.
 *
 * Additionally, we have some knowledge that we probably only need to consider
 * distortion between R, A, and ξ. So it's possible that I only need to find
 * the smallest possible R value for large values of these other variables.
 * Then I can just interpolate linearly...
 *
 * For my existing transformation, (from Smolyak), I use:
 *
 * X = (((b-a)*(x+1.0)/2.0)+a)
 *   = ((b-a)/2)*x + ((a+b)/2)
 *
 * which is basically a stretch (by how much wider) and a shift (by the mean
 * of the target).
 *
 * Essentially, we need a matrix that shears R in various directions, in order
 * to accommodate the limitations of the region for which there exists a sol-
 * ution. We must 
 *----------------------------------------------------------------------------*/
