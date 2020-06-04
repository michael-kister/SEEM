
#include "solution_subroutines.h"

int is_converged(const double* prev, const double* curr, int dim)
{
    double tol = 3.0e-6;
    double norm1 = 0.0;
    double norm2 = 0.0;
    double shift = 0.0;
    for (int i = 0; i < dim; i++) {
	norm1 += prev[i]*prev[i];
	norm2 += curr[i]*curr[i];
	shift -= curr[i]*prev[i]*2.0;
    }
    // compare the squared distance to the squared mean of the norms
    double check = 4.0*(norm1 + norm2 + shift)/(norm1 + norm2 + 2.0*sqrt(norm1*norm2));
    return (check > tol) ? 0 : 1;
}


int solve_model
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB, 
 Approximation*** funs_all, Graph** graphs,
 int num_state, int num_decxn, int num_snglr, int num_meas)
{
    ArgVec<SMOLYAK> state_vec (num_state, 0.0);
    ArgVec<TILDE> decxn_vec (num_decxn, 0.0);

    double*** coeffs = new double**[num_decxn];
    
    MPK_SGU perturbation(0); // "0" refers to the tape
    std::vector<double> pert_pars = static_cast<std::vector<double>>(P);
    perturbation.load_parameters(&pert_pars[0], 0);
    perturbation.display(2);
    abort();
    
    Grid G  = smolyak_grid(num_state, 1 - num_state);
    
    for (const auto& it : G) {
	
	state_vec = static_cast<ArgVec<SMOLYAK>>(it);
	for (int j = 0; j < num_state; j++)
	    state_vec[j] = UNMAP(state_vec[j], LB[j], UB[j]);

	state_vec[0] -= SS.ln_v_ss;
	state_vec[1] -= SS.ln_R_ss;
	state_vec[2] -= SS.ln_k_ss;
	state_vec[3] -= P.ln_A_ss;
	state_vec[4] -= P.ln_sg_ss;
	state_vec[5] -= 0.0;
	
	perturbation.decision(&state_vec[0], &decxn_vec[0]);

	for (int k = 0; k < num_decxn; k++)
	    graphs[k]->insert(std::pair<Point,double>(it, decxn_vec[k]));
    }
    
    for (int k = 0; k < num_decxn; k++)
	funs_all[0][k]->set_coefficients(graphs[k]);

    int error    = 0;
    int num_iter = 0;
    
    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    int* error_hist = new int[4];
    error_hist[0] = 0;

    int num_level = 4;
    
    for (int q = 1; q < num_level; q++) {

	G  = smolyak_grid(num_state, q - num_state);
	
	for (int qq = q-1; qq >= 0; --qq) {

	    if (error_hist[qq])
		
		continue;
	    
	    start = high_resolution_clock::now();

	    error = iterate_decision_rules
		(P, SS, LB, UB, funs_all[qq], funs_all[q], graphs, G, coeffs,
		 num_state, num_decxn, q, 0, 0, q, &num_iter);

	    error_hist[q] = error;

	    stop = high_resolution_clock::now();
	    
	    duration = duration_cast<milliseconds>(stop - start);

	    printf("Level %d -> %d : %5d ms, %4d iterations ---> ",
		   qq, q, (int)duration.count(), num_iter);
	    
	    if (error) {
		printf("(FAILED)\n\n");
	    } else {
		printf("(SUCCEEDED)\n\n");
	    }
	    
	    if (!error)

		break;
	}
    }
    
    delete[] coeffs;

    return error;
}


int iterate_decision_rules
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 Approximation** funs_in, Approximation** funs_out, Graph** graphs, Grid& G, double*** coeffs,
 int num_state, int num_decxn, int smolyak_q, int for_R, int which_plot, int integration_level,
 int* num_iter_out)
{
    
    int num_collocation = G.size();
    int max_proj_iter = 200;//(smolyak_q == 2) ? 40 : 200;
    
    // allocate space for the coefficients (and order arrays)
    for (int i = 0; i < num_decxn; i++) {
	coeffs[i] = new double*[2];
	for (int j = 0; j < 2; j++)
	    coeffs[i][j] = new double[num_collocation];
    }

    // clear the graphs to start fresh
    for (int i = 0; i < num_decxn; i++)
	graphs[i]->clear();
    
    // some stopping criterion
    int iter = 0;
    int stop = 0;
    int i_coeff;
    int ib = 1;

    // make a graph to hold whether or not you succeeded
    std::map<Point, int> error_map;
    int num_error = 0;
    int num_error_prev;
    
    // booleans relevant for sweeping
    int available = 0;
    int transplant_error;

    // our two patients
    Point donor     = *G.begin();
    Point recipient = *G.begin();

    // and their respective solutions
    ArgVec<TILDE> donated(num_decxn, 0.0);
    ArgVec<TILDE> received(num_decxn, 0.0);
    
    #pragma omp parallel //num_threads(1)
    {
	int error;
	Point it_par = *G.begin();
	ArgVec<SMOLYAK> state_vec_par (num_state, 0.0);
	ArgVec<TILDE> decxn_vec_par (num_decxn, 0.0);

	while (!stop) {
	    
            #pragma omp for
	    for (int j = 0; j < num_collocation; j++) {
		it_par = G.element(j);
		state_vec_par = static_cast<ArgVec<SMOLYAK>>(it_par);
		
		if (iter == 0) {
                    #pragma omp critical
		    {
			for (int k = 0; k < num_decxn; k++)
			    graphs[k]->insert(std::pair<Point,double>(it_par, (*funs_in[k])(state_vec_par)));
		    }
		} else {
		    
		    for (int k = 0; k < num_decxn; k++)
			decxn_vec_par[k] = (*funs_out[k])(state_vec_par);

		    error = solve_at_point
			(P, SS, LB, UB, state_vec_par, decxn_vec_par, funs_out,
			 num_state, num_decxn, integration_level, 0);
		    
                    #pragma omp critical
		    {
			if (error) {
			    ++num_error;
			    //printf("%2d ", iter);
			    //it_par.print();
			} else {
			    for (int k = 0; k < num_decxn; k++)
				graphs[k]->insert(std::pair<Point,double>(it_par, decxn_vec_par[k]));
			}
		    }
		}
	    }
	    
	    #pragma omp master
	    {
		if (num_error) {
		    stop = 1;
		} else {
		    for (int k = 0; k < num_decxn; k++) {
			funs_out[k]->set_coefficients(graphs[k]);
			graphs[k]->clear();
			funs_out[k]->print_receipt(coeffs[k][iter % 2]);
		    }
		    // check whether the approximation has converged
		    if (iter > 0 &&
			is_converged(coeffs[0][0], coeffs[0][1], num_collocation) &&
			is_converged(coeffs[1][0], coeffs[1][1], num_collocation) &&
			is_converged(coeffs[2][0], coeffs[2][1], num_collocation) &&
			is_converged(coeffs[3][0], coeffs[3][1], num_collocation) &&
			is_converged(coeffs[4][0], coeffs[4][1], num_collocation)) {
			stop = 1;
		    } else if (++iter > max_proj_iter) {
			stop = 1;
		    }
		}
	    }
            #pragma omp barrier
	}
    } // pragma omp parallel
    
    for (int i = 0; i < num_decxn; i++) {
	for (int j = 0; j < 2; j++) {
	    delete[] coeffs[i][j];
	}
	delete[] coeffs[i];
    }

    // maybe we're curious how many times it took
    num_iter_out[0] = iter;
    
    if (num_error)
	return num_error;
    else
	return 0;
}

/*
 *------------------------------------------------------------------------------
 * We have our state vector, and then our decision rules, and our observation
 * vector, but really there are four steps that go into an iteration of any
 * transition algorithm.
 *
 * S1(t) := { v(t-1), k(t-1) }
 *
 * S2(t) := { R(t-1), A(t), sg(t), xi(t) }
 *
 * Y1(t) := { i(t), l(t), Pi(t) }
 *
 * Y2(t) := { y(t), c(t) }
 *
 * I'm actually not sure if I'm happy thinking about this in the following way,
 * but I don't think it's the worst? Essentially this is how you'll do it for
 * the particle filter.
 *
 * When you're solving the model, you enter this flow effectively just having
 * completed step
 * 
 * -----------------------------------------------------------------------------
 *
 *     { S1(t-1),          Y1(t-1)          } |--> S1(t)   
 *                                                         
 *     { S1(t),   S2(t-1), Y1(t-1)          } |--> Y2(t-1) 
 *                                                        
 *     {          S2(t-1), Y1(t-1), Y2(t-1) } |--> S2(t)   
 * 
 *     { S1(t),   S2(t),                    } |--> Y1(t)   
 *
 * ------------------------<<<START SOLVER>>>-----------------------------------
 * 
 *     { S1(t),            Y1(t)            } |--> S1(t+1)     STEP I
 *                                                          
 *     { S1(t+1), S2(t),   Y1(t+1)          } |--> Y2(t)       STEP II
 *                                                         
 *     {          S2(t),   Y1(t),   Y2(t)   } |--> S2(t+1)     STEP III
 *
 *     { S1(t+1), S2(t+1),                  } |--> Y1(t+1)     STEP IV
 *
 *     { S1(t+1),          Y1(t+1)          } |--> S1(t+2)     STEP V
 *
 *     { S1(t+2), S2(t+1), Y1(t+2)          } |--> Y2(t+1)     STEP VI
 *
 * -------------------------<<<END SOLVER>>>------------------------------------
 * 
 *     {          S2(t+1), Y1(t+1), Y2(t+1) } |--> S2(t+2)
 *
 *     { S1(t+2), S2(t+2),                  } |--> Y1(t+2)
 *------------------------------------------------------------------------------
 */



int solve_at_point
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 const ArgVec<SMOLYAK>& collocation, ArgVec<TILDE>& solution,
 Approximation** funs_in, int num_state, int num_decxn, int integration_level, int verbose)
{
    // constants
    //int verbose = 0;
    int dim = 3;
    double pi  = 3.141592653589793E+00;
    double sq2 = 1.414213562373095E+00;
    double q_scale = pow(pi, -0.5*dim);

    int o;
    double* q_loc;
    double* w_val;

    // obtain quadrature
    if (integration_level == 100) {
	
	o = en_r2_03_1_size(dim);
	q_loc = new double [dim*o];
	w_val = new double [o];
	en_r2_03_1(dim, o, q_loc, w_val);
	
    } else if (integration_level == 200) {
	
	o = en_r2_05_1_size(dim);
	q_loc = new double [dim*o];
	w_val = new double [o];
	en_r2_05_1(dim, 1, o, q_loc, w_val);
	
    } else {
	
	o = en_r2_09_1_size(dim);
	q_loc = new double [dim*o];
	w_val = new double [o];
	en_r2_09_1(dim, 1, o, q_loc, w_val);
	
    }
    
    // read in collocation point [-1,1] -> [LB,UB]
    // (we want the map-from in regular numbers -- not deviations from SS)
    double ln_vtm1 = UNMAP(collocation[0], LB[0], UB[0]);
    double ln_Rtm1 = UNMAP(collocation[1], LB[1], UB[1]);
    double ln_ktm1 = UNMAP(collocation[2], LB[2], UB[2]);
    double ln_At   = UNMAP(collocation[3], LB[3], UB[3]);
    double ln_sgt  = UNMAP(collocation[4], LB[4], UB[4]);
    double ln_xit  = UNMAP(collocation[5], LB[5], UB[5]);

    double ln_Vt = solution[0] + SS.ln_V_ss;
    double ln_it = solution[1] + SS.ln_i_ss;
    double ln_lt = solution[2] + SS.ln_l_ss;
    double ln_Pt = solution[3] + SS.ln_P_ss;
    double ln_xt = solution[4] + SS.ln_x_ss;
    
    if (verbose == 1) {
	printf("\nPOINT:  -[% 10.7f  % 10.7f  % 10.7f  % 10.7f  % 10.7f  % 10.7f]\n",
	       collocation[0], collocation[1], collocation[2],
	       collocation[3], collocation[4], collocation[5]);
    } else if (verbose > 1) {
	printf("\nPoint:   Smolyak    Natural   Steady-St    Tilde\n");
	printf("        ---------  ---------  ---------  ---------\n");
	printf("v(t-1)  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[0], ln_vtm1, SS.ln_v_ss, ln_vtm1 - SS.ln_v_ss);
	printf("R(t-1)  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[1], ln_Rtm1, SS.ln_R_ss, ln_Rtm1 - SS.ln_R_ss);
	printf("k(t-1)  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[2], ln_ktm1, SS.ln_k_ss, ln_ktm1 - SS.ln_k_ss);	
	printf("A(t)    % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[3], ln_At,   SS.ln_A_ss, ln_At   - SS.ln_A_ss);
	printf("sg(t)   % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[4], ln_sgt,  SS.ln_sg_ss,ln_sgt  - SS.ln_sg_ss);
	printf("xi(t)   % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[5], ln_xit,  0.0,        ln_xit  - 0.0);

	printf("\n");
	printf("V(t)               % 9.6f  % 9.6f  % 9.6f\n",
	       ln_Vt, SS.ln_V_ss, ln_Vt - SS.ln_V_ss);
	printf("i(t)               % 9.6f  % 9.6f  % 9.6f\n",
	       ln_it, SS.ln_i_ss, ln_it - SS.ln_i_ss);
	printf("l(t)               % 9.6f  % 9.6f  % 9.6f\n",
	       ln_lt, SS.ln_l_ss, ln_lt - SS.ln_l_ss);
	printf("P(t)               % 9.6f  % 9.6f  % 9.6f\n",
	       ln_Pt, SS.ln_P_ss, ln_Pt - SS.ln_P_ss);
	printf("x(t)               % 9.6f  % 9.6f  % 9.6f\n",
	       ln_xt, SS.ln_x_ss, ln_xt - SS.ln_x_ss);

    }

    double hold_Pt = ln_Pt;
    
    // mean of log for expectations
    double mu_ln_A  = (1.0-P.rho_A)*P.ln_A_ss   + P.rho_A*ln_At;
    double mu_ln_sg = (1.0-P.rho_sg)*P.ln_sg_ss + P.rho_sg*ln_sgt;

    int tmp_verbose = (ln_Pt > .025) ? 0 : 0;


    
    // set the lambda to be passed into nonlinear solver
    std::function<int(const double*, double*)> FUN = [&](const double* DEC, double* ERR) -> int {

	// expectations/integrals
	double Int1 = 0.0;
	double Int2 = 0.0;
	double Int3 = 0.0;
	double Int4 = 0.0;
	double Int5 = 0.0;

	// vector for collocation
	ArgVec<SMOLYAK> coltp1 (num_state, 0.0);
	
	//----------
	// Y1(t)
	ln_Vt = DEC[0] + SS.ln_V_ss;
	ln_it = DEC[1] + SS.ln_i_ss;
	ln_lt = DEC[2] + SS.ln_l_ss;
	ln_Pt = DEC[3] + SS.ln_P_ss;
	ln_xt = DEC[4] + SS.ln_x_ss;
	
	//----------
	// S1(t+1)
	double ln_kt  = CAP_EQN(P, ln_it,  ln_ktm1);
	double ln_Pst = PST_EQN(P, ln_Pt);
	double ln_vt  = SPR_EQN(P, ln_Pt,  ln_Pst, ln_vtm1);
	coltp1[0] = MAP_SMO(ln_vt, LB[0], UB[0]);
	coltp1[2] = MAP_SMO(ln_kt, LB[2], UB[2]);

	//----------
	// Y2(t)
	double ln_ct  = CON_EQN(P, ln_sgt, ln_At,  ln_kt,  ln_lt, ln_vt, ln_it);
	double ln_wt  = WAG_EQN(P, ln_ct,  ln_lt);
	double ln_mct = MAR_EQN(P, ln_wt,  ln_At,  ln_kt,  ln_lt, ln_vt);
	double ln_Ot  = OME_EQN(P, ln_ct,  ln_lt);
	double ln_yt  = OUT_EQN(P, ln_ct,  ln_it,  ln_sgt);

	//----------
	// S2(t+1)
	double ln_Rt, ln_Atp1, ln_sgtp1, ln_xitp1;
	ln_Rt = TAY_EQN(P, SS, ln_Rtm1,   ln_Pt,  ln_yt, ln_xit);
	ln_Rt = INT_EQN(P,ln_Rt);
	coltp1[1] = MAP_SMO(ln_Rt, LB[1], UB[1]);
	
	//----------
	// Y1(t+1)
	double ln_Vtp1, ln_itp1, ln_ltp1, ln_Ptp1, ln_xtp1;

	//----------
	// S1(t+2)
	double ln_ktp1, ln_Pstp1, ln_vtp1;

	//----------
	// Y2(t+1)
	double ln_ctp1, ln_wtp1, ln_mctp1, ln_rktp1, ln_Otp1;

	//----------
	double ln_v_g_p; // (\gamma-\P.psi)*log(V_{t+1})
	double ln_p_eps; // \eps * log(\Pi_{t})
	
	// iterate over each point in the quadrature
	for (int i = 0; i < o; i++) {
	    
	    //----------
	    // S2(t+1)
	    ln_Atp1  = (P.sigma_A /sq2) * q_loc[i*dim+0] + mu_ln_A;
	    ln_sgtp1 = (P.sigma_sg/sq2) * q_loc[i*dim+1] + mu_ln_sg;
	    ln_xitp1 = (P.sigma_xi/sq2) * q_loc[i*dim+2];
	    coltp1[3] = MAP_SMO(ln_Atp1,  LB[3], UB[3]);
	    coltp1[4] = MAP_SMO(ln_sgtp1, LB[4], UB[4]);
	    coltp1[5] = MAP_SMO(ln_xitp1, LB[5], UB[5]);

	    //----------
	    // Y1(t+1)
	    ln_Vtp1 = (*funs_in[0])(coltp1) + SS.ln_V_ss;
	    ln_itp1 = (*funs_in[1])(coltp1) + SS.ln_i_ss;
	    ln_ltp1 = (*funs_in[2])(coltp1) + SS.ln_l_ss;
	    ln_Ptp1 = (*funs_in[3])(coltp1) + SS.ln_P_ss;
	    ln_xtp1 = (*funs_in[4])(coltp1) + SS.ln_x_ss;

	    //----------
	    // S1(t+2)
	    ln_ktp1  = CAP_EQN(P, ln_itp1,  ln_kt);
	    ln_Pstp1 = PST_EQN(P, ln_Ptp1); 
	    ln_vtp1  = SPR_EQN(P, ln_Ptp1,  ln_Pstp1, ln_vt);

	    //----------
	    // Y2(t+1)
	    ln_ctp1  = CON_EQN(P, ln_sgtp1, ln_Atp1,  ln_ktp1, ln_ltp1, ln_vtp1, ln_itp1);
	    ln_wtp1  = WAG_EQN(P, ln_ctp1,  ln_ltp1);
	    ln_mctp1 = MAR_EQN(P, ln_wtp1,  ln_Atp1,  ln_ktp1, ln_ltp1, ln_vtp1);
	    ln_rktp1 = RAT_EQN(P, ln_mctp1, ln_Atp1,  ln_ktp1, ln_ltp1, ln_vtp1);
	    ln_Otp1  = OME_EQN(P, ln_ctp1,  ln_ltp1);
	    
	    //----------
	    ln_v_g_p = (P.gama-P.psi)*ln_Vtp1;
	    ln_p_eps = P.epsilon*ln_Ptp1;
	    
	    //----------
	    Int1 += w_val[i] * exp((1.0-P.gama)*ln_Vtp1);
	    Int2 += w_val[i] * exp(ln_v_g_p + ln_Otp1 - ln_Ptp1);
	    Int3 += w_val[i] * exp(ln_v_g_p + ln_Otp1 + log(exp(ln_rktp1) + 1.0 - P.delta));
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
	ln_v_g_p = (P.gama-P.psi)*log(Int1)/(1.0-P.gama);
	
	// we would like to find where this is zero
	ERR[0] = (1.0-P.psi)*ln_Vt - log((1.0-P.beta)*exp(ln_Ot+ln_ct) + (P.beta*pow(Int1,(1.0-P.psi)/(1.0-P.gama))));
	ERR[1] = ln_v_g_p + ln_Ot - log(P.beta) - ln_Rt - log(Int2);
	ERR[2] = ln_v_g_p + ln_Ot - log(P.beta) - log(Int3);
	ERR[3] = ln_v_g_p + ln_Ot + log(exp(ln_xt)-exp(ln_mct+ln_yt-ln_vt)) - log(P.theta*P.beta*Int4);
	ERR[4] = ln_v_g_p + ln_Ot + log(exp(ln_xt)-((P.epsilon-1.0)/P.epsilon)*exp(ln_Pst+ln_yt)) - ln_Pst - log(P.theta*P.beta*Int5);

	// check if any of them are NaN-- don't do anything, but alert the user.
	VAL_CHECK_PAR(ERR[0]);
	VAL_CHECK_PAR(ERR[1]);
	VAL_CHECK_PAR(ERR[2]);
	VAL_CHECK_PAR(ERR[3]);
	VAL_CHECK_PAR(ERR[4]);

	
	if (tmp_verbose) {
	
	    printf("\n");
	    printf("------------------\n");
	    printf("V(t)               % 9.6f  % 9.6f  % 9.6f\n",
		   ln_Vt, SS.ln_V_ss, ln_Vt - SS.ln_V_ss);
	    printf("i(t)               % 9.6f  % 9.6f  % 9.6f\n",
		   ln_it, SS.ln_i_ss, ln_it - SS.ln_i_ss);
	    printf("l(t)               % 9.6f  % 9.6f  % 9.6f\n",
		   ln_lt, SS.ln_l_ss, ln_lt - SS.ln_l_ss);
	    printf("P(t)               % 9.6f  % 9.6f  % 9.6f\n",
		   ln_Pt, SS.ln_P_ss, ln_Pt - SS.ln_P_ss);
	    printf("x(t)               % 9.6f  % 9.6f  % 9.6f\n",
		   ln_xt, SS.ln_x_ss, ln_xt - SS.ln_x_ss);

	    
	    printf("\n");
	    printf("Ps(t)              % 9.6f  % 9.6f  % 9.6f\n",
		   ln_Pst, SS.ln_Ps_ss, ln_Pst - SS.ln_Ps_ss);
	    
	    printf("\n");
	    printf("v(t)    % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   coltp1[0], ln_vt, SS.ln_v_ss, ln_vt - SS.ln_v_ss);
	    printf("R(t)    % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   coltp1[1], ln_Rt, SS.ln_R_ss, ln_Rt - SS.ln_R_ss);
	    printf("k(t)    % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   coltp1[2], ln_kt, SS.ln_k_ss, ln_kt - SS.ln_k_ss);
	    
	    printf("\n%.2e %.2e %.2e %.2e %.2e\n", ERR[0], ERR[1], ERR[2], ERR[3], ERR[4]);
	    printf("------------------\n");
	    
	}

	//tmp_verbose = 0;
	
	return 0;

	closure_failure:

	for (int i = 0; i < 5; i++)
	    ERR[i] = 1.0e+10;

	//abort();
	
	return 1;
    };
    
    // we can use this to check the quality of our solution
    double err_arr[5];
    double err_norm = 0.0;
    std::function<double(const double*)> FUN2 = [&](const double* dec_arr) {
	FUN(dec_arr, err_arr);
	err_norm = 0.0;
	for (int i = 0; i < 5; i++)
	    err_norm += err_arr[i]*err_arr[i];
	return err_norm;
    };

    double lb_sa[5] = { -1.5 - SS.ln_V_ss,
			-3.0 - SS.ln_i_ss,
			-1.6 - SS.ln_l_ss,
			 0.00- SS.ln_P_ss,
			 1.0 - SS.ln_x_ss };
    
    double ub_sa[5] = {  0.0 - SS.ln_V_ss,
			-0.7 - SS.ln_i_ss,
			-0.8 - SS.ln_l_ss,
			 0.02- SS.ln_P_ss,
			 1.3 - SS.ln_x_ss };
    
    int sa_err = 0;
    if (verbose == 1) {
	printf("IN:     -[% 10.7f  % 10.7f  % 10.7f  % 10.7f  % 10.7f] --> (%.2e)\n",
	       solution[0], solution[1], solution[2], solution[3], solution[4],
	       FUN2(&solution[0]));
	
    }
    
    // attempt to solve the system of equations
    int sol_err = broydens_method(FUN, &solution[0], num_decxn, 50, 1.0e-14, tmp_verbose);

    int hold_again = solution[3];
    
    if (verbose == 1) {
	printf("OUT:    -[% 10.7f  % 10.7f  % 10.7f  % 10.7f  % 10.7f] --> (%.2e)\n",
	       solution[0], solution[1], solution[2], solution[3], solution[4],
	       FUN2(&solution[0]));
	
    } else if (verbose > 1) {
	tmp_verbose = 1;
	printf("\nValue out: %.2e\n", FUN2(&solution[0]));
    }
    
    delete[] w_val;
    delete[] q_loc;

    return sol_err;
}
