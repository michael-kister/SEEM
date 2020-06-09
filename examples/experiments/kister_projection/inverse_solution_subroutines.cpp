void transformation_setup
(const VecVecSmo& cube_points, const VecVecSmo& para_points,
 std::function<void(ArgVec<SMOLYAK>&)>& cube_to_para,
 std::function<void(ArgVec<SMOLYAK>&)>& para_to_cube)
{
    /*------------------------------------------------------------------------------
     * We must set up the function that will map interest rates based on other
     * values. Essentially this means finding two determinants, and adding in a
     * few ratios.
     *
     * Detailed description:
     * given points in a hypercube, which must be mapped into a hyper-parallele-
     * piped, we desire a matrix M to perform this linear transformation:
     *
     * ┌    ┐   ┌                  ┐   ┌   ┐
     * │ v' │   │ 1                │   │ v │
     * │ R' │   │ a  b  c  d  e  f │   │ R │  
     * │ k' │ = │       1          │ * │ k │
     * │ A' │   │          1       │   │ A │
     * │ s' │   │             1    │   │ s │
     * │ ξ' │   │                1 │   │ ξ │
     * └    ┘   └                  ┘   └   ┘
     *
     * or, more concisely, x' = M * x. Note that for a,...,f, (except for b), we
     * can just use the slopes obtained from the previously obtained points. For
     * element 'b,' recall that if P = M * C, then:
     *
     * det(M) = det(P) / det(C)
     *
     * and since all the other diagonal elements of M are 1, we must have that:
     *
     * b = det(para_points) / det(cube_points)
     *
     * Later, we will concern ourselves with the Jacobian of this transformation,
     * but since this doesn't involve too many calculations, we won't worry about
     * it now.
     *----------------------------------------------------------------------------*/

    int num_state = 6;

    double point_array[num_state*num_state];

    for (int i = 0; i < num_state; i++)
	for (int j = 0; j < num_state; j++)
	    point_array[num_state*i+j] = cube_points[i][j];
    
    double det_c = calculate_determinant(num_state, point_array);

    for (int i = 0; i < num_state; i++)
	for (int j = 0; j < num_state; j++)
	    point_array[num_state*i+j] = para_points[i][j];
    
    double det_p = calculate_determinant(num_state, point_array);

    // i: starting point   j: ending point   k: variable index
    #define SETSLOPE(i,j,k) (para_points[j][1]-para_points[i][1])/(para_points[j][k]-para_points[i][k])
    
    double d_v  = SETSLOPE(0,1,0);

    // interest rates
    double d_R  = det_p / det_c;
    
    // capital
    double d_k  = SETSLOPE(0,2,2);

    // productivity
    double d_A  = SETSLOPE(0,3,3);

    // government
    double d_sg = SETSLOPE(1,4,4);

    // monetary policy
    double d_xi = SETSLOPE(0,5,5);
    
    
    printf("\n\n");
    for (int i = 0; i < num_state; i++) {
	printf("[  ");
	if (i == 1) {
	    printf("% 7.4f  % 7.4f  % 7.4f  % 7.4f  % 7.4f  % 7.4f  ", d_v, d_R, d_k, d_A, d_sg, d_xi);
	} else {
	    for (int j = 0; j < num_state; j++)
		if (i == j)
		    printf(" 1       ");
		else
		    printf("         ");
	}
	printf("]\n");
    }
    printf("\n\n");
    
    
    // set up the lambda that will transform from hypercube to hyper-parallelepiped
    cube_to_para = [=](ArgVec<SMOLYAK>& X) -> void {

	X[1] = d_v*X[0] + d_R*X[1] +  d_k*X[2] +  d_A*X[3] +  d_sg*X[4] +  d_xi*X[5];

    };

    // ... and back again.
    para_to_cube = [=](ArgVec<SMOLYAK>& X) -> void {

	X[1] = -1.0*(d_v*X[0] - X[1] +  d_k*X[2] +  d_A*X[3] +  d_sg*X[4] +  d_xi*X[5])/d_R;

    };
}



/*----------------------------------------------------------------------------------
 * We want to solve for the inverse functions of the singular states; however, we
 * are not guaranteed a solution for the inverse over any domain. Therefore, we
 * must choose a subset of our domain over which we can define our inverses.
 *
 * Once we know our limits, we can map: [-1,1] --> [-1,1]* --> [LB,UB]
 *
 * In words, we will aim to solve for the inverse at all the lovely points in our
 * grid; however, upon receiving each of these points, we will shrink it down so
 * that we end up looking at a sheared version of the grid. We will then stretch
 * out the subset of the grid into our true lower and upper bounds, but don't tell
 * them that there's actually no one in most of the vertices, so that 
 *--------------------------------------------------------------------------------*/

int set_inverse_domain_cheby
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 Approximation** funs, int num_state, int num_snglr, int smolyak_q,
 Approximation& bound_fun, VecVecSmo& cube_points, VecVecSmo& para_points)
{
    Graph bound_graph;
    
    Grid G  = smolyak_grid((num_state-1), 1 - (num_state-1));
    int num_collocation = G.size();

    int error;

    #pragma omp parallel num_threads(1)
    {
	int error;
	Point it_par = *G.begin();

	ArgVec<SMOLYAK> loc_vec_par_1 (num_state,   0.0);
	ArgVec<SMOLYAK> loc_vec_par_2 (num_state,   0.0);

	ArgVec<TILDE> sol_vec_par_1 (num_snglr,   0.0);
	ArgVec<TILDE> sol_vec_par_2 (num_snglr,   0.0);
	
        #pragma omp for
	for (int j = 0; j < num_collocation; j++) {

	    j = 5;
	    
	    it_par = G.element(j);
	    
	    loc_vec_par_2 = static_cast<ArgVec<SMOLYAK>>(it_par);
	    
	    // add in interest rates
	    loc_vec_par_2.insert(++loc_vec_par_2.begin(), -1.0);

	    //printf("[% 6.3f % 6.3f % 6.3f % 6.3f % 6.3f]\n",
		   //loc_vec_par_2[0], loc_vec_par_2[1], loc_vec_par_2[2],
		   //loc_vec_par_2[3], loc_vec_par_2[4]);

	    //printf("GOAL: -[% 6.3f % 6.3f % 6.3f % 6.3f % 6.3f % 6.3f]\n",
		   //loc_vec_par_2[0], loc_vec_par_2[1], loc_vec_par_2[2],
		   //loc_vec_par_2[3], loc_vec_par_2[4], loc_vec_par_2[5]);

	    loc_vec_par_1 = loc_vec_par_2;
	    loc_vec_par_1[1] = 1.0;
	    loc_vec_par_1[1] = 0.0;
	    
	    error = solve_inverse_at_point(P, SS, LB, UB, loc_vec_par_1, sol_vec_par_1, funs,
					   num_state, num_snglr, 0, 2);

	    error = perform_transplant(P, SS, LB, UB, funs, loc_vec_par_1, loc_vec_par_2, sol_vec_par_1,
				       sol_vec_par_2, num_state, num_snglr, 1, &solve_inverse_at_point, 2);

	    //printf("HAVE: -[% 6.3f % 6.3f % 6.3f % 6.3f % 6.3f % 6.3f]\n\n",
		   //loc_vec_par_2[0], loc_vec_par_2[1], loc_vec_par_2[2],
		   //loc_vec_par_2[3], loc_vec_par_2[4], loc_vec_par_2[5]);

	    #pragma omp critical
	    {
		bound_graph.insert(std::pair<Point,double>(it_par, loc_vec_par_2[1]));
	    }
	    
	} // for loop
	
    } // parallel section

    bound_fun.set_coefficients(bound_graph);

    ArgVec<SMOLYAK> zeros (num_state-1, 0.0); // where we evaluate the slopes
    ArgVec<TILDE> slope (num_state-1, 0.0); // where we store the slopes

    // obtain slopes
    for (int i = 0; i < (num_state-1); i++)

	slope[i] = (bound_fun % i)(zeros);
    
    // calculate the sum of the slopes
    double slope_sum = fabs(bound_fun.get_constant());

    for (int i = 0; i < (num_state-1); i++) {

	slope_sum += fabs(slope[i]);

	zeros[i] = (2.0*(slope[i] < 0.0))-1.0;
    }

    // later we'll want to check that this is at least 1.0, since that means
    // we can stretch all the way to the bottom.
    /*
    printf("Constant: %f\n", slope_sum);

    printf("\n% 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	   slope[0], slope[1], slope[2], slope[3], slope[4]);

    printf("\n% 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n\n\n",
	   zeros[0], zeros[1], zeros[2], zeros[3], zeros[4]);
    */
    slope.insert(++slope.begin(), 0.0); // meaningless

    zeros.insert(++zeros.begin(), -1.0); // since we want to look at points with low R
    
    // now we need to actually set the points we care about.
    int ivv = 0;
    
    cube_points[ivv] = zeros;
    
    para_points[ivv++] = zeros;
    
    for (int i = 0; i < num_state; i++) {

	if (i != 1) {

	    zeros[i] *= -1.0;
	    
	    cube_points[ivv] = zeros;

	    zeros[1] += fabs(2.0*slope[i]);

	    para_points[ivv++] = zeros;

	    zeros[1] = -1.0;

	    zeros[i] *= -1.0;
	}	    
    }

    // something to show what we took in/got out:
    /*
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
    
/*
    for (int i = 0; i < (num_state-1); i++) {

	if (i == 0) {

	    // then it's spread

	}


    }
*/

    return 0;
}


int solve_inverse
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 Approximation** funs, Approximation** invs, Graph** graphs,
 std::function<void(ArgVec<SMOLYAK>&)>& cube_to_para,
 int num_state, int num_snglr, int smolyak_q)
{
    // containers for examining projection (double for comparing them)
    double** coeffs = new double*[num_snglr];

    // the grid over which we'll loop
    Grid G = smolyak_grid(num_state, smolyak_q - num_state);
    int num_collocation = G.size();
    
    // allocate space for the coefficients (and order arrays)
    for (int i = 0; i < num_snglr; i++)
	coeffs[i] = new double[num_collocation];
	
    // clear the graphs to start fresh
    for (int i = 0; i < num_snglr; i++)
	graphs[i]->clear();

    // make a graph to hold whether or not you succeeded
    std::map<Point, int> error_map;
    int num_error = 0;
    
    auto start = high_resolution_clock::now();
    printf("Inverse: ");

    Point tt = *G.begin(); 
    
    #pragma omp parallel num_threads(1)
    {
	int error;
	Point it_par = *G.begin();
	ArgVec<SMOLYAK> state_vec_par (num_state, 0.0);
	ArgVec<TILDE> decxn_vec_par (num_snglr, 0.0);
	
        #pragma omp for
	for (int j = 0; j < num_collocation; j++) {

	    // obtain the j'th element
	    it_par = G.element(j);

	    if (j == 326)
		tt = it_par;

	    // cast it into a vector
	    state_vec_par = static_cast<ArgVec<SMOLYAK>>(it_par);

	    // shift interest rates to dodge bad regions
	    cube_to_para(state_vec_par);

	    // solve for the inverse at this point

	    //if (j == 326)
	    //error = solve_inverse_at_point(P, SS, LB, UB, state_vec_par, decxn_vec_par, funs,
					   //num_state, num_snglr, 0, 1);
	    //else
	    error = solve_inverse_at_point(P, SS, LB, UB, state_vec_par, decxn_vec_par, funs,
					   num_state, num_snglr, 0, 0);

            #pragma omp critical
	    {
		error_map.insert(std::pair<Point,int>(it_par, error));
		if (!error) {
		    for (int k = 0; k < num_snglr; k++)
			graphs[k]->insert(std::pair<Point,double>(it_par, decxn_vec_par[k]));
 		} else {
		    ++num_error;
		}
	    }
	}
    }

    // some setup in case we want to go back and sweep through
    
    // status of whether we're ready to improve
    int available = 0;
    int transplant_error;
    
    // our two patients
    Point donor     = *G.begin();
    Point recipient = *G.begin();

    // and their respective solutions
    ArgVec<TILDE> donated(3, 0.0);
    ArgVec<TILDE> received(3, 0.0);

    // here is where we're going to go back and comb through our error graph to find
    // locations of failure.
    int prev_num_error = 0;

    ArgVec<SMOLYAK> recipient_vec (num_state, 0.0);
    ArgVec<SMOLYAK> donor_vec     (num_state, 0.0);

    if (num_error)

	printf("(%d errors)", num_error);
    
    while (num_error != prev_num_error) {

	prev_num_error = num_error;
	
	if (num_error) {
	    
	    for (std::map<Point,int>::iterator it = error_map.begin(); it != error_map.end(); it++) {
		if (it->second) {
		    if (available) {
			recipient = it->first;
			
			recipient_vec = static_cast<ArgVec<SMOLYAK>>(recipient);
			donor_vec     = static_cast<ArgVec<SMOLYAK>>(donor);

			cube_to_para(recipient_vec);
			cube_to_para(donor_vec);
			
			transplant_error = perform_transplant
			    (P, SS, LB, UB, funs, donor_vec, recipient_vec, donated, received,
			     num_state, num_snglr, 1, &solve_inverse_at_point, 0);
			
			if (!transplant_error) {
			    it->second = 0;
			    for (int i = 0; i < 3; i++)
				graphs[i]->insert(std::pair<Point,double>(it->first, received[i]));
			    donor = recipient;
			    --num_error;
			}
		    }
		} else {
		    available = 1;
		    donor = it->first;
		    for (int i = 0; i < 3; i++)
			donated[i] = (*graphs[i])[it->first];
		}
	    }

	    printf("-->(%d errors)", num_error);
	}

	if (num_error) {
	    
	    for (std::map<Point,int>::reverse_iterator it = error_map.rbegin(); it != error_map.rend(); it++) {
		if (it->second) {
		    if (available) {
			recipient = it->first;
			
			recipient_vec = static_cast<ArgVec<SMOLYAK>>(recipient);
			donor_vec     = static_cast<ArgVec<SMOLYAK>>(donor);

			cube_to_para(recipient_vec);
			cube_to_para(donor_vec);
			
			transplant_error = perform_transplant
			    (P, SS, LB, UB, funs, donor_vec, recipient_vec, donated, received,
			     num_state, num_snglr, 1, &solve_inverse_at_point, 0);
			if (!transplant_error) {
			    it->second = 0;
			    for (int i = 0; i < 3; i++)
				graphs[i]->insert(std::pair<Point,double>(it->first, received[i]));
			    donor = recipient;
			    --num_error;
			}
		    }
		} else {
		    available = 1;
		    donor = it->first;
		    for (int i = 0; i < 3; i++)
			donated[i] = (*graphs[i])[it->first];
		}
	    }
	
	    printf("--<(%d errors)", num_error);
	}
    }
    
    if (!num_error)
	for (int k = 0; k < num_snglr; k++)
	    invs[k]->set_coefficients(graphs[k]);


    tt.print();

    double ttt = (*invs[1])(tt);

    printf("%f\n", ttt);

    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    
    printf(" %5d ms ---> ", (int)duration.count());
    if (num_error) {
	printf("(FAILED)\n\n");
    } else {
	printf("(SUCCEEDED)\n\n");
    }

    for (int i = 0; i < num_snglr; i++)
	delete[] coeffs[i];

    return num_error;    
}


int solve_inverse_at_point
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 const ArgVec<SMOLYAK>& collocation, ArgVec<TILDE>& solution,
 Approximation** funs, int num_state, int num_decxn, int use_input, int verbose)
{
    /*----------------------------------------------------------------------------
     * This is a function that we need for the EIS filter; we need to obtain:
     *
     *           ψ : (vₜ,Rₜ,kₜ,Aₜ₋₁,sₜ₋₁,ξₜ₋₁) ↦ (vₜ₋₁,Rₜ₋₁,kₜ₋₁)
     *
     * Our typical method of solving for such an approximation is to take points
     * in the domain of the function, and solve for the mappings.
     *
     * As usual, our output will be in deviations from the steady state, and so
     * that is how we might expect to receive an initial guess for the solution.
     * However, when we solve the nonlinear equations, we will work with the
     * natural logarithm of (logged) R and v, since that will allow the solver
     * to go as far negative as it wants, but we still actually have positive
     * values for ln(R) and ln(v). Therefore, if we want to use a previously ob-
     * tained solution, then we add back the steady state, and take logarithms.
     *
     * Basically what we're doing is trying to find the values for lagged sing-
     * ular states that, when combined with a given set of lagged stochastic
     * states, yield the given contemporaneous singular states.
     *--------------------------------------------------------------------------*/
    
    // constants
    //int num_state = 6;
    //int num_decxn = 3;
    
    // contemporaneous endogenous (not in deviations)
    double ln_vt_ = UNMAP(collocation[0], LB[0], UB[0]);
    double ln_Rt_ = UNMAP(collocation[1], LB[1], UB[1]);
    double ln_kt_ = UNMAP(collocation[2], LB[2], UB[2]);
    
    // "lagged" stochastic (not in deviations)
    double ln_At  = UNMAP(collocation[3], LB[3], UB[3]);
    double ln_sgt = UNMAP(collocation[4], LB[4], UB[4]);
    double ln_xit = UNMAP(collocation[5], LB[5], UB[5]);

    if (verbose == 1) {
	printf("\nPOINT:  -[% 8.5f  % 8.5f  % 8.5f  % 8.5f  % 8.5f  % 8.5f]\n",
	       collocation[0], collocation[1], collocation[2],
	       collocation[3], collocation[4], collocation[5]);
    } else if (verbose > 1) {
	printf("\nPoint:   Smolyak    Natural   Steady-St    Tilde\n");
	printf("        ---------  ---------  ---------  ---------\n");
	printf("v(t)    % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[0], ln_vt_, SS.ln_v_ss, ln_vt_ - SS.ln_v_ss);
	printf("R(t)    % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[1], ln_Rt_, SS.ln_R_ss, ln_Rt_ - SS.ln_R_ss);
	printf("k(t)    % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
	       collocation[2], ln_kt_, SS.ln_k_ss, ln_kt_ - SS.ln_k_ss);
    }
    
    if (use_input) {

	// if we're receiving a solution, we'll expect it in deviations from the steady state,
	// which needs to be slightly adjusted for the way that we work with things
	solution[0] = log(solution[0] + SS.ln_v_ss);
	solution[1] = log(solution[1] + SS.ln_R_ss);
	solution[2] = solution[2];
	
    } else {

	// otherwise, we must set things up ourselves
	solution[0] = log(ln_vt_);
	solution[1] = log(ln_Rt_);
	solution[2] = ln_kt_ - SS.ln_k_ss;
	
    }
    
    // vector for collocation
    ArgVec<SMOLYAK> collo (num_state, 0.0);
    collo[3] = MAP_SMO(ln_At,  LB[3], UB[3]);
    collo[4] = MAP_SMO(ln_sgt, LB[4], UB[4]);
    collo[5] = MAP_SMO(ln_xit, LB[5], UB[5]);
    
    int tmp_verbose = 0;

    // set the lambda to be passed into nonlinear solver
    std::function<int(const double*, double*)> FUN = [&](const double* DEC, double* ERR) -> int {

	//double ln_vtm1 = DEC[0] + SS.ln_v_ss;
	double ln_vtm1 = exp(DEC[0]);
	double ln_Rtm1 = exp(DEC[1]);
	double ln_ktm1 = DEC[2] + SS.ln_k_ss;

	collo[0] = MAP_SMO(ln_vtm1, LB[0], UB[0]);
	collo[1] = MAP_SMO(ln_Rtm1, LB[1], UB[1]);
	collo[2] = MAP_SMO(ln_ktm1, LB[2], UB[2]);
	
	double ln_it = (*funs[1])(collo) + SS.ln_i_ss;
	double ln_lt = (*funs[2])(collo) + SS.ln_l_ss;
	double ln_Pt = (*funs[3])(collo) + SS.ln_P_ss;

	double ln_kt  = log(exp(ln_it) + (1.0-P.delta)*exp(ln_ktm1));
	double ln_Pst = (log(1.0-P.theta*exp((P.epsilon-1.0)*ln_Pt))-log(1.0-P.theta))/(1.0-P.epsilon);
	double ln_vt  = log((1.0-P.theta)*exp(-1.0*P.epsilon*ln_Pst) + P.theta*exp(P.epsilon*ln_Pt + ln_vtm1));
	double ln_ct  = log((1.0-exp(ln_sgt))*exp(ln_At + P.zeta*ln_kt + (1.0-P.zeta)*ln_lt - ln_vt) - exp(ln_it));
	double ln_yt  = log(exp(ln_ct)+exp(ln_it)) - log(1.0-exp(ln_sgt));
	double ln_Zt  = P.rho_R*ln_Rtm1 + (1.0-P.rho_R)*(P.ln_P_ss-log(P.beta) + P.phi_pi*(ln_Pt-P.ln_P_ss) + P.phi_y*(ln_yt-SS.ln_y_ss)) + ln_xit;
	double ln_Rt  = (ln_Zt > 0.0) ? ln_Zt : 0.0;
	
	ERR[0] = ln_vt - ln_vt_;
	ERR[1] = ln_Rt - ln_Rt_;
	ERR[2] = ln_kt - ln_kt_;

	

	if (1) {

	    printf("--------------------------------\n% 9.6f  % 9.6f  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n\n",
		   collo[0], collo[1], collo[2], collo[3], collo[4], collo[5]);
	    
	    printf("\n");
	    printf("v(t-1)  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   MAP_SMO(ln_vtm1, LB[0], UB[0]),
		   ln_vtm1, SS.ln_v_ss, ln_vtm1 - SS.ln_v_ss);
	    printf("R(t-1)  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   MAP_SMO(ln_Rtm1, LB[1], UB[1]),
		   ln_Rtm1, SS.ln_R_ss, ln_Rtm1 - SS.ln_R_ss);
	    printf("k(t-1)  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   MAP_SMO(ln_ktm1, LB[2], UB[2]),
		   ln_ktm1, SS.ln_k_ss, ln_ktm1 - SS.ln_k_ss);

	    printf("\n");
	    printf("v(t)*   % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   MAP_SMO(ln_vt, LB[0], UB[0]),
		   ln_vt , SS.ln_v_ss, ln_vt  - SS.ln_v_ss);
	    printf("R(t)*   % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   MAP_SMO(ln_Rt, LB[1], UB[1]),
		   ln_Rt , SS.ln_R_ss, ln_Rt  - SS.ln_R_ss);
	    printf("k(t)*   % 9.6f  % 9.6f  % 9.6f  % 9.6f\n",
		   MAP_SMO(ln_kt, LB[2], UB[2]),
		   ln_kt , SS.ln_k_ss, ln_kt  - SS.ln_k_ss);

	printf("\nError: % 12.9f  % 12.9f  % 12.9f\n\n", ERR[0], ERR[1], ERR[2]);
	}
	
	VAL_CHECK_PAR2(ERR[0]);
	VAL_CHECK_PAR2(ERR[1]);
	VAL_CHECK_PAR2(ERR[2]);
	
	return 0;

	inverse_closure_failure:

	for (int i = 0; i < num_decxn; i++)
	    ERR[i] = 100.0;

	return 1;
    };

    // we can use this to check the quality of our solution
    std::function<double(const double *)> FUN2 = [&](const double* dec_arr) {
	double err_arr[3];
	FUN(dec_arr, err_arr);
	double err_norm = 0.0;
	for (int i = 0; i < 3; i++)
	    err_norm += err_arr[i]*err_arr[i];
	return err_norm;
    };
    
    if (verbose == 1) {
	printf("IN:     -[% 8.5f  % 8.5f  % 8.5f]\n",
	       exp(solution[0])-SS.ln_v_ss,
	       exp(solution[1])-SS.ln_R_ss,
	       solution[2]);
    }

    int b_fail;
    
    if (!isnormal(exp(solution[0])-SS.ln_v_ss) ||
	!isnormal(exp(solution[1])-SS.ln_R_ss) ||
	!isnormal(solution[2])) {

	b_fail = 1;
	
    } else {

	b_fail = broydens_method(FUN, &solution[0], 3, 50, 1.0e-14, verbose);

    }

    if (verbose == 1) {
	printf("OUT:    -[% 8.5f  % 8.5f  % 8.5f] -----------------------------> (%.2e)\n",
	       exp(solution[0])-SS.ln_v_ss,
	       exp(solution[1])-SS.ln_R_ss,
	       solution[2],
	       FUN2(&solution[0]));
    } else if (verbose > 1) {
	tmp_verbose = 1;
	printf("\nValue out: %.2e\n", FUN2(&solution[0]));
    }
    
    // we want to address interest rates
    solution[0] = exp(solution[0]) - SS.ln_v_ss;
    solution[1] = exp(solution[1]) - SS.ln_R_ss;

    return b_fail;
}


/*
	#pragma omp master
	{
	    for (int k = 0; k < num_snglr; k++) {
		invs[k]->set_coefficients(graphs[k]);
		graphs[k]->clear();
		invs[k]->print_receipt(coeffs[k]);
	    }

	    if (0) {
		
		printf("bounds[[1]] <- list(");
		for (int j = 0; j < num_state; j++)
		    printf("c(%13.10f,%13.10f), ", LB[j], UB[j]);
		printf("\b\b)\n");
		    
		printf("coeffs[[%d]] <- c(", 1);
		for (int j = 0; j < num_collocation; j++)
		    printf("%+11.4e, ", coeffs[0][j]);
		printf("\b\b)\n");
		    
		printf("coeffs[[%d]] <- c(", 2);
		for (int j = 0; j < num_collocation; j++)
		    printf("%+11.4e, ", coeffs[1][j]);
		printf("\b\b)\n");
		    
		printf("coeffs[[%d]] <- c(", 3);
		for (int j = 0; j < num_collocation; j++)
		    printf("%+11.4e, ", coeffs[2][j]);
		printf("\b\b)\n");
		    
	    }
	} // only master checks for convergence
 */
/*
    // something to write the output
    ArgVec<SMOLYAK> X(6, 0.0);
    double Y;
    FILE* F = fopen("jul11_test.txt", "w");
    // loop over decision rules
    for (int i = 0; i < 5; i++) {
	// loop over states
	for (int j = 0; j < 6; j++) {
	    X = ArgVec<SMOLYAK>(6, 0.0);
	    for (double x = -1.1; x < 1.14; x += .04) {
		X[j] = x;
		Y = (*funs_all[i_succ][i])(X);
		fprintf(F, "%+12.9f ", Y);
	    }
	    fprintf(F, "\n");
	    fflush(F);
	}
    }
    fclose(F);
*/
