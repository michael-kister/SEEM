int set_extra_approximations
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 Approximation** exog, Approximation** endog, Approximation** measr, Graph** graphs, Grid& G,
 int num_state, int num_snglr, int num_measr)
{
    int num_collocation = G.size();

    // clear the graphs to start fresh
    for (int i = 0; i < (num_snglr+num_measr); i++)
	graphs[i]->clear();

    double** coeffs = new double*[num_snglr+num_measr];
    for (int i = 0; i < (num_snglr+num_measr); i++)
	coeffs[i] = new double[num_collocation];

    #pragma omp parallel //num_threads(1)
    {
	Point it_par = *G.begin();
	ArgVec<SMOLYAK> state_vec_par (num_state, 0.0);
	ArgVec<TILDE> snglr_vec_par (num_snglr, 0.0);
	ArgVec<TILDE> measr_vec_par (num_measr, 0.0);
	
        #pragma omp for
	for (int j = 0; j < num_collocation; j++) {
	    
	    it_par = G.element(j);
	    state_vec_par = static_cast<ArgVec<SMOLYAK>>(it_par);

	    calculate_values(P, SS, LB, UB, state_vec_par, measr_vec_par, snglr_vec_par, exog);
	    
            #pragma omp critical
	    {
	        for (int k = 0; k < num_snglr; k++)
		    graphs[k]->insert(std::pair<Point,double>(it_par, snglr_vec_par[k]));

		for (int k = num_snglr; k < num_snglr+num_measr; k++)
		    graphs[k]->insert(std::pair<Point,double>(it_par, measr_vec_par[k-num_snglr]));
	    }
	}	
	#pragma omp master
	{
	    for (int k = 0; k < num_snglr; k++)
		endog[k]->set_coefficients(graphs[k]);
	    
	    for (int k = 0; k < num_measr; k++)
		measr[k]->set_coefficients(graphs[num_snglr+k]);
	    
	    endog[0]->print_receipt(coeffs[0]);
	    endog[1]->print_receipt(coeffs[1]);
	    endog[2]->print_receipt(coeffs[2]);
	    measr[0]->print_receipt(coeffs[3]);
	    measr[1]->print_receipt(coeffs[4]);
	    
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
		    
		printf("coeffs[[%d]] <- c(", 4);
		for (int j = 0; j < num_collocation; j++)
		    printf("%+11.4e, ", coeffs[3][j]);
		printf("\b\b)\n");
		    
		printf("coeffs[[%d]] <- c(", 5);
		for (int j = 0; j < num_collocation; j++)
		    printf("%+11.4e, ", coeffs[4][j]);
		printf("\b\b)\n");
		    
	    }
	    
	}
        #pragma omp barrier
    }

    return 0;
}


void calculate_values
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 ArgVec<SMOLYAK>& St, ArgVec<TILDE>& Yt, ArgVec<TILDE>& Stp1, Approximation** exog)
{
    double ln_vtm1 = UNMAP(St[0], LB[0], UB[0]);
    double ln_Rtm1 = UNMAP(St[1], LB[1], UB[1]);
    double ln_ktm1 = UNMAP(St[2], LB[2], UB[2]);
    double ln_At   = UNMAP(St[3], LB[3], UB[3]);
    double ln_sgt  = UNMAP(St[4], LB[4], UB[4]);
    double ln_xit  = UNMAP(St[5], LB[5], UB[5]);

    double ln_it = (*exog[1])(St) + SS.ln_i_ss;
    double ln_lt = (*exog[2])(St) + SS.ln_l_ss;
    double ln_Pt = (*exog[3])(St) + SS.ln_P_ss;

    /*
    double ln_kt  = log(exp(ln_it) + (1.0-P.delta)*exp(ln_ktm1));
    double ln_Pst = (log(1.0-P.theta*exp((P.epsilon-1.0)*ln_Pt))-log(1.0-P.theta))/(1.0-P.epsilon);
    double ln_vt  = log((1.0-P.theta)*exp(-1.0*P.epsilon*ln_Pst) + P.theta*exp(P.epsilon*ln_Pt + ln_vtm1));
    double ln_ct  = log((1.0-exp(ln_sgt))*exp(ln_At + P.zeta*ln_kt + (1.0-P.zeta)*ln_lt - ln_vt) - exp(ln_it));
    double ln_wt  = log((1.0-P.alpha)/P.alpha) + ln_ct - log(1.0-exp(ln_lt));
    double ln_mct = ln_wt - log(1.0-P.zeta) - ln_At - P.zeta*(ln_kt-ln_lt) + ln_vt;
    double ln_Ot  = (1.0-P.psi)*(P.alpha*ln_ct + (1.0-P.alpha)*log(1.0-exp(ln_lt))) - ln_ct;
    double ln_yt  = log(exp(ln_ct)+exp(ln_it)) - log(1.0-exp(ln_sgt));
    double ln_Zt  = P.rho_R*ln_Rtm1 + (1.0-P.rho_R)*(P.ln_P_ss-log(P.beta) + P.phi_pi*(ln_Pt-P.ln_P_ss) + P.phi_y*(ln_yt-SS.ln_y_ss)) + ln_xit;
    double ln_Rt  = (ln_Zt > 0.0) ? ln_Zt : 0.0;
    */
    /*
    double ln_kt  = CAP_EQN(P, ln_it,  ln_ktm1);
    double ln_Pst = PST_EQN(P, ln_Pt);
    double ln_vt  = SPR_EQN(P, ln_Pt,  ln_Pst, ln_vtm1);
    double ln_ct  = CON_EQN(P, ln_sgt, ln_At,  ln_kt,  ln_lt, ln_vt, ln_it);
    double ln_wt  = WAG_EQN(P, ln_ct,  ln_lt);
    double ln_mct = MAR_EQN(P, ln_wt,  ln_At,  ln_kt,  ln_lt, ln_vt);
    double ln_Ot  = OME_EQN(P, ln_ct,  ln_lt);
    double ln_yt  = OUT_EQN(P, ln_ct,  ln_it,  ln_sgt);
    
    double ln_Zt  = TAY_EQN(P, SS,
			    ln_Rtm1,    ln_Pt,      ln_yt,
			    ln_xit);

    double ln_Rt  = INT_EQN(P,ln_Zt);

    */
    double ln_vt = SPR_EQN_LONG(P,SS,ln_vtm1,ln_Rtm1,ln_ktm1,ln_At,ln_sgt,ln_xit,
				0.0,ln_it,ln_lt,ln_Pt,0.0);

    double ln_Rt = INT_EQN_LONG(P,SS,ln_vtm1,ln_Rtm1,ln_ktm1,ln_At,ln_sgt,ln_xit,
				0.0,ln_it,ln_lt,ln_Pt,0.0);

    double ln_kt = CAP_EQN_LONG(P,SS,ln_vtm1,ln_Rtm1,ln_ktm1,ln_At,ln_sgt,ln_xit,
				0.0,ln_it,ln_lt,ln_Pt,0.0);

    double ln_yt = OUT_EQN_LONG(P,SS,ln_vtm1,ln_Rtm1,ln_ktm1,ln_At,ln_sgt,ln_xit,
				0.0,ln_it,ln_lt,ln_Pt,0.0);

    double ln_ct = CON_EQN_LONG(P,SS,ln_vtm1,ln_Rtm1,ln_ktm1,ln_At,ln_sgt,ln_xit,
				0.0,ln_it,ln_lt,ln_Pt,0.0);


    
    Yt[0] = ln_yt - SS.ln_y_ss;
    Yt[1] = ln_ct - SS.ln_c_ss;
    
    Stp1[0] = ln_vt - SS.ln_v_ss;
    Stp1[1] = ln_Rt - SS.ln_R_ss;
    Stp1[2] = ln_kt - SS.ln_k_ss;

}


