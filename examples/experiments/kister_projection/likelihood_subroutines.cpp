
#include "likelihood_subroutines.h"

void likelihood_setup
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 std::vector<AppVec>& J, std::vector<AppVec>& Z, std::vector<AppVec>& A,
 std::function<double(const double*)>& Jacobian,
 DblDbl& measr_eqn, DblDbl& trans_eqn, DblDbl& invrs_eqn, DblDbl& print_Zmat, DblDbl& print_Amat,
 Approximation** decxn, Approximation** measr, Approximation** endog, Approximation** invrs,
 std::function<void(ArgVec<SMOLYAK>&)>& para_to_cube,
 std::function<void(ArgVec<SMOLYAK>&)>& cube_to_para)
{
    int num_state = 6;
    int num_decxn = 5;
    int num_snglr = 3;
    
    double lb0 = LB[0], ub0 = UB[0];
    double lb1 = LB[1], ub1 = UB[1];
    double lb2 = LB[2], ub2 = UB[2];
    double lb3 = LB[3], ub3 = UB[3];
    double lb4 = LB[4], ub4 = UB[4];
    double lb5 = LB[5], ub5 = UB[5];

    double ss0 = SS.ln_v_ss;
    double ss1 = SS.ln_R_ss;
    double ss2 = SS.ln_k_ss;
    double ss3 = SS.ln_A_ss;
    double ss4 = SS.ln_sg_ss;
    double ss5 = 0.0;
    
    double d0 = 2.0/(UB[0]-LB[0]);
    double d1 = 2.0/(UB[1]-LB[1]);
    double d2 = 2.0/(UB[2]-LB[2]);
    double d3 = 2.0/(UB[3]-LB[3]);
    double d4 = 2.0/(UB[4]-LB[4]);
    double d5 = 2.0/(UB[5]-LB[5]);

    double rho_A  = P.rho_A;
    double rho_sg = P.rho_sg;

    double yss0 = SS.ln_y_ss;
    double yss1 = SS.ln_c_ss;
    double yss2 = SS.ln_l_ss;
    double yss3 = SS.ln_P_ss;
    
    #define MAPSTATE(i) S[i] = MAP_SMO(S_arr[i]+ss##i,lb##i,ub##i)
    
    #define MAPALL MAPSTATE(0), MAPSTATE(1), MAPSTATE(2), MAPSTATE(3), MAPSTATE(4), MAPSTATE(5)

    trans_eqn = [=](const double* S_arr, double* Stp1) -> void {

	printf("\nX-SS:\n");
	printf("> v(t-1) = %9.6f\n", S_arr[0]);
	printf("> R(t-1) = %9.6f\n", S_arr[1]);
	printf("> k(t-1) = %9.6f\n", S_arr[2]);
	
	ArgVec<SMOLYAK> S(6, 0.0);
	MAPALL;


	printf("\nmap_smo((X-SS)+SS):\n");
	printf("> v'(t-1) = %9.6f\n", S[0]);
	printf("> R'(t-1) = %9.6f\n", S[1]);
	printf("> k'(t-1) = %9.6f\n\n", S[2]);

	// there's not really a reason to use approximations here...

	double ln_it = (*decxn[1])(S); // investment
	double ln_lt = (*decxn[2])(S); // labor
	double ln_Pt = (*decxn[3])(S); // inflation
	
	// transition equation
	Stp1[0] = (*endog[0])(S);
	Stp1[1] = (*endog[1])(S);
	Stp1[2] = (*endog[2])(S);
	
	double ln_vt = SPR_EQN_LONG(P,SS,
				    S_arr[0]+ss0,S_arr[1]+ss1,S_arr[2]+ss2,
				    S_arr[3]+ss3,S_arr[4]+ss4,S_arr[5]+ss5,
				    0.0,ln_it+SS.ln_i_ss,ln_lt+SS.ln_l_ss,
				    ln_Pt+SS.ln_P_ss,0.0);
	
	double ln_Rt = INT_EQN_LONG(P,SS,
				    S_arr[0]+ss0,S_arr[1]+ss1,S_arr[2]+ss2,
				    S_arr[3]+ss3,S_arr[4]+ss4,S_arr[5]+ss5,
				    0.0,ln_it+SS.ln_i_ss,ln_lt+SS.ln_l_ss,
				    ln_Pt+SS.ln_P_ss,0.0);
	
	double ln_kt = CAP_EQN_LONG(P,SS,
				    S_arr[0]+ss0,S_arr[1]+ss1,S_arr[2]+ss2,
				    S_arr[3]+ss3,S_arr[4]+ss4,S_arr[5]+ss5,
				    0.0,ln_it+SS.ln_i_ss,ln_lt+SS.ln_l_ss,
				    ln_Pt+SS.ln_P_ss,0.0);
	
	Stp1[3] = rho_A*S_arr[3];
	Stp1[4] = rho_sg*S_arr[4];
	Stp1[5] = 0.0;

	printf("\nCalculation:   %f %f %f",   Stp1[0], Stp1[1], Stp1[2]);
	printf("\nCalculation:   %f %f %f", S_arr[0]+ss0,S_arr[1]+ss1,S_arr[2]+ss2);
	printf("\nCalculation:   %f %f %f", ln_vt, ln_Rt, ln_kt);
	printf("\nCalculation:   %f %f %f\n",
	       ln_vt-SS.ln_v_ss,
	       ln_Rt-SS.ln_R_ss,
	       ln_kt-SS.ln_k_ss);
	
	// lagged stochastic values
	Stp1[6] = S_arr[3];
	Stp1[7] = S_arr[4];
	Stp1[8] = S_arr[5];
    };

    measr_eqn = [=](const double* S_arr, double* Yt) -> void {
	
	ArgVec<SMOLYAK> S(6, 0.0);
	MAPALL;
	
	Yt[0] = (*measr[0])(S); // output
	Yt[1] = (*measr[1])(S); // consumption
	Yt[2] = (*decxn[2])(S); // labor
	Yt[3] = (*decxn[3])(S); // inflation
	Yt[4] = S_arr[1];       // interest rate
    };

    invrs_eqn = [=,&para_to_cube](const double* S_arr, double* Stm1) -> void {

	printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n");
	printf("ENTERING INVERSE\n\n");
	
	Grid G = smolyak_grid(6, 3-6);

	// CUBIC
	Point pp = G.element(326);
	pp.print();
	
	// CUBIC
	ArgVec<SMOLYAK> S_smo = static_cast<ArgVec<SMOLYAK>>(pp);
	printf("\n(1) Cube:   % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       S_smo[0], S_smo[1], S_smo[2], S_smo[3], S_smo[4], S_smo[5]);

	ArgVec<SMOLYAK> S_nat(6, 0.0);

	S_nat[0] = UNMAP(S_smo[0], lb0, ub0);
	S_nat[1] = UNMAP(S_smo[1], lb1, ub1);
	S_nat[2] = UNMAP(S_smo[2], lb2, ub2);
	
	S_nat[3] = UNMAP(S_smo[3], lb3, ub3);
	S_nat[4] = UNMAP(S_smo[4], lb4, ub4);
	S_nat[5] = UNMAP(S_smo[5], lb5, ub5);
	
	printf("\n(2) Starting (natural): % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       S_nat[0], S_nat[1], S_nat[2], S_nat[3], S_nat[4], S_nat[5]);

	


	
	
	// invrs : CUBIC |--> TILDE
	Stm1[0] = (*invrs[0])(S_smo); // spread
	Stm1[1] = (*invrs[1])(S_smo); // interest
	Stm1[2] = (*invrs[2])(S_smo); // capital
	
	// TILDE
	printf("\n(3) Calculation:   % 9.6f % 9.6f % 9.6f\n", Stm1[0], Stm1[1], Stm1[2]);
	
	// CUBIC |--> PARA
	cube_to_para(S_smo);
	printf("\n(4) Para:   % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       S_smo[0], S_smo[1], S_smo[2], S_smo[3], S_smo[4], S_smo[5]);


	S_nat[0] = UNMAP(S_smo[0], lb0, ub0);
	S_nat[1] = UNMAP(S_smo[1], lb1, ub1);
	S_nat[2] = UNMAP(S_smo[2], lb2, ub2);
	
	S_nat[3] = UNMAP(S_smo[3], lb3, ub3);
	S_nat[4] = UNMAP(S_smo[4], lb4, ub4);
	S_nat[5] = UNMAP(S_smo[5], lb5, ub5);
	
	printf("\n(5) Starting (natural): % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       S_nat[0], S_nat[1], S_nat[2], S_nat[3], S_nat[4], S_nat[5]);




	
	// TILDE
	ArgVec<TILDE> S_inv(9, 0.0);
	
	// PARA --> TILDE
	printf("\n(6) \n================================================================\n");
	int error = solve_inverse_at_point(P, SS, LB, UB, S_smo, S_inv, decxn,
					   6, 3, 0, 2);
	printf("================================================================\n\n\n");

	S_inv[3] = S_nat[3] - SS.ln_A_ss;
	S_inv[4] = S_nat[4] - SS.ln_sg_ss;
	S_inv[5] = S_nat[5] - 0.0;
	printf("\n(7) Solution (tilde):   % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       S_inv[0], S_inv[1], S_inv[2], S_inv[3], S_inv[4], S_inv[5]);
	
	// NATURAL
	ArgVec<NATURAL> S_inv_nat(6, 0.0);
	S_inv_nat[0] = S_inv[0] + SS.ln_v_ss;
	S_inv_nat[1] = S_inv[1] + SS.ln_R_ss;
	S_inv_nat[2] = S_inv[2] + SS.ln_k_ss;
	S_inv_nat[3] = S_inv[3] + SS.ln_A_ss;
	S_inv_nat[4] = S_inv[4] + SS.ln_sg_ss;
	S_inv_nat[5] = S_inv[5] + 0.0;
	
	printf("\n(8) Solution (natural): % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       S_inv_nat[0], S_inv_nat[1], S_inv_nat[2],
	       S_inv_nat[3], S_inv_nat[4], S_inv_nat[5]);


	// SMOLYAK
	ArgVec<SMOLYAK> S_inv_smo(9, 0.0);

	S_inv_smo[0] = MAP_SMO(S_inv_nat[0], lb0, ub0);
	S_inv_smo[1] = MAP_SMO(S_inv_nat[1], lb1, ub1);
	S_inv_smo[2] = MAP_SMO(S_inv_nat[2], lb2, ub2);
	
	S_inv_smo[3] = MAP_SMO(S_inv_nat[3], lb3, ub3);
	S_inv_smo[4] = MAP_SMO(S_inv_nat[4], lb4, ub4);
	S_inv_smo[5] = MAP_SMO(S_inv_nat[5], lb5, ub5);
	
	S_inv_smo[6] = MAP_SMO(S_inv_nat[3], lb3, ub3);
	S_inv_smo[7] = MAP_SMO(S_inv_nat[4], lb4, ub4);
	S_inv_smo[8] = MAP_SMO(S_inv_nat[5], lb5, ub5);

	printf("\n(9) Solution (Smolyak): % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       S_inv_smo[0], S_inv_smo[1], S_inv_smo[2],
	       S_inv_smo[3], S_inv_smo[4], S_inv_smo[5],
	       S_inv_smo[6], S_inv_smo[7], S_inv_smo[8]);
	
	// TILDE
	double a0 = (*decxn[0])(S_inv_smo);
	double a1 = (*decxn[1])(S_inv_smo);
	double a2 = (*decxn[2])(S_inv_smo);
	double a3 = (*decxn[3])(S_inv_smo);
	double a4 = (*decxn[4])(S_inv_smo);

	printf("\n% 9.6f  % 9.6f  % 9.6f  % 9.6f  % 9.6f  % 9.6f\n\n",
	       S_inv_smo[0], S_inv_smo[1], S_inv_smo[2],
	       S_inv_smo[3], S_inv_smo[4], S_inv_smo[5]);

	printf("\n(a) Decision (tilde):   % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       a0,a1,a2,a3,a4);

	a0 += SS.ln_V_ss;
	a1 += SS.ln_i_ss;
	a2 += SS.ln_l_ss;
	a3 += SS.ln_P_ss;
	a4 += SS.ln_x_ss;

	printf("\n(b) Decision (natural): % 9.6f % 9.6f % 9.6f % 9.6f % 9.6f\n",
	       a0,a1,a2,a3,a4);

	
	double tt_v = SPR_EQN_LONG(P, SS,
				   S_inv_nat[0], S_inv_nat[1], S_inv_nat[2],
				   S_inv_nat[3], S_inv_nat[4], S_inv_nat[5], 
				   a0, a1, a2, a3, a4);

	double tt_R = INT_EQN_LONG(P,SS,
				   S_inv_nat[0], S_inv_nat[1], S_inv_nat[2],
				   S_inv_nat[3], S_inv_nat[4], S_inv_nat[5], 
				   a0, a1, a2, a3, a4);

	double tt_k = CAP_EQN_LONG(P,SS,
				   S_inv_nat[0], S_inv_nat[1], S_inv_nat[2],
				   S_inv_nat[3], S_inv_nat[4], S_inv_nat[5], 
				   a0, a1, a2, a3, a4);

	
	printf("\n(c) Transition (natur): % 9.6f % 9.6f % 9.6f\n", tt_v, tt_R, tt_k);

	printf("\n\n\n");
    };
    
    // Inverse derivatives
    for (int i = 0; i < num_snglr; i++) {	
	for (int j = 0; j < num_snglr; j++) {
	    J[i][j] = *invrs[i];
	    J[i][j].differentiate(i);
	}
    }
    
    /*------------------------------------------------------------------------------
     * OLD:
     *
     * ∂ dec   ∂ dec   ∂ smo
     * ----- = ----- * -----
     * ∂ sta   ∂ smo   ∂ sta
     *
     *          2
     * smo = ------- * (sta - LB) - 1
     *       UB - LB
     *
     * (so we take the differentiated Chebyshev, which is the middle term, and we
     * multiply it by 2/(UB-LB).)
     *
    /*------------------------------------------------------------------------------
     * NEW:
     *
     * ∂ dec   ∂ dec    ∂ cube   ∂ para
     * ----- = ------ * ------ * ------
     * ∂ sta   ∂ cube   ∂ para   ∂ sta
     *
     * ┌         ┐   ┌                  ┐   ┌         ┐
     * │ v(para) │   │ 1                │   │ v(cube) │
     * │ R(para) │   │ a  b  c  d  e  f │   │ R(cube) │  
     * │ k(para) │ = │       1          │ * │ k(cube) │
     * │ A(para) │   │          1       │   │ A(cube) │
     * │ s(para) │   │             1    │   │ s(cube) │
     * │ ξ(para) │   │                1 │   │ ξ(cube) │
     * └         ┘   └                  ┘   └         ┘
     *
     * where we have that:
     * 
     * ┌                  ┐-1   ┌                                    ┐
     * │ 1                │     │   1                                │
     * │ a  b  c  d  e  f │     │ -a/b   1/b  -c/b  -d/b  -e/b  -f/b │  
     * │       1          │   = │               1                    │
     * │          1       │     │                     1              │
     * │             1    │     │                           1        │
     * │                1 │     │                                 1  │
     * └                  ┘     └                                    ┘
     * 
     *           2
     * para = ------- * (sta - LB) - 1
     *        UB - LB
     *
     *----------------------------------------------------------------------------*/
    Jacobian = [=,&J](const double* S_arr) -> double {

	#define SUBXPROD(i,j,k) (J[0][i](S)*(J[1][j](S)*J[2][k](S)-J[1][k](S)*J[2][j](S)))
	ArgVec<SMOLYAK> S(6, 0.0);
	MAPALL;
	return fabs((d0*d1*d2)*(SUBXPROD(0,1,2)-SUBXPROD(1,0,2)+SUBXPROD(2,0,1)));
    };
    
    #define PUTELEM(M,i,j) M##mat[9*i+j] = d##j*M[i][j](S)

    #define PUTZROW(i) PUTELEM(Z,i,0), PUTELEM(Z,i,1), PUTELEM(Z,i,2), \
	               PUTELEM(Z,i,3), PUTELEM(Z,i,4), PUTELEM(Z,i,5)

    #define PUTAROW(i) PUTELEM(A,i,0), PUTELEM(A,i,1), PUTELEM(A,i,2), \
	               PUTELEM(A,i,3), PUTELEM(A,i,4), PUTELEM(A,i,5)

    // measurement derivatives
    for (int j = 0; j < num_state; j++) {
	Z[0][j] = *measr[0];
	Z[1][j] = *measr[1];
	Z[2][j] = *decxn[2];
	Z[3][j] = *decxn[3];
    }
    for (int i = 0; i < 4; i++) {
	for (int j = 0; j < num_state; j++) {
	    Z[i][j].differentiate(j);
	}
    }
    
    print_Zmat = [=,&Z](const double* S_arr, double* Zmat) -> void {
	
	ArgVec<SMOLYAK> S(6, 0.0);
	MAPALL;
	for (int i = 0; i < 45; i++) Zmat[i] = 0.0;
	PUTZROW(0);
	PUTZROW(1);
	PUTZROW(2);
	PUTZROW(3);
	Zmat[9*4+1] = 1.0; // interest rate
    };

    // transition derivatives
    for (int i = 0; i < num_snglr; i++) {
	for (int j = 0; j < num_state; j++) {
	    A[i][j] = *endog[i];
	    A[i][j].differentiate(j);
	}
    }

    print_Amat = [=,&A](const double* S_arr, double* Amat) -> void {

	ArgVec<SMOLYAK> S(6, 0.0);
	MAPALL;
	for (int i = 0; i < 81; i++) Amat[i] = 0.0;
	PUTAROW(0), PUTAROW(1), PUTAROW(2);
	Amat[9*3+3] = rho_A;
	Amat[9*4+4] = rho_sg;
	for (int i = 0; i < 3; i++) Amat[9*(num_state+i)+(3+i)] = 1.0;
    };
}
