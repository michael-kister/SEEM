
#include "likelihood_subroutines.h"

void likelihood_setup
(const Parameters& P, const SteadyStates& SS, const double* LB, const double* UB,
 std::vector<AppVec>& J, std::vector<AppVec>& Z, std::vector<AppVec>& A,
 std::function<double(const double*)>& Jacobian,
 DblDbl& measr_eqn, DblDbl& trans_eqn, DblDbl& print_Zmat, DblDbl& print_Amat,
 Approximation** decxn, Approximation** measr, Approximation** endog, Approximation** invrs)
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
    double ss3 = P.ln_A_ss;
    double ss4 = P.ln_sg_ss;
    double ss5 = 0.0;    
    
    double d0 = 2.0/(UB[0]-LB[0]);
    double d1 = 2.0/(UB[1]-LB[1]);
    double d2 = 2.0/(UB[2]-LB[2]);
    double d3 = 2.0/(UB[3]-LB[3]);
    double d4 = 2.0/(UB[4]-LB[4]);
    double d5 = 2.0/(UB[5]-LB[5]);

    double rho_A = P.rho_A;
    double rho_sg = P.rho_sg;

    double yss0 = SS.ln_y_ss;
    double yss1 = SS.ln_c_ss;
    double yss2 = SS.ln_l_ss;
    double yss3 = P.ln_P_ss;
    
    #define MAPSTATE(i) S[i] = MAP_SMO(S_arr[i]+ss##i,lb##i,ub##i)
    
    #define MAPALL MAPSTATE(0), MAPSTATE(1), MAPSTATE(2), MAPSTATE(3), MAPSTATE(4), MAPSTATE(5)

    trans_eqn = [=,&endog](const double* S_arr, double* Stp1) -> void {
	
	std::vector<double> S(6, 0.0);
	MAPALL;
	Stp1[0] = (*endog[0])(S) - ss0;
	Stp1[1] = (*endog[1])(S) - ss1;
	Stp1[2] = (*endog[2])(S) - ss2;
	Stp1[3] = rho_A*S_arr[3];
	Stp1[4] = rho_sg*S_arr[4];
	Stp1[5] = 0.0;
    };
    
    measr_eqn = [=,&decxn,&measr](const double* S_arr, double* Yt) -> void {
	
	std::vector<double> S(6, 0.0);
	MAPALL;
	Yt[0] = (*measr[0])(S) - yss0; // output
	Yt[1] = (*measr[1])(S) - yss1; // consumption
	Yt[2] = (*decxn[2])(S) - yss2; // labor
	Yt[3] = (*decxn[3])(S) - yss3; // inflation
	Yt[4] = S_arr[1];              // interest rate
    };
    
    // Inverse derivatives
    for (int i = 0; i < num_snglr; i++) {	
	for (int j = 0; j < num_snglr; j++) {
	    J[i][j] = *invrs[i];
	    J[i][j].differentiate(i);
	}
    }

    Jacobian = [=,&J](const double* S_arr) -> double {

	#define SUBXPROD(i,j,k) (J[0][i](S)*(J[1][j](S)*J[2][k](S)-J[1][k](S)*J[2][j](S)))
	std::vector<double> S(6, 0.0);
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
	
	std::vector<double> S(6, 0.0);
	MAPALL;
	for (int i = 0; i < 45; i++) Zmat[i] = 0.0;
	PUTZROW(0), PUTZROW(1);PUTZROW(2);
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

	std::vector<double> S(6, 0.0);
	MAPALL;
	for (int i = 0; i < 81; i++) Amat[i] = 0.0;
	PUTAROW(0), PUTAROW(1), PUTAROW(2);
	Amat[9*3+3] = rho_A;
	Amat[9*4+4] = rho_sg;
	for (int i = 0; i < 3; i++) Amat[9*(num_state+i)+(3+i)] = 1.0;
    };

	    
}
