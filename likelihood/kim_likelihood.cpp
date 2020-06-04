
// requires exp
// requires matrix multiplication

#include "kim_likelihood.h"

double kim_likelihood(int nstate, int nobs, int nT, int nregime, double *data, ssm *models[], double *trans) {

    //------------------------------------------------------------------
    // NOTES
    //
    // Some notation that should be obeyed throughout this function:
    // 
    // 1. Indices with variable name "i" should always refer to the
    //    lagged time index.
    // 2. Indices with variable name "j" should always refer to the
    //    current time index.
    // 3. We index with variable names in alphabetical order -- so "i"
    //    preceeds "j." Also, we will say that "i" refers to "row," and
    //    "j" refers to "column."
    // 4. trans[i][j] := Pr[ s=j | s`=i ], which must add up along "j",
    //    therefore we must have that the rows all add up to one.

    
    //------------------------------------------------------------------
    // SETUP

    double log_lik = 0.0; // log-likelihood
    double lik_t   = 1.0; // individual log-likelihood
    double t_check;
    
    // multi-dimensional arrays of pointers to state quantities
    double *x_ij_t  [nregime][nregime]; // predictive mean
    double *x_ij_tt [nregime][nregime]; // updated mean
    double *x_j_tt  [nregime];          // collapsed filtered mean
    
    double *P_ij_t  [nregime][nregime]; // predictive cov
    double *P_ij_tt [nregime][nregime]; // updated cov
    double *P_j_tt  [nregime];          // collapsed filtered cov
    
    // 1D arrays to probability filtering quantities
    double Pr_ij_t [nregime*nregime];   // joint probabilities
    double Pr_y_t  [nregime*nregime];   // conditional probabilities (from MVN)
    double Pr_j_t  [nregime];           // integrated probabilities  (from sum)

    // multi-dimensional arrays of pointers to filtering tools
    double *v_t [nregime][nregime];
    double *F_t [nregime][nregime];
    
    // additional data containers
    double dx_ij_tt [nstate];

    // arrays for steady state Markov probability
    
    
    // allocate memory
    for (int i = 0; i < nregime; i++) {
	for (int j = 0; j < nregime; j++) {
	    x_ij_t  [i][j] = new double(nstate);
	    x_ij_tt [i][j] = new double(nstate);
	    
	    P_ij_t  [i][j] = new double(nstate*nstate);
	    P_ij_tt [i][j] = new double(nstate*nstate);
	    
	    v_ij_t  [i][j] = new double(nobs);
	    F_ij_t  [i][j] = new double(nobs*nobs);
	}
	x_j_tt [i] = new double(nstate);
	P_j_tt [i] = new double(nstate);
    }
    
    // initialize Pr[S|Y] using eigenvalues

    // initialize x_j_tt, P_j_tt using the different models
    
    
    // t : time
    for (int t = 0; t < nT; t++) {
	
	// ---------------
	// State Filtering
	
	for (int j = 0; j < nregime; j++) {
	    for (int i = 0; i < nregime; i++) {
		models[j]->predictive_step (x_j_tt[i], P_j_tt[i], x_ij_t[j][i], P_ij_t[j][i]);
		models[j]->updating_step   (x_ij_t[i], P_ij_t[i], data + t*nobs, x_ij_tt[j][i],
					    P_ij_tt[j][i], v_ij_t[j][i], F_ij_t[j][i]);
	    }
	}
	
	
	// ---------------------
	// Probability Filtering
	
	// Pr[S,S`|Y]
	for (int j = 0; j < nregime; j++) {
	    for (int i = 0; i < nregime; i++) {
		Pr_y_t  [j+i*nregime] = log_mvn_density(v_ij_t[j][i], F_ij_t[j][i], nobs);
		Pr_ij_t [j+i*nregime] = Pr_y_t[j+i*nregime] + trans[j+i*nregime] + Pr_j_t[i];
		Pr_ij_t [j+i*nregime] = exp(Pr_ij_t[j+i*nregime]);
	    }
	}
	// Pr[S|Y]
	for (int j = 0; j < nregime; j++) {
	    Pr_j_t[j] = 0;
	    for (int i = 0; i < nregime; i++) {
		Pr_j_t[j] += Pr_ij_t[j+i*nregime];
	    }
	}
	// Pr[y|Y`]
	lik_t = 1.0;
	for (int j = 0; j < nregime; j++) {
	    lik_t += Pr_j_t[j];
	}
	log_lik += log(lik_t);
	// Pr[S|Y], Pr[S,S`|Y]
	for (int j = 0; j < nregime; j++) {
	    Pr_j_t[j] /= lik_t;
	    for (int i = 0; i < nregime; i++) {
		Pr_ij_t[j+i*nregime] /= lik_t;
	    }
	}
	
	
	// ----------------
	// State Collapsing
	
	for (int j = 0; j < nregime; j++) {
	    for (int i = 0; i < nregime; i++) {
		for (int k = 0; k < nstate; k++) {
		    x_j_tt[j][k] = Pr_ij_t[j+i*nregime] * x_ij_tt[j][i][k] / Pr_j_t[j];
		}
	    }
	}
	for (int j = 0; j < nregime; j++) {
	    for (int i = 0; i < nregime; i++) {
		for (int k = 0; k < nstate; k++)
		    dx_ij_tt[k] = x_j_tt[j][k] - x_ij_tt[j][i][k];
		for (int k = 0; k < nstate*nstate; k++)
		    P_j_tt[j][i][k] = 0.0;
		LAPACKE_dgemm('N','N',nstate,nstate,1,1.0,dx_ij_tt,nstate,dx_ij_tt,1,1.0,&P_ij_tt[j][i]);
		for (int k = 0; k < nstate*nstate; k++) {
		    P_j_tt[j][k] = Pr_ij_t[j+i*nregime] * P_ij_tt[j][i][k] / Pr_j_t[j];
		}
	    }
	}
	
    }
    
    //------------------------------------------------------------------
    // CLEANUP
    
    // deallocate memory
    for (int i = 0; i < nregime; i++) {
	for (int j = 0; j < nregime; j++) {
	    delete[] x_ij_t  [i][j];
	    delete[] x_ij_tt [i][j];
	    
	    delete[] P_ij_t  [i][j];
	    delete[] P_ij_tt [i][j];

	    delete[] v_ij_t  [i][j];
	    delete[] F_ij_t  [i][j];
	}
	delete[] x_j_t  [i];
	delete[] P_j_tt [i];
    }
    
    // output
    return log_lik;

}
