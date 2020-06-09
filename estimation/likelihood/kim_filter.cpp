
#include "kim_filter.h"

void swap(kim_model& first, kim_model& second)
{
    printf("  ~~~Swap idiom 'kim_model.' (%p <-> %p)\n", &first, &second);
    std::swap(first.Zmat, second.Zmat);
    std::swap(first.Hmat, second.Hmat);
    std::swap(first.Tmat, second.Tmat);
    std::swap(first.Rmat, second.Rmat);
    std::swap(first.Qmat, second.Qmat);
    
    std::swap(first.Cvec, second.Cvec);
    std::swap(first.Dvec, second.Dvec);
    std::swap(first.x_ss, second.x_ss);
    printf("  ~~~Done.\n");
}
void swap(kim_opts& first, kim_opts& second)
{
    printf("  >>>Swap idiom 'kim_opts.' (%p <-> %p)\n", &first, &second);
    std::swap(first.T,   second.T);
    std::swap(first.n_x, second.n_x);
    std::swap(first.n_y, second.n_y);
    std::swap(first.n_e, second.n_e);

    std::swap(first.Pmats, second.Pmats);
    std::swap(first.n_ms_vars, second.n_ms_vars);
    std::swap(first.n_ms_dims, second.n_ms_dims);
    printf("  >>>Done.\n");
}


void print_percent_bar(double d)
{
    int id;
    if (d < 0 || d > 1) {
	printf("(?????)[           ]   ");
    } else {
	printf("(%04.1f%%)[", 100*d);
	id = (int)round(10*d);
	for (int i = 0; i < id; i++)
	    printf(" ");
	printf("*");
	for (int i = 0; i < (10-id); i++)
	    printf(" ");
	printf("]   ");
    }
}



int kim_filter_1st_order(kim_model** mods, kim_opts* opts, const double* data, double* LL_out)
{
    /**********************************************************************
     *                                                                    *
     *                        INITIAL SETUP                               *
     *                                                                    *
     **********************************************************************/

    //-----------
    // constants
    
    int T   = opts->T;
    int n_x = opts->n_x;
    int n_y = opts->n_y;
    int n_e = opts->n_e;

    int* n_ms_dims = opts->n_ms_dims; // number of states that can be inhabited
    int  n_ms_vars = opts->n_ms_vars; // number of markov switching variables

    int n_ms_st = 1;
    for (int i = 0; i < n_ms_vars; i++)
	n_ms_st *= n_ms_dims[i];

    int n_ms_ststm1 = n_ms_st * n_ms_st;

    double** Pmats = opts->Pmats;

    int verbose = 2;
    
    //------------
    // containers

    double*  Pr_ststm1 = new double [n_ms_ststm1]();
    double*  Pr_st     = new double [n_ms_st]();
    double** mar_Pr_st = new_2d_double(n_ms_vars, n_ms_dims);
    
    double** x_t_ij = new_2d_double(n_ms_ststm1, n_x);
    double** x_t_i  = new_2d_double(n_ms_st,     n_x);
	
    double** P_t_ij = new_2d_double(n_ms_ststm1, n_x*n_x);
    double** P_t_i  = new_2d_double(n_ms_st,     n_x*n_x);

    double*  ll_yt_ij = new double [n_ms_ststm1]();

    
    //------------------------
    // some important objects
    
    int* dims_dims = new int [2 * n_ms_vars]();
    for (int i = 0; i < n_ms_vars; i++)
	dims_dims[i] = dims_dims[i+n_ms_vars] = n_ms_dims[i];

    int* ind_ij = new int [2 * n_ms_vars]();

    double ll_loc;
    
    //-------------------------------
    // initialization for the filter

    for (int i = 0; i < n_ms_vars; i++)
	KIM_CALL(markov_steady_state(Pmats[i], mar_Pr_st[i], n_ms_dims[i]));

    for (int i = 0; i < dims_dims[0]; i++) {
	ind_ij[0] = i;
	KIM_CALL(recursive_initialization(n_x, n_e, mods, x_t_i, P_t_i, Pmats, Pr_st,
					  mar_Pr_st, ind_ij, dims_dims, n_ms_vars, 1));
    }
    
    /**********************************************************************
     *                                                                    *
     *                      PERFORM THE FILTER                            *
     *                                                                    *
     **********************************************************************/
    
    for (int t = 0; t < T; t++) {

	// first the exponential increase in the number of vectors

	ll_loc = 0.0;
	for (int i = 0; i < dims_dims[0]; i++) {
	    ind_ij[0] = i;
	    KIM_CALL(recursive_expand_step(n_x, n_y, n_e, mods, data+(t*n_y), x_t_i, P_t_i, x_t_ij, P_t_ij,
					   Pmats, Pr_st, mar_Pr_st, Pr_ststm1, ll_yt_ij,
					   ind_ij, dims_dims, n_ms_vars, 1, 1));
	}
	ll_loc = cblas_dasum(n_ms_ststm1, Pr_ststm1, 1);
	cblas_dscal(n_ms_ststm1, 1.0/ll_loc, Pr_ststm1, 1);
	*LL_out += log(ll_loc);

        // then the subsequent collapse for the next time period
	
	for (int i = 0; i < dims_dims[0]; i++) {
	    ind_ij[0] = i;
	    recursive_collapse_probabilities(Pr_ststm1, Pr_st, mar_Pr_st,
					     ind_ij, dims_dims, n_ms_vars, 1, 1);
	}
	
        if (verbose > 0) {
	    if (verbose > 1)
		for (int i = 0; i < n_ms_vars; i++) {
		    cblas_dscal(n_ms_dims[i], 1.0 / cblas_dasum(n_ms_dims[i], mar_Pr_st[i], 1), mar_Pr_st[i], 1);
		    print_percent_bar(mar_Pr_st[i][0]);
		}
	    printf("ll(%03d) = %+08.4f\n", t, log(ll_loc));
	}
	
	for (int i = 0; i < dims_dims[0]; i++) {
	    ind_ij[0] = i;
	    recursive_collapse_means(n_x, x_t_ij, x_t_i, Pr_st, Pr_ststm1,
				     ind_ij, dims_dims, n_ms_vars, 1, 1);
	}

	for (int i = 0; i < dims_dims[0]; i++) {
	    ind_ij[0] = i;
	    recursive_collapse_covariances(n_x, x_t_ij, x_t_i, P_t_ij, P_t_i, Pr_st,
					   Pr_ststm1, ind_ij, dims_dims, n_ms_vars, 1, 1);
	}
    }
    
    /**********************************************************************
     *                                                                    *
     *                        FINAL CLEANUP                               *
     *                                                                    *
     **********************************************************************/
    
    delete[] Pr_st;
    delete[] Pr_ststm1;
    delete_2d_double(mar_Pr_st, n_ms_vars);

    delete_2d_double(x_t_ij, n_ms_ststm1);
    delete_2d_double(x_t_i,  n_ms_st);
	
    delete_2d_double(P_t_ij, n_ms_ststm1);
    delete_2d_double(P_t_i,  n_ms_st);
    
    delete[] ll_yt_ij;
    delete[] dims_dims;
    delete[] ind_ij;
    
    return 0;
}



/**
 * This is a function to recursively initialize the Kim filter. Note that we don't
 * have to double-dive for this one.
 */ 
int recursive_initialization(int n_x, int n_e, kim_model** mods,
			     double** x_t_i, double** P_t_i, double** Pmats,
			     double* Pr_st, double** mar_Pr_st, int* ind_ij,
			     const int* dims, int n_dim, int d)
{
    int i_i, i_d;
    
    for (int c = 0; c < dims[d]; c++) {
 
	i_d = d;
	ind_ij[i_d] = c;
	
	if (d+1 < n_dim) {
	    
	    KIM_CALL(recursive_initialization(n_x, n_e, mods, x_t_i, P_t_i, Pmats, Pr_st, mar_Pr_st,
					      ind_ij, dims, n_dim, d+1));
	} else {
	    
	    i_i  = rowMajorIndex(ind_ij, dims, n_dim);
	    Pr_st[i_i] = 1.0;
	    
	    // we initialize the joint probabilities by taking the product of all n_dim
	    // of the marginal probabilities, in the state corresponding to the current index.
	    for (int v = 0; v < n_dim; v++)
		Pr_st[i_i] *= mar_Pr_st[v][ind_ij[v]];

	    // then we initialize the state means and covariances 
	    for (int i = 0; i < n_x; i++)
		x_t_i[i_i][i] = mods[i_i]->x_ss[i];
	    
	    KIM_CALL(steady_state_covariance(n_x, n_e, mods[i_i]->Tmat, mods[i_i]->Rmat,
					     mods[i_i]->Qmat, P_t_i[i_i]));
	    
	}
    }
    // reset the current state over which we just looped
    ind_ij[i_d] = 0;

    return 0;
}


/**
 * This is a function to recursively collapse all of the mean vectors for the next
 * loop of the Kim filter.
 */ 
void recursive_collapse_covariances(int n_x, double** x_t_ij, double** x_t_i,
				    double** P_t_ij, double** P_t_i,
				    const double* Pr_st, const double* Pr_ststm1,
				    int* ind_ij, const int* dims, int n_dim, int d, int type)
{
    int i_i, i_j, i_ij, i_d;

    for (int c = 0; c < dims[d]; c++) {
 
	i_d = (type == 1) ? d : n_dim+d;
	ind_ij[i_d] = c;
	
	if (d+1 < n_dim) {

	    recursive_collapse_covariances(n_x, x_t_ij, x_t_i, P_t_ij, P_t_i, Pr_st,
					   Pr_ststm1, ind_ij, dims, n_dim, d+1, type);
	} else {
	    
	    if (type == 1) {

		// now is the time to make sure the current state is 0
		i_i = rowMajorIndex(ind_ij, dims, n_dim);
		set_to_zero(P_t_i[i_i], n_x*n_x);
		
		for (int p = 0; p < dims[n_dim]; p++) {

		    ind_ij[n_dim] = p;
		    recursive_collapse_covariances(n_x, x_t_ij, x_t_i, P_t_ij, P_t_i, Pr_st,
						   Pr_ststm1, ind_ij, dims, n_dim, 1, 2);
		}
		ind_ij[n_dim] = 0;
		
		// now is the time to scale the state
		cblas_dscal(n_x*n_x, 1.0 / Pr_st[i_i], P_t_i[i_i], 1);
		
		
	    } else {
		
		i_i  = rowMajorIndex(ind_ij, dims, n_dim);
		i_ij = rowMajorIndex(ind_ij, dims, n_dim+n_dim);

		// add another component
		for (int i = 0; i < n_x; i++)
		    for (int j = 0; j < n_x; j++)
			P_t_i[i_i][n_x*i + j] +=
			    (Pr_ststm1[i_ij] *
			     (P_t_ij[i_ij][n_x*i + j] +
			      ((x_t_i[i_i][i] - x_t_ij[i_ij][i]) *
			       (x_t_i[i_i][j] - x_t_ij[i_ij][j]))));
		
	    }
	}
    }
    // reset the current state over which we just looped
    ind_ij[i_d] = 0;
	
}


/**
 * This is a function to recursively collapse all of the mean vectors for the next
 * loop of the Kim filter.
 */ 
void recursive_collapse_means(int n_x, double** x_t_ij, double** x_t_i,
			      const double* Pr_st, const double* Pr_ststm1,
			      int* ind_ij, const int* dims, int n_dim, int d, int type)
{
    int i_i, i_j, i_ij, i_d;

    for (int c = 0; c < dims[d]; c++) {

	i_d = (type == 1) ? d : n_dim+d;
	ind_ij[i_d] = c;
	
	if (d+1 < n_dim) {

	    recursive_collapse_means(n_x, x_t_ij, x_t_i, Pr_st, Pr_ststm1,
				     ind_ij, dims, n_dim, d+1, type);
	} else {
	    
	    if (type == 1) {

		// now is the time to make sure the current state is 0
		i_i = rowMajorIndex(ind_ij, dims, n_dim);
		set_to_zero(x_t_i[i_i], n_x);
		
		for (int p = 0; p < dims[n_dim]; p++) {

		    ind_ij[n_dim] = p;
		    recursive_collapse_means(n_x, x_t_ij, x_t_i, Pr_st, Pr_ststm1,
					     ind_ij, dims, n_dim, 1, 2);
		}
		ind_ij[n_dim] = 0;

		// now is the time to scale the state
		cblas_dscal(n_x, 1.0 / Pr_st[i_i], x_t_i[i_i], 1);
		
		
	    } else {
		
		i_i  = rowMajorIndex(ind_ij, dims, n_dim);
		i_ij = rowMajorIndex(ind_ij, dims, n_dim+n_dim);

		// add another component
		for (int i = 0; i < n_x; i++)
		    x_t_i[i_i][i] += (Pr_ststm1[i_ij] * x_t_ij[i_ij][i]);
		
	    }
	}
    }
    // reset the current state over which we just looped
    ind_ij[i_d] = 0;
	
}

int are_all(int* vec, int target, int n)
{
    for (int i = 0; i < n; i++)
	if (vec[i] != target)
	    return 0;
    return 1;
}

/**
 * This is a function to recursively collapse the probabilities of the discrete states.
 * It is twofold in the sense that it obtains both the probability of the current states,
 * as well as the marginal probabilities of each discrete variable.
 *
 * (expansion version has comments)
 */ 
void recursive_collapse_probabilities(const double* Pr_ststm1, double* Pr_st, double** mar_Pr_st,
				     int* ind_ij, const int* dims, int n_dim, int d, int type)
{
    int i_i, i_j, i_ij, i_d;

    for (int c = 0; c < dims[d]; c++) {
	
	i_d = (type == 1) ? d : n_dim+d;
	ind_ij[i_d] = c;
	
	if (d+1 < n_dim) {

	    recursive_collapse_probabilities(Pr_ststm1, Pr_st, mar_Pr_st,
					     ind_ij, dims, n_dim, d+1, type);
	    
	} else {
	    
	    if (type == 1) {

		// now is the time to make sure the current probability is 0
		i_i = rowMajorIndex(ind_ij, dims, n_dim);
		Pr_st[i_i] = 0.0;

		for (int p = 0; p < dims[n_dim]; p++) {

		    ind_ij[n_dim] = p;
		    recursive_collapse_probabilities(Pr_ststm1, Pr_st, mar_Pr_st,
						     ind_ij, dims, n_dim, 1, 2);
		    
		}
		ind_ij[n_dim] = 0;
		
 		// now is the time to do the extra collapse into marginals
		// If it is the very first, then set them to zero.
		if (are_all(ind_ij, 0, n_dim))
		    for (int i = 0; i < n_dim; i++)
			for (int j = 0; j < dims[i]; j++)
			    mar_Pr_st[i][j] = 0.0;
		for (int v = 0; v < n_dim; v++) {

		    // note that we've just obtained a joint likelihood, but that
		    // this joint likelihood is part of n_dim different marginal
		    // likelihoods. So, for each variable that we're touching, we're
		    // going to give something back.
		    mar_Pr_st[v][ind_ij[v]] += Pr_st[i_i];
		}
		
	    } else {

		i_i  = rowMajorIndex(ind_ij, dims, n_dim);
		i_ij = rowMajorIndex(ind_ij, dims, n_dim+n_dim);

		// At each combination of state indices, we need to add to Pr_st one of
		// the elements of Pr_ststm1.
		Pr_st[i_i] += Pr_ststm1[i_ij];
		
	    }
	}
    }
    ind_ij[i_d] = 0;
	
}

/**
 * This is a function to recursively hit all the dimensions of x_t_ij. If type == 1, then
 * we are running over all the contemporaneous states; if type == 2, then we are running
 * over the previous times.
 *
 * It also recursively expands the probabilities of the states.
 *
 * d : which dimension to do (zero-based)
 */ 
int recursive_expand_step(int n_x, int n_y, int n_e, kim_model** mods, const double* data,
			  double** x_t_j, double** P_t_j,
			  double** x_t_ij, double** P_t_ij,
			  double** Pmats, const double* Pr_stm1, double** mar_Pr_stm1,
			  double* Pr_ststm1, double* ll_yt_ij, int* ind_ij,
			  const int* dims, int n_dim, int d, int type)
{
    // we might need to know which state to pick
    int i_i, i_j, i_ij, i_d;

    // loop over states in this dimension
    for (int c = 0; c < dims[d]; c++) {

	// set the index equal to the corresponding spot
	i_d = (type == 1) ? d : n_dim+d;
	ind_ij[i_d] = c;
	
	if (d+1 < n_dim) {

	    // if we're not at full dimension, go deeper
	    KIM_CALL(recursive_expand_step(n_x, n_y, n_e, mods, data, x_t_j, P_t_j, x_t_ij, P_t_ij,
					   Pmats, Pr_stm1, mar_Pr_stm1, Pr_ststm1, ll_yt_ij,
					   ind_ij, dims, n_dim, d, type));
	    
	} else {

	    // if we're still in the first stage, then we must start again
	    if (type == 1) {
		
		// otherwise, we should start to loop over the previous states
		for (int p = 0; p < dims[n_dim]; p++) {

		    // set the first previous state to p
		    ind_ij[n_dim+0] = p;

		    // call the recursion over the previous states
		    KIM_CALL(recursive_expand_step(n_x, n_y, n_e, mods, data, x_t_j, P_t_j, x_t_ij, P_t_ij,
						   Pmats, Pr_stm1, mar_Pr_stm1, Pr_ststm1, ll_yt_ij,
						   ind_ij, dims, n_dim, 1, 2));
		}
		// reset the first previous state
		ind_ij[n_dim+0] = 0;
		
	    } else {
		
		// obtain the indices of the final location at which you've arrived
		i_i  = rowMajorIndex(ind_ij,       dims,       n_dim);
		i_j  = rowMajorIndex(ind_ij+n_dim, dims+n_dim, n_dim);
		i_ij = rowMajorIndex(ind_ij,       dims,       n_dim+n_dim);

		
		//======================================================================//
		//                           KALMAN STEP                                //
		//======================================================================//

		predict_mu_1st_order(n_x, n_x, x_t_j[i_j], x_t_ij[i_ij],
				     mods[i_i]->Tmat, mods[i_i]->Cvec);

		predict_cov_1st_order(n_x, n_e, P_t_j[i_j], P_t_ij[i_ij],
				      mods[i_i]->Tmat, mods[i_i]->Rmat, mods[i_i]->Qmat);
		
		KIM_CALL(update_mu_cov_1st_order(n_x, n_y, x_t_ij[i_ij], P_t_ij[i_ij],
						 data, mods[i_i]->Zmat, mods[i_i]->Hmat,
						 mods[i_i]->Dvec, &ll_yt_ij[i_ij]));

		
		//======================================================================//
		//                          HAMILTON STEP                               //
		//======================================================================//
		// the gag is, you don't just have to hit each of these spots once, but
		// you have to hit each of the combinations of states n_dim times

		Pr_ststm1[i_ij] = exp(ll_yt_ij[i_ij]);
		for (int v = 0; v < n_dim; v++) {

		    // note that since the transition matrices individually add up to one,
		    // we must multiply them with the marginal probabilities which also add
		    // up to one.
		    Pr_ststm1[i_ij] *= Pmats[v][dims[v]*ind_ij[v] + ind_ij[n_dim+v]];
		    Pr_ststm1[i_ij] *= mar_Pr_stm1[v][ind_ij[n_dim+v]];
		    
		}
	    }
	}
    }
    // reset the current state over which we just looped
    ind_ij[i_d] = 0;
	
    return 0;
}



double** new_2d_double(int n1, int n2)
{
    double** ptr = new double* [n1];
    for (int i = 0; i < n1; i++)
	ptr[i] = new double [n2]();

    return ptr;
}


double** new_2d_double(int n1, int* n2)
{
    double** ptr = new double* [n1];
    for (int i = 0; i < n1; i++)
	ptr[i] = new double [n2[i]]();

    return ptr;
}


void delete_2d_double(double** ptr, int n1)
{
    for (int i = 0; i < n1; i++)
	delete[] ptr[i];
    delete[] ptr;
}


void set_to_zero(double* vec, int n)
{
    for (int i = 0; i < n; i++)
	vec[i] = 0.0;
}


int find_one(const double* vec, int n)
{
    double diffs[n];
    for (int i = 0; i < n; i++)
	diffs[i] = fabs(vec[i] - 1.0);

    int out = 0;
    for (int i = 1; i < n; i++)
	out = (diffs[out] < diffs[i]) ? out : i;

    printf("At index %d, the value %f was determined to be closest to one.\n", out, vec[out]);
    return out;
}


int markov_steady_state(const double* Pmat, double* p_ss, int n)
{
    double epsilon = 1.0e-10;
    double sum_check = 0.0;
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++)
	    sum_check += Pmat[n*j + i];
	if (fabs(sum_check - 1.0) > epsilon)
	    printf("WARNING: probabilities may not add up. (see %f)\n", sum_check);
	sum_check = 0.0;
    }

    double Ploc[n*n];
    for (int i = 0; i < n*n; i++)
	Ploc[i] = Pmat[i];

    double  wr[n];
    double  wi[n];
    double* vl;
    double  vr[n * n];

    KIM_CALL(LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, Ploc, n, wr, wi, vl, n, vr, n));
    int i_one = find_one(wr, n);

    for (int i = 0; i < n; i++)
	p_ss[i] = vr[n*i + i_one];

    // maybe they're all negative -- then give them a switch
    if (p_ss[0] < 0.0)
	for (int i = 0; i < n; i++)
	    p_ss[i] *= -1.0;
    
    double scale = 0.0;
    for (int i = 0; i < n; i++) {
	if (p_ss[i] < 0.0) {
	    printf("ERROR: negative probability. %s: %d (%s)\n", __FILE__, __LINE__, __func__);
	    return 1;
	}
	scale += p_ss[i];
    }

    for (int i = 0; i < n; i++)
	p_ss[i] /= scale;
    
    return 0;
}
