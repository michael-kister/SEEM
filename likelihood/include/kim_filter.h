#ifndef __KIM_FILTER__
#define __KIM_FILTER__

#include <mkl.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include <miscellaneous.h>
#include <matrices.h>
#include <kalman_utilities.h>

#ifndef KIM_CALL
#define KIM_CALL(x) do { if ((x) != 0) { \
	    printf("ERROR: %s : %d (%s)\n", __FILE__, __LINE__, __func__); \
	    return 1;}} while(0)
#endif


struct kim_model
{
    double* Zmat;
    double* Hmat;
    double* Tmat;
    double* Rmat;
    double* Qmat;

    double* Cvec;
    double* Dvec;
    double* x_ss;

    kim_model(){};
    kim_model(double* Zmat_, double* Hmat_, double* Tmat_, double* Rmat_, double* Qmat_,
	      double* Cvec_, double* Dvec_, double* x_ss_) :
	Zmat(Zmat_), Hmat(Hmat_), Tmat(Tmat_), Rmat(Rmat_), Qmat(Qmat_),
	Cvec(Cvec_), Dvec(Dvec_), x_ss(x_ss_)
    {
    }
    kim_model(const kim_model& that):
	Zmat(that.Zmat), Hmat(that.Hmat), Tmat(that.Tmat), Rmat(that.Rmat), Qmat(that.Qmat),
	Cvec(that.Cvec), Dvec(that.Dvec), x_ss(that.x_ss)
    {
	printf("  ~Copy constructor for 'kim_model.'(%p <- %p)\n  ~Done.\n", this, &that);
    }
    friend void swap(kim_model&, kim_model&);
    kim_model& operator=(kim_model other)
    {
	printf("  ~~Assignment operator 'kim_model.' (%p)\n", this);
	swap(*this, other);
	printf("  ~~Done.\n");
	return *this;
    }
};


struct kim_opts
{
    int T;
    int n_x; // number of unobservable states
    int n_y; // number of observables
    int n_e; // number of shocks
    
    double** Pmats; // array of markov transition arrays

    int* n_ms_dims;
    int  n_ms_vars;

    kim_opts(){};
    kim_opts(int T_, int n_x_, int n_y_, int n_e_, double** Pmats_,
	     int n_ms_vars_, int* n_ms_dims_) :
	T(T_), n_x(n_x_),  n_y(n_y_),  n_e(n_e_), Pmats(Pmats_),
	n_ms_vars(n_ms_vars_), n_ms_dims(n_ms_dims_)
    {
    }
    kim_opts(const kim_opts& that):
	T(that.T), n_x(that.n_x),  n_y(that.n_y),  n_e(that.n_e), Pmats(that.Pmats),
	n_ms_vars(that.n_ms_vars), n_ms_dims(that.n_ms_dims)
    {
	printf("  >Copy constructor for 'kim_opts.'\n  >Done.\n");
    }
    friend void swap(kim_opts&, kim_opts&);
    kim_opts& operator=(kim_opts other)
    {
	printf("  >>Assignment operator 'kim_opts.'\n");
	swap(*this, other);
	printf("  >>Done.\n");
	return *this;
    }
};


double** new_2d_double(int n1, int n2);
double** new_2d_double(int n1, int* n2);
void delete_2d_double(double** ptr, int n1);
int find_one(const double* vec, int n);
int markov_steady_state(const double* Pmat, double* p_ss, int n);
void set_to_zero(double* vec, int n);


int kim_filter_1st_order(kim_model** mods, kim_opts* opts, const double* data, double* LL_out);

int recursive_initialization(int n_x, int n_e, kim_model** mods,
			     double** x_t_i, double** P_t_i, double** Pmats,
			     double* Pr_st, double** mar_Pr_st, int* ind_ij,
			     const int* dims, int n_dim, int d);

int recursive_expand_step(int n_x, int n_y, int n_e, kim_model** mods, const double* data,
			  double** x_t_j, double** P_t_j,
			  double** x_t_ij, double** P_t_ij,
			  double** Pmats, const double* Pr_stm1, double** mar_Pr_stm1,
			  double* Pr_ststm1, double* ll_yt_ij, int* ind_ij,
			  const int* dims, int n_dim, int d, int type);

void recursive_collapse_probabilities(const double* Pr_ststm1, double* Pr_st, double** mar_Pr_st,
				     int* ind_ij, const int* dims, int n_dim, int d, int type);

void recursive_collapse_means(int n_x, double** x_t_ij, double** x_t_i,
			     const double* Pr_st, const double* Pr_ststm1,
			     int* ind_ij, const int* dims, int n_dim, int d, int type);

void recursive_collapse_covariances(int n_x, double** x_t_ij, double** x_t_i,
				   double** P_t_ij, double** P_t_i,
				   const double* Pr_st, const double* Pr_ststm1,
				   int* ind_ij, const int* dims, int n_dim, int d, int type);








#endif
