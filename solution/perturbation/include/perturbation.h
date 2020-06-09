#ifndef _PERTURBATION_
#define _PERTURBATION_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <set>

#include <mkl.h>
#include <adolc/adolc.h>

#include <matrices.h>

#ifndef SGU_CALL
#define SGU_CALL(x) do { if ((x) != 0) { \
	    printf("ERROR: %s : %d\n", __FILE__, __LINE__);    \
	    return 1;}} while(0)
#endif



lapack_logical mod_gt_one(const double* ar, const double* ai, const double* be);

void insert_sort_int(int*, int);

void flatten_tensor(double*, double*, int, int, int, int);


/**
 * -----------------------------------------------------------------------------
 * This is the lowest-level class relevant for solving a model via perturbation
 * methods.
 *
 * Its main function is to store arrays containing first- and second-order
 * coefficients for quadratic approximations to transition and observation
 * functions.
 * -----------------------------------------------------------------------------
 */
class SGU_BASE
{
private:

    // methods relevant to setting stuff up
    void block_size_recursion(int*, int*, int*, int, int, int, int);
    void derivative_map_recursion1(int**, int*, int*, int*, int, int, int*, int*);
    void derivative_map_recursion2(int**, int*, int*, int*, int*, int, int, int*, int*);
    
protected:

    // constants:
    int nx, ny, neps, npar, n_all, deg;

    // parameters: (soon to be replaced by its own class)
    double* theta;

    // state space matrices
    double *x_ss, *y_ss, *eta, sigma;
    double *gx, *hx, *gxx, *hxx, *gss, *hss;
    double *g_lev_1, *h_lev_1, *gx_lev_1, *hx_lev_1;
    double *g_lev_2, *h_lev_2, *gx_lev_2, *hx_lev_2, *gxx_lev_2, *hxx_lev_2;

    // ADOL-C Setup
    static const int nblock = 25;
    int *block_sizes, **derivative_map;
    double** derivatives;
    double* X_ss;
    short int tag;

    // ADOL-C Implementation
    int tensor_length;
    double** adolc_tensor;

public:
    
    // standard constructors
    SGU_BASE();
    SGU_BASE(int nx_, int ny_, int neps_, int npar_, int deg_, short int tag);
    SGU_BASE(int nx_, int ny_, int neps_, int npar_, int deg_) :
	SGU_BASE(nx_, ny_, neps_, npar_, deg_, 0) {}

    // destructor
    ~SGU_BASE();

    // copy constructor
    SGU_BASE(const SGU_BASE& that);

    // swap function
    friend void swap(SGU_BASE&, SGU_BASE&);

    // copy assignment operator
    SGU_BASE& operator=(SGU_BASE other)
    {
	printf("\t\t\t\t\tCopy assignment operator 'SGU_BASE' (%p receiving %p)\n\n", this, &other);
	swap(*this, other);
	printf("\t\t\t\t\tDone.\n\n");
	return *this;
    }

};


// this class has all the methods, but is an abstract class because of its
// pure virtual functions. This makes it hard to copy, since it cannot be
// instantiated. Hopefully this is a reasonable workaround.
class perturbation: public SGU_BASE
{
private:

    // obtaining the derivatives
    void differentiate_tag();
    void map_tetrahedral_to_tensors();

    // solving the model    
    int solve_gx_hx();
    int solve_gxx_hxx();
    int solve_gss_hss();
    void ghxx_fun(double*, double*, double*, double*, int, int, int, int);
    void ghss_fun(double*, double*, double*, double*, int, int, int, int);

    // converting to a level solution
    void set_level_solution();

protected:

    // model-specific functions
    virtual void set_steady_state() = 0;
    virtual void perform_autodiff() = 0;
    
public:

    // print the matrices
    void display(int);
    
    // default constructor (shouldn't really be used)
    perturbation();

    // standard constructor
    perturbation(int nx_, int ny_, int neps_, int npar_, int deg_, short int tag_) :
	SGU_BASE(nx_, ny_, neps_, npar_, deg_, tag_)
    {
	printf("\t\t\t\tConstructing 'perturbation' (%p)\n\n", this);
	printf("\t\t\t\tDone.\n\n");	
    }
    perturbation(int nx_, int ny_, int neps_, int npar_, int deg_) :
	perturbation(nx_, ny_, neps_, npar_, deg_, 0) {}

    perturbation(const perturbation& other): SGU_BASE(other)
    {
	printf("\t\t\t\tCopy constructor 'perturbation' (%p receiving %p)\n\n", this, &other);
	printf("\t\t\t\tDone.\n\n");
    }

    ~perturbation()
    {
	printf("\t\t\t\t-Destructing 'perturbation' (%p)... ", this);
	printf("Done.\n\n");	
    }
    
    int load_parameters(double*, int = 0);
};

#endif
