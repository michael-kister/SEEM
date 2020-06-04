#ifndef __SOE_PERT_MS_GPU__
#define __SOE_PERT_MS_GPU__

#include <vector>

#include <small_open_economy_perturbation.h>
#include <kim_filter.h>


/**
 * You don't actually want a class here -- all you need is SOE_SGU, and then make a vector
 * of them.
 */

class SOE_SGU_MSGPU: public SOE_SGU
{
private:
    double* x_intercept;
    double* y_intercept;
public: 
    kim_model mod;
    SOE_SGU_MSGPU();
    SOE_SGU_MSGPU(data Y_, short int tag);
    SOE_SGU_MSGPU(data Y_): SOE_SGU_MSGPU(Y_, 0) {}
    
    SOE_SGU_MSGPU(const SOE_SGU_MSGPU&);
    friend void swap(SOE_SGU_MSGPU&, SOE_SGU_MSGPU&);
    SOE_SGU_MSGPU& operator=(SOE_SGU_MSGPU other)
    {
	printf("\t\tAssignment operator 'SOE_SGU_MSGPU' (%p)\n", this);
	swap(*this, other);
	printf("\t\tDone\n");
	return *this;
    }
    ~SOE_SGU_MSGPU();
    void operator()(const double*);    
};





class MS_SOE_SGU_MSGPU
{
private:
    
    static const int npar_all = 25;
    static const int npar     = 19;

    int nx = 6;
    int n_me = 4;
    int neps = 4;

    int num_ms_vars = 2;
    int ms_var_dims[2] = {2, 2};

    double** Pmats;
    double* theta_all;
    double* theta_ind;

    std::vector<SOE_SGU_MSGPU> mods;
    
    // an array of pointers (not a 2D array), each of which basically just
    // contains more pointers
    kim_model** k_mods;

    kim_opts k_opts;
    data Y;
public:
    MS_SOE_SGU_MSGPU();
    MS_SOE_SGU_MSGPU(data Y_);
    ~MS_SOE_SGU_MSGPU();
    double operator()(const double*);
};





#endif
