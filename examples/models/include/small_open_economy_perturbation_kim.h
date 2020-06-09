#ifndef __SOE_PERT_KIM__
#define __SOE_PERT_KIM__

#include <vector>

#include <small_open_economy_perturbation.h>
#include <kim_filter.h>


class SOE_SGU_Kim
{
private:

    // this is the most important member of the class: a vector of small open economy
    // models, that serve 
    std::vector<SOE_SGU> models;
    
    static const int npar_all = 25;
    static const int npar     = 19;

    double* theta_all;
    double* theta_ind;

    int num_ms_vars = 2;
    int ms_var_dims[2] = {2, 2};
    int num_tot_mods = 4;

    // Q: What wouldn't otherwise appear in SOE_SGU?
    // A: Transition probabilities for markov-switching states 
    double** Pmats;

    data Y;
    kim_model mods;
    kim_opts  opts;

    
public:
    SOE_SGU_Kim();
    SOE_SGU_Kim(data Y_);
    ~SOE_SGU_Kim();
    double operator()(const double*);
};




/*
class SOE_SGU_Kim: public SOE_SGU
{
private:
    double* x_intercept;
    double* y_intercept;
public: 
    kim_model mod;
    SOE_SGU_Kim();
    SOE_SGU_Kim(data Y_, short int tag);
    SOE_SGU_Kim(data Y_): SOE_SGU_Kim(Y_, 0) {}
    
    SOE_SGU_Kim(const SOE_SGU_Kim&);
    friend void swap(SOE_SGU_Kim&, SOE_SGU_Kim&);
    SOE_SGU_Kim& operator=(SOE_SGU_Kim other)
    {
	printf("\t\tAssignment operator 'SOE_SGU_Kim' (%p)\n", this);
	swap(*this, other);
	printf("\t\tDone\n");
	return *this;
    }
    ~SOE_SGU_Kim();
    void operator()(const double*);    
};
*/

#endif
