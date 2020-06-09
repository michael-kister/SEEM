#ifndef __SOE_PERT_GPU__
#define __SOE_PERT_GPU__


#include <small_open_economy_perturbation.h>


#include <gpu_pf.h>

// there's a spot where you need the SS covariance
#include <kalman_utilities.h>


class SOE_SGU_GPU: public SOE_SGU
{
private:
    int nsamp;
    
    // necessary for calling particle filter
    double* RQ_L;
    double* HmatL;
    double* det_Hmat;

    double* X0;
    double* P0_L;

    gpu_model mod;
    gpu_options opts;
    
public:

    SOE_SGU_GPU();
    SOE_SGU_GPU(data Y_, short int tag_, int nsamp_);
    SOE_SGU_GPU(data Y_, int nsamp_): SOE_SGU_GPU(Y_, 0, nsamp_) {}


    ~SOE_SGU_GPU();
    
    void set_seed(int seed_){
	opts.seed = seed_;
    }

    double operator()(const double*, int bounded = 0);

    // copy constructor
    SOE_SGU_GPU(const SOE_SGU_GPU&);

    // swap function
    friend void swap(SOE_SGU_GPU&, SOE_SGU_GPU&);

    // assignment operator
    SOE_SGU_GPU& operator=(SOE_SGU_GPU other)
    {
	printf("\t\tAssignment operator 'SOE_SGU_GPU'\n");
	swap(*this, other);
	printf("\t\tDone.\n");
	return *this;
    }
};



#endif
