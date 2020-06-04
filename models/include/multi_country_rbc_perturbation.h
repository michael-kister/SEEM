#ifndef __MCRBC_PERT__
#define __MCRBC_PERT__

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include <mkl.h>
#include <adolc/adolc.h>
#include <perturbation.h>
#include <data_class.h>

#include <miscellaneous.h>

/**
 * Multi-country real business cycle model.
 */
class MCRBC_SGU: public perturbation
{
protected:

    data Y;

    int num_country = 10;
    
    int n_me = 2 * num_country;
    
    double *LB, *UB, *theta_0;
    double *Rmat, *Qmat, *Hmat;
    
    std::vector<std::string> names, descriptions;
    
    void set_steady_state();
    void perform_autodiff();

public:

    int verbose;
    
    MCRBC_SGU();
    MCRBC_SGU(data Y_, short int tag_);
    MCRBC_SGU(data Y_): MCRBC_SGU(Y_, 1) {}
	
    
    ~MCRBC_SGU();
    
    void print_theta(void);
    void print_descriptions(void);

    void operator()(const double*);
    
    // copy constructor
    MCRBC_SGU(const MCRBC_SGU&);

    // swap function
    friend void swap(MCRBC_SGU&, MCRBC_SGU&);
    
    // assignment operator
    MCRBC_SGU& operator=(MCRBC_SGU other)
    {
	printf("\t\t\tAssignment operator 'MCRBC_SGU' (%p receiving %p)\n\n", this, &other);
	swap(*this, other);
	printf("\t\t\tDone.\n\n");
	return *this;
    }
    
};



#endif
