#ifndef __MPK_PERT__
#define __MPK_PERT__

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
#include <matrices.h>

class MPK_SGU: public perturbation
{
protected:

    int n_me = 4;
    
    double *LB, *UB, *theta_0;
    double *Rmat, *Qmat, *Hmat;
    
    std::vector<std::string> names, descriptions;
    
    void set_steady_state();
    void perform_autodiff();

public:

    int verbose;
    
    MPK_SGU(short int tag_);
    MPK_SGU(): MPK_SGU(1) {}
	
    ~MPK_SGU();
    
    void print_theta(void);
    void print_descriptions(void);

    void operator()(const double*, int bounded = 0);

    void decision(const double*, double*);
    
    // copy constructor
    MPK_SGU(const MPK_SGU&);

    // swap function
    friend void swap(MPK_SGU&, MPK_SGU&);
    
    // assignment operator
    MPK_SGU& operator=(MPK_SGU other)
    {
	printf("\t\t\tAssignment operator 'MPK_SGU' (%p receiving %p)\n\n", this, &other);
	swap(*this, other);
	printf("\t\t\tDone.\n\n");
	return *this;
    }
    
};

#endif
