#ifndef __SOE_PERT__
#define __SOE_PERT__

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
 * Small open economy model, with 6 states, 9 controls, 4 shocks, 19 parameters, and 4 observables.
 */
class SOE_SGU: public perturbation
{
protected:

    data Y;
    int n_me = 4;
    
    double *LB, *UB, *theta_0;
    double *Rmat, *Qmat, *Hmat;
    
    std::vector<std::string> names, descriptions;
    
    void set_steady_state();
    void perform_autodiff();

public:

    int verbose;
    
    SOE_SGU();
    SOE_SGU(data Y_, short int tag_);
    SOE_SGU(data Y_): SOE_SGU(Y_, 1) {}
	
    
    ~SOE_SGU();
    
    void print_theta(void);
    void print_descriptions(void);

    void operator()(const double*, int bounded = 0);
    
    // copy constructor
    SOE_SGU(const SOE_SGU&);

    // swap function
    friend void swap(SOE_SGU&, SOE_SGU&);
    
    // assignment operator
    SOE_SGU& operator=(SOE_SGU other)
    {
	printf("\t\t\tAssignment operator 'SOE_SGU' (%p receiving %p)\n\n", this, &other);
	swap(*this, other);
	printf("\t\t\tDone.\n\n");
	return *this;
    }
    
};



#endif

/*
  Notes on the inversion:
 
  Note that we're looking for the absolute value of the determinant of:

  [ d d_t1  d d_t1 ]   [ d d_t1         ]
  [ ------  ------ ]   [ ------    0.0  ]
  [ d d_t   d k_t  ]   [ d d_t          ]
  [                ] = [                ] ,
  [ d k_t1  d k_t1 ]   [ d k_t1  d k_t1 ]
  [ ------  ------ ]   [ ------  ------ ]
  [ d d_t   d k_t  ]   [ d d_t   d k_t  ]

  which is just

  | d d_t1   d k_t1 |
  | ------ * ------ | .
  | d d_t    d k_t  |
      
  We have that

  d_t1 = (-b + sqrt(b*b - 4*a*c)) / (2*a) ,

  where c is the only one that depends on d_t. Therefore we have that

  d d_t1   d d_t1    d c            1
  ------ = ------ * ----- = ----------------- * (-1) .
  d d_t     d c     d d_t   sqrt(b*b - 4*a*c)

  Next, we have that

  k_t1 = (-b + sqrt(b*b - 4*a*c)) / (2*a) ,

  where, again, c is the only one who depends on k_t. Subsequently, we have that

  d k_t1   d k_t1    d c            1
  ------ = ------ * ----- = ----------------- * (-1) .
  d k_t     d c     d k_t   sqrt(b*b - 4*a*c)

  Therefore, the absolute value of the jacobian is given by one over the square
  root of the product of the respective (b*b - 4*a*c).
*/
