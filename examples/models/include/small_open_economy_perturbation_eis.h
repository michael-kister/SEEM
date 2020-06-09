#ifndef __SOE_PERT_EIS__
#define __SOE_PERT_EIS__

#include <small_open_economy_perturbation.h>
#include <eis_second_order.h>


class SOE_SGU_EIS: public SOE_SGU, public eis_model
{
private:

    int n_p = 4;
    int n_q = 2;
    
    // necessary for calling EIS filter
    mod_2nd_order mod;
    eis_opts opts;

    double* y_intercept_1;
    double* y_intercept_2;
    double* x_intercept_1;
    double* x_intercept_2;
    
protected:

    int invert_distribution(double*, double*, double*, double*, double*);
    
public: 

    SOE_SGU_EIS();
    SOE_SGU_EIS(data Y_, int nsamp_, short int tag_);
    SOE_SGU_EIS(data Y_, int nsamp_): SOE_SGU_EIS(Y_, nsamp_, 0) {}
	
    
    ~SOE_SGU_EIS();
    
    double operator()(const double*, int bounded = 0);
    
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
