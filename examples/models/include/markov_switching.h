#ifndef __MARKOV_SWITCHING__
#define __MARKOV_SWITCHING__

#include <parameters.h>


template <typename T, typename U>
class markov_switching
{
private:
    int n_s;
    double* Pmat;
    
    T* models;
    U* mod_structs; // because this gets passed into the likelihood evaluation

    double** thetas;

    void parameter_parsing(void)
    {
	// something needs to take the state and map it to the different models
    }
    
public:
    int operator()(double*)
    {
	// take in the parameters
	// unbround them...
	// if some boolean is set, then you want to include the priors and the jacobian
    }

    markov_switching(){};
    markov_switching(int n_s, int n_par){

	models = new T*[n_s];

	thetas = new double*[n_s];
	for (int i = 0; i < n_s; i++)
	    thetas[i] = new double[n_par];
    }

};





#endif
