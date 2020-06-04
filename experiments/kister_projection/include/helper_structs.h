#ifndef __HELPER_STRUCTS__
#define __HELPER_STRUCTS__

#include <algorithm> // for std::swap
#include <vector>    // for std::vector

struct Parameters
{
    // members
    double psi, gama, beta, ln_P_ss, phi_pi, phi_y, rho_R, epsilon, 
	theta, alpha, zeta, delta, ln_A_ss, rho_A, sigma_A, ln_sg_ss, 
	rho_sg, sigma_sg, sigma_xi;
    
    // default constructor
    Parameters();

    // standard constructor
    Parameters(const double* P_in);

    // copy constructor
    Parameters(const Parameters& that);

    // swap function
    friend void swap(Parameters& first, Parameters& second);

    // copy assignment operator
    Parameters& operator=(Parameters other);

    // casting to a vector
    operator std::vector<double>() const;
    
};

struct SteadyStates
{
    // members
    double ln_Ps_ss, ln_R_ss, ln_v_ss, ln_m_ss, ln_mc_ss, ln_rk_ss, ln_l_ss,
	ln_k_ss,  ln_i_ss, ln_c_ss, ln_V_ss, ln_y_ss, ln_x_ss,
	ln_P_ss, ln_A_ss, ln_sg_ss;

    // default constructor
    SteadyStates();

    // standard constructor
    SteadyStates(const Parameters& P);

    // copy constructor
    SteadyStates(const SteadyStates& that);

    // swap function
    friend void swap(SteadyStates&, SteadyStates&);

    // copy assignment operator
    SteadyStates& operator=(SteadyStates other);

    void print(void);
};

#endif
