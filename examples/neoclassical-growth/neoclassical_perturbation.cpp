class NeoclassicalPerturbation : public Perturbation {

    // Keep these here for convenience
    int num_s = 2; // number of states
    int num_c = 2; // number of controls
    int num_e = 1; // number of shocks
    int num_p = 7; // number of parameters
    int deg   = 2; // degree of approximation
    
public:

    // constructor
    Neoclassical(double *p_) : Perturbation(num_s,num_c,num_e,num_p,deg)
    {
	for (int i = 0; i < num_param; ++i)
	    parameters[i] = p_[i];
    }

    // This is what makes the model unique
    void PerformAutomaticDifferentiation(void) {
	
	// obtain steady states
	double Phi = pow(((1/beta)-1+delta)/alpha,1/(alpha-1));
	double Psi = (thxta/(1-thxta))*(1-alpha)*pow(Phi,alpha);

	double l_ss = Psi/(Psi-(delta*Phi)+pow(Phi,alpha));
	double c_ss = (1-l_ss)*Psi;
	double k_ss = Phi*l_ss;
	double z_ss = 0;
  
	// store them all together
	xss_xss_yss_yss[0] = k_ss; // current states
	xss_xss_yss_yss[1] = z_ss;
	xss_xss_yss_yss[2] = k_ss; // future states
	xss_xss_yss_yss[3] = z_ss;
	xss_xss_yss_yss[4] = l_ss; // current policy
	xss_xss_yss_yss[5] = c_ss;
	xss_xss_yss_yss[6] = l_ss; // future policy
	xss_xss_yss_yss[7] = c_ss;
  
	/***********************************************************************
	 * AUTOMATIC DIFFERENTIATION
	 **********************************************************************/

	double pxxy;
	
	// Step 1 ................................................ turn on trace
	trace_on(0);

	// Step 2 .............................................. load parameters
	for (int i = 0; i < num_param; ++i) {
	    param_loc[i] = mkparam_idx(parameters[i]);
	}
	
	// Step 3 ................................ make shortcuts for parameters
        #define BETA_X getparam(param_loc[0])
        #define TAU__X getparam(param_loc[1])
        #define THETAX getparam(param_loc[2])
        #define ALPHAX getparam(param_loc[3])
        #define DELTAX getparam(param_loc[4])
        #define RHO__X getparam(param_loc[5])
        #define SIGMAX getparam(param_loc[6])
  
	// Step 4 ............................... register independent variables
	X_ad[0] <<= k_ss; // current states
	X_ad[1] <<= z_ss;
	X_ad[2] <<= k_ss; // future states
	X_ad[3] <<= z_ss;
	X_ad[4] <<= l_ss; // current policy
	X_ad[5] <<= c_ss;
	X_ad[6] <<= l_ss; // future policy
	X_ad[7] <<= c_ss;

	// Step 5 ............................... make shortcuts for ind. values
	adouble k_t    = X_ad[0]; // current states
	adouble z_t    = X_ad[1];
	adouble k_tp1  = X_ad[2]; // future states
	adouble z_tp1  = X_ad[3];
	adouble l_t    = X_ad[4]; // current policy
	adouble c_t    = X_ad[5];
	adouble l_tp1  = X_ad[6]; // future policy
	adouble c_tp1  = X_ad[7];
  
	// Step 6 .............................. construct some helper variables
	adouble eq1_lhs   = pow(pow(c_t  ,THETAX)*pow(1-l_t  ,1-THETAX),1-TAU__X)*THETAX/c_t  ;
	adouble eq1_rhs_1 = pow(pow(c_tp1,THETAX)*pow(1-l_tp1,1-THETAX),1-TAU__X)*THETAX/c_tp1;
	adouble eq1_rhs_2 = (1-DELTAX) + ALPHAX*exp(z_tp1)*pow(k_tp1/l_tp1,ALPHAX-1);

	adouble eq2_lhs = (c_t*(1-THETAX))/(THETAX*(1-l_t));
	adouble eq2_rhs = (1-ALPHAX)*exp(z_t)*pow(k_t/l_t,ALPHAX);
  
	adouble eq3_lhs = c_t + k_tp1;
	adouble eq3_rhs = (1-DELTAX)*k_t + exp(z_t)*pow(k_t,ALPHAX)*pow(l_t,1-ALPHAX);
  
	adouble eq4_lhs = z_tp1;
	adouble eq4_rhs = RHO__X*z_t;
  
	// Step 6 ............................... write out our target equations
	Y_ad[0] = eq1_lhs - BETA_X * eq1_rhs_1 * eq1_rhs_2;
	Y_ad[1] = eq2_lhs - eq2_rhs;
	Y_ad[2] = eq3_lhs - eq3_rhs;
	Y_ad[3] = eq4_lhs - eq4_rhs;
  
	// Step 7 ............................. store evaluated for use later???
	for (int i = 0; i < num_variable; ++i)
	    Y_ad[i] >>= pxxy;//pxy[i];

	// Step 8 ............................................... turn off trace
	trace_off();
    
	// Step 9 ............................ perform automatic differentiation
	tensor_eval(0, num_variable, 2*num_variable, 2,
		    2*num_variable, xss_xss_yss_yss, adolc_tensor, S);

	// Step 10 ............................... un-define parameter shortcuts
        #undef BETA_X
        #undef TAU__X
        #undef THETAX
        #undef ALPHAX
        #undef DELTAX
        #undef RHO__X
        #undef SIGMAX

	
    }
    
};
