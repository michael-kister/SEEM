#include "small_open_economy_perturbation.h"

//========================================//
//                                        //
//          MODEL SPECIFICATION           //
//                                        //
//========================================//

SOE_SGU::SOE_SGU(data Y_, short int tag_):
    perturbation(6, 9, 4, 19, 2, tag_), Y(Y_)
{
    printf("\t\t\tConstructing 'SOE_SGU' (%p)\n\n", this);
    
    LB = new double [npar];
    UB = new double [npar];
    theta_0 = new double [npar];

    Hmat = new double[n_me * n_me]();
    Qmat = new double[neps * neps]();
    Rmat = new double[nx * neps]();

    Rmat[neps*(2+0)+0] = 1.0;
    Rmat[neps*(2+1)+1] = 1.0;
    Rmat[neps*(2+2)+2] = 1.0;
    Rmat[neps*(2+3)+3] = 1.0;
    
    names.resize(npar);
    descriptions.resize(npar);

    names[0]  = "γ  "; descriptions[0]  = "Intertemporal elasticity of substitution";
    names[1]  = "ω  "; descriptions[1]  = "Exponent of labor in utility function";
    names[2]  = "ψ  "; descriptions[2]  = "Negative elasticity of discount factor";
    names[3]  = "α  "; descriptions[3]  = "Capital elasticity of production function";
    names[4]  = "φ  "; descriptions[4]  = "Parameter in adjustment cost function";
    names[5]  = "r* "; descriptions[5]  = "World interest rate";
    names[6]  = "δ  "; descriptions[6]  = "Depreciation rate";
    names[7]  = "ρ_A"; descriptions[7]  = "Serial correlation of productivity shock";
    names[8]  = "σ_A"; descriptions[8]  = "StDev of tech/productivity shock";
    names[9]  = "ρ_r"; descriptions[9]  = "Serial correlation of interest rate shock";
    names[10] = "σ_r"; descriptions[10] = "StDev of interest rate shock";
    names[11] = "ρ_v"; descriptions[11] = "Serial correlation of technology shock";
    names[12] = "σ_v"; descriptions[12] = "StDev of investment/technology shock";
    names[13] = "ρ_φ"; descriptions[13] = "Serial correlation of preference shock";
    names[14] = "σ_φ"; descriptions[14] = "StDev of preference shock";
    names[15] = "σ_x"; descriptions[15] = "StDev of (GDP) measurement error";
    names[16] = "σ_c"; descriptions[16] = "StDev of (consumption) measurement error";
    names[17] = "σ_i"; descriptions[17] = "StDev of (investment) measurement error ";
    names[18] = "σ_n"; descriptions[18] = "StDev of (hours) measurement error";

    /*
    LB[0]  = 2.0000;  theta[0]  = 2.4929;  UB[0]  = 7.0000; // intertemp elast of subs
    LB[1]  = 1.3200;  theta[1]  = 1.3338;  UB[1]  = 1.3450; // exp of labor in util fun
    LB[2]  = 0.0800;  theta[2]  = 0.1100;  UB[2]  = 0.1250; // Neg elast of d-factor wrt arg
    LB[3]  = 0.2330;  theta[3]  = 0.2346;  UB[3]  = 0.2370; // Cap elast of prod ftn
    LB[4]  = 0.0330;  theta[4]  = 0.0387;  UB[4]  = 0.0450; // Par. of adj. cost ftn
    LB[5]  = 0.0070;  theta[5]  = 0.0076;  UB[5]  = 0.0100; // world interest rate
    LB[6]  = 0.0195;  theta[6]  = 0.0200;  UB[6]  = 0.0206; // Depreciation rate
    LB[7]  = 0.7500;  theta[7]  = 0.8195;  UB[7]  = 0.8700; // Ser corr of prod shock
    LB[8]  = 0.0015;  theta[8]  = 0.0019;  UB[8]  = 0.0023; // std of technology shock
    LB[9]  = -0.600;  theta[9]  = 0.7960;  UB[9]  = 0.9000; // Ser corr of int rate shock
    LB[10] = 0.0003;  theta[10] = 0.0022;  UB[10] = 0.0025; // std of interest rate shock
    LB[11] = 0.8000;  theta[11] = 0.8707;  UB[11] = 0.9200; // Ser corr of invest tech shock
    LB[12] = .00094;  theta[12] = 0.0010;  UB[12] = 0.0016; // std of investment shock
    LB[13] = 0.8200;  theta[13] = 0.8634;  UB[13] = 0.9000; // Ser corr of pref. shock
    LB[14] = 0.0024;  theta[14] = 0.0031;  UB[14] = 0.0038; // std of pref. shock
    LB[15] = 0.0028;  theta[15] = 0.0038;  UB[15] = 0.0045; // m.e.
    LB[16] = 0.0055;  theta[16] = 0.0065;  UB[16] = 0.0075; // m.e. 
    LB[17] = 0.0004;  theta[17] = 0.0046;  UB[17] = 0.0067; // m.e. 
    LB[18] = 0.0045;  theta[18] = 0.0058;  UB[18] = 0.0070; // m.e.
    */

    LB[0]  = 1.5000;  theta[0]  = 3.9687;  UB[0]  = 7.0000; // intertemp elast of subs
    LB[1]  = 1.3200;  theta[1]  = 1.3387;  UB[1]  = 1.345 ; // exp of labor in util fun
    LB[2]  = 0.0600;  theta[2]  = 0.0871;  UB[2]  = 0.125 ; // Neg elast of d-factor wrt arg
    LB[3]  = 0.2330;  theta[3]  = 0.2354;  UB[3]  = 0.240 ; // Cap elast of prod ftn
    LB[4]  = 0.0330;  theta[4]  = 0.0386;  UB[4]  = 0.045 ; // Par. of adj. cost ftn
    LB[5]  = 0.0060;  theta[5]  = 0.0075;  UB[5]  = 0.008 ; // world interest rate
    LB[6]  = 0.0195;  theta[6]  = 0.0201;  UB[6]  = 0.0215; // Depreciation rate
    LB[7]  = 0.7000;  theta[7]  = 0.8210;  UB[7]  = 0.87  ; // Ser corr of prod shock
    LB[8]  = 0.0015;  theta[8]  = 0.0020;  UB[8]  = 0.0023; // std of technology shock
    LB[9]  = -0.600;  theta[9]  = -.2091;  UB[9]  = 0.50  ; // Ser corr of int rate shock
    LB[10] = 0.0001;  theta[10] = 0.0004;  UB[10] = 0.0025; // std of interest rate shock
    LB[11] = 0.8500;  theta[11] = 0.8800;  UB[11] = 0.91  ; // Ser corr of invest tech shock
    LB[12] = .00075;  theta[12] = 0.0001;  UB[12] = 0.0016; // std of investment shock
    LB[13] = 0.8200;  theta[13] = 0.8660;  UB[13] = 0.88  ; // Ser corr of pref. shock
    LB[14] = 0.0024;  theta[14] = 0.0032;  UB[14] = 0.0038; // std of pref. shock
    LB[15] = 0.0028;  theta[15] = 0.0036;  UB[15] = 0.0045; // m.e.
    LB[16] = 0.0055;  theta[16] = 0.0065;  UB[16] = 0.0075; // m.e. 
    LB[17] = 0.0005;  theta[17] = 0.0046;  UB[17] = 0.0067; // m.e. 
    LB[18] = 0.0045;  theta[18] = 0.0058;  UB[18] = 0.007 ; // m.e.

    
    perform_autodiff(); 
    verbose = 0;

    printf("\t\t\tDone.\n\n");
}

//    perturbation(that.nx, that.ny, that.neps, that.npar, that.deg), Y(that.Y)
// copy constructor
SOE_SGU::SOE_SGU(const SOE_SGU& that):
    perturbation(that), Y(that.Y)
{
    printf("\t\t\tCopy constructor for 'SOE_SGU' (%p receiving %p)\n\n", this, &that);

    LB = new double [npar];
    UB = new double [npar];
    theta_0 = new double [npar];

    memcpy(LB, that.LB, npar*sizeof(double));
    memcpy(UB, that.UB, npar*sizeof(double));

    Rmat = new double [nx * neps]();
    Qmat = new double [neps * neps]();
    Hmat = new double [n_me * n_me]();
    
    memcpy(Rmat, that.Rmat, nx*neps*sizeof(double));
    memcpy(Qmat, that.Qmat, neps*neps*sizeof(double));
    memcpy(Hmat, that.Hmat, n_me*n_me*sizeof(double));
    
    printf("\t\t\tDone.\n\n");
}

// swap function
void swap(SOE_SGU& first, SOE_SGU& second)
{
    printf("\t\t\tSwapping 'SOE_SGU' (%p <--> %p)\n\n", &first, &second);

    swap(static_cast<SGU_BASE&>(first), static_cast<SGU_BASE&>(second));
    
    swap(first.Y, second.Y);

    std::swap(first.nx, second.nx);
    std::swap(first.ny, second.ny);
    std::swap(first.neps, second.neps);
    std::swap(first.npar, second.npar);

    std::swap(first.Rmat, second.Rmat);
    std::swap(first.Qmat, second.Qmat);
    std::swap(first.Hmat, second.Hmat);

    first.names.swap(second.names);
    first.descriptions.swap(second.descriptions);
    
    for (int i = 0; i < first.npar; i++) {
	std::swap(first.LB[i],      second.LB[i]);
	std::swap(first.UB[i],      second.UB[i]);
	std::swap(first.theta_0[i], second.theta_0[i]);
    }
    
    printf("\t\t\tDone.\n\n");
}


SOE_SGU::~SOE_SGU(void)
{
    printf("\t\t\t-Destructing 'SOE_SGU' (%p)... ", this);

    delete[] LB;
    delete[] UB;
    delete[] theta_0;

    delete[] Rmat;
    delete[] Hmat;
    delete[] Qmat;

    printf("Done.\n\n");
}

void SOE_SGU::print_theta(void)
{
    printf("------------------------------------------------\n");
    printf("Parameter Values:\n\n");
    for (int i = 0; i < 6; i++) {
	for (int j = 0; j < 3; j++)
	    printf(" %3s = %+08.5f ", names[3*i+j].c_str(), theta[3*i+j]);
	printf("\n");
    }
    printf(" %3s = %+08.5f \n", names[18].c_str(), theta[18]);
    printf("------------------------------------------------\n");
}


void SOE_SGU::print_descriptions(void)
{
    printf("----------------------------------------------------\n");
    printf("Parameter Descriptions:\n\n");
    for (int i = 0; i < npar; i++)
	    printf(" %3s : %-50s\n", names[i].c_str(), descriptions[i].c_str());
    printf("----------------------------------------------------\n");
}


void SOE_SGU::operator()(const double* theta_unbounded, int bounded)
{
    double theta_bounded[npar];

    if (bounded == 0) {
	for (int i = 0; i < npar; i++) {
	    theta_bounded[i] = inverse_logit(theta_unbounded[i]);
	    theta_bounded[i] = change_bounds(theta_bounded[i], 0.0, 1.0, LB[i], UB[i]);
	    //printf("theta[%d] = %f;\n", i, theta_bounded[i]);
	}
    } else if (bounded == 1) {
	for (int i = 0; i < npar; i++) {
	    theta_bounded[i] = theta_unbounded[i];
	}
    }
    
    load_parameters(theta_bounded, 0);
    
    if (verbose)
	print_theta();

    Hmat [n_me*0 + 0] = theta[15] * theta[15];
    Hmat [n_me*1 + 1] = theta[16] * theta[16];
    Hmat [n_me*2 + 2] = theta[17] * theta[17];
    Hmat [n_me*3 + 3] = theta[18] * theta[18];

    Qmat [neps*0 + 0] = theta[8]  * theta[8];
    Qmat [neps*1 + 1] = theta[10] * theta[10];
    Qmat [neps*2 + 2] = theta[12] * theta[12];
    Qmat [neps*3 + 3] = theta[14] * theta[14];
    
    
    // and then we manually shift the steady state of the observables
    // to account for the way in which we standardize the data.
    //
    // I don't like this, but I'm not sure what choice we have...

    g_lev_1[1] -= y_ss[0];  g_lev_2[1] -= y_ss[0];  y_ss[1] -= y_ss[0];
    g_lev_1[2] -= y_ss[0];  g_lev_2[2] -= y_ss[0];  y_ss[2] -= y_ss[0];
    g_lev_1[0] -= y_ss[0];  g_lev_2[0] -= y_ss[0];  y_ss[0] -= y_ss[0];

    g_lev_1[3] -= y_ss[3];  g_lev_2[3] -= y_ss[3];  y_ss[3] -= y_ss[3];

}


void SOE_SGU::set_steady_state(void)
{
    const double gamma = theta[0];
    const double omega = theta[1];
    const double psi   = theta[2];
    const double alpha = theta[3];
    const double phi   = theta[4];
    const double rstar = theta[5];
    const double delta = theta[6];
    const double rho   = theta[7];
    const double sig_e = theta[8];  // this is the big sigma (absorbs sigma_a)
    const double rho_r = theta[9];
    const double sig_r = theta[10];
    const double rho_v = theta[11];
    const double sig_v = theta[12];
    const double rho_p = theta[13];
    const double sig_p = theta[14];
    
    // helper term
    double T = pow( alpha/(rstar+delta), 1.0/(1.0-alpha) );
    double H = pow( (1.0-alpha)*pow( T, alpha ), 1.0/(omega-1.0) );
    double K = T * H;
    double Y = pow( T, alpha ) * H;
    double INV = delta * T * H;
    double C = pow( 1.0+rstar, 1.0/psi ) - 1.0 + pow( H, omega )/omega;
    double D = (1.0/rstar) * (Y - C - INV);
    double LA = pow( C - pow( H, omega )/omega, -1.0*gamma );
    double TB = Y - C - INV;
    double TBY = TB / Y;
    double CAY = (-1.0*rstar*D + TB) / Y;

    // exogenous controls (yy c invest h tby cay tb la k1)
    y_ss[0] = log( Y );
    y_ss[1] = log( C );
    y_ss[2] = log( INV );
    y_ss[3] = log( H );
    y_ss[4] = TBY;
    y_ss[5] = CAY;
    y_ss[6] = TB;
    y_ss[7] = log( LA );
    y_ss[8] = log( K );

    // endogenous states (k d a r v phi)
    x_ss[0] = log( K );
    x_ss[1] = log( D );
    x_ss[2] = 0.0;//epsilon;
    x_ss[3] = log( rstar );
    x_ss[4] = 0.0;//epsilon;
    x_ss[5] = 0.0;//epsilon;

    // shock matrix ?
    sigma = sig_e;

    eta[neps*2 + 0] = 1.0;
    eta[neps*3 + 1] = sig_r / sig_e;
    eta[neps*4 + 2] = sig_v / sig_e;
    eta[neps*5 + 3] = sig_p / sig_e;
    
    int i_X = 0;
    for (int j = 0; j < 2; j++)
	for (int i = 0; i < nx; i++)
	    X_ss[i_X++] = x_ss[i];
	
    for (int j = 0; j < 2; j++)
	for (int i = 0; i < ny; i++)
	    X_ss[i_X++] = y_ss[i];
    
}

void SOE_SGU::perform_autodiff(void)
{
    //------------------------------------------------------------
    // 0.) Obtain the steady state values
    int N = nx + nx + ny + ny;
    
    set_steady_state();
    
    double* pxy  = new double [nx+ny];
    
    //printf("0: obtained steady state.\n");
    //------------------------------------------------------------
    // 1.) define the tag; start the trace; set up the variables

    printf("T:\t\t\t\tTurning on trace %d\n\n", tag);
    trace_on(tag);

    adouble* X = new adouble [N];
    adouble* Y = new adouble [nx+ny];
    
    //printf("1: set up tag.\n");
    //------------------------------------------------------------
    // 2.) some work with the parameters

    locint* para = new locint [npar];
    for (int i = 0; i < npar; i++)
	para[i] = mkparam_idx(theta[i]);
    
    // you must manually name all the variables here!
    #define GAMMX getparam(para[0])  // intertemporal elasticity of substitution
    #define OMEGX getparam(para[1])  // exponent of labor in utility function
    #define PSI_X getparam(para[2])  // Minus elasticity of discount factor wrt its argument
    #define ALPHX getparam(para[3])  // Capital elasticity of the production function
    #define PHI_X getparam(para[4])  // Parameter of adjustment cost function
    #define RSTAX getparam(para[5])  // world interest rate
    #define DELTX getparam(para[6])  // Depreciation rate
    #define RHOAX getparam(para[7])  // Serial correlation of productivity shock
    // this is the big sigma (absorbs sigma_a)
    #define SIGEX getparam(para[8])  // std of technology shock
    #define RHORX getparam(para[9])  // Serial correlation of interest rate shock
    #define SIGRX getparam(para[10]) // std of interest rate shock
    #define RHOVX getparam(para[11]) // Serial correlation of investment specific tech shock
    #define SIGVX getparam(para[12]) // std of investment shock
    #define RHOPX getparam(para[13]) // Serial correlation of preference shock
    #define SIGPX getparam(para[14]) // std of preference shock
    // these are measurement errors... they won't show up below
    #define MEEYX getparam(para[15]) // m.e. of preference shock
    #define MEECX getparam(para[16]) // m.e. of preference shock
    #define MEEIX getparam(para[17]) // m.e. of preference shock
    #define MEEHX getparam(para[18]) // m.e. of preference shock
    
    //printf("2: set up dynamic parameters.\n");
    //------------------------------------------------------------
    // 3.) transfer steady state values to the tape
    
    for (int i = 0; i < N; i++)
	X[i] <<= X_ss[i];
    

    //------------------------------------------------------------
    // 4.) fix variable names (I know it's a pain, but the alter-
    //     native is just remembering where all of them are...)

    adouble k_t = exp( X[0] );
    adouble d_t = exp( X[1] );
    adouble a_t = exp( X[2] );
    adouble r_t = exp( X[3] );
    adouble v_t = exp( X[4] );
    adouble p_t = exp( X[5] );

    adouble k_tt = exp( X[6] );
    adouble d_tt = exp( X[7] );
    adouble a_tt = exp( X[8] );
    adouble r_tt = exp( X[9] );
    adouble v_tt = exp( X[10] );
    adouble p_tt = exp( X[11] );

    adouble y_t   = exp( X[12] );
    adouble c_t   = exp( X[13] );
    adouble i_t   = exp( X[14] );
    adouble h_t   = exp( X[15] );
    adouble tby_t = exp( X[16] );
    adouble cay_t = exp( X[17] );
    adouble tb_t  = exp( X[18] );
    adouble la_t  = exp( X[19] );
    adouble k1_t  = exp( X[20] );

    adouble y_tt   = exp( X[21] );
    adouble c_tt   = exp( X[22] );
    adouble i_tt   = exp( X[23] );
    adouble h_tt   = exp( X[24] );
    adouble tby_tt = exp( X[25] );
    adouble cay_tt = exp( X[26] );
    adouble tb_tt  = exp( X[27] );
    adouble la_tt  = exp( X[28] );
    adouble k1_tt  = exp( X[29] );

    // some useful elements for the following equations
    adouble phoo = p_t * pow( h_t, OMEGX ) / OMEGX;
    
    adouble beta_disc = pow( 1.0 + c_t - phoo, -1.0*PSI_X );
    
    adouble du_dc = pow( c_t - phoo, -1.0*GAMMX );
    adouble du_dh = -1.0 * p_t * pow( h_t, OMEGX - 1.0 ) * du_dc;

    adouble do_dh = a_t * pow( h_t, -1.0*ALPHX ) * pow( k_t, ALPHX ) * (1 - ALPHX);
    adouble dopdk = a_tt * pow( h_tt, 1.0-ALPHX ) * pow( k_tt, ALPHX-1.0 ) * (ALPHX);
    
    //------------------------------------------------------------
    // 5.) define your DSGE
    
    // Eq. 1: foreign debt 
    Y[0] = (1.0 + r_t) * d_t - log(tb_t) - d_tt;

    // Eq. 2: trade balance
    Y[1] = -1.0*log(tb_t) + y_t - c_t - i_t - PHI_X*(k_tt - k_t)*(k_tt - k_t)/2.0;

    // Eq. 3: production function
    Y[2] = -1.0*y_t + a_t * pow( k_t, ALPHX ) * pow( h_t, 1.0-ALPHX );

    // Eq. 4: capital stock
    Y[3] = (-1.0/v_t)*i_t + k_tt - (1.0-DELTX) * k_t;

    // Eq. 5: lagrange condition?
    Y[4] = -1.0*la_t + beta_disc * (1.0+r_tt) * la_tt;

    // Eq. 6: lagrange condition?
    Y[5] = -1.0*la_t + du_dc;

    // Eq. 7: lagrange condition?
    Y[6] = -1.0*du_dh - la_t * do_dh;

    // Eq. 8: lagrange condition?
    Y[7] = -1.0*la_t*(v_t+PHI_X*(k_tt-k_t)) + beta_disc * la_tt*(dopdk + (1.0-DELTX)*v_tt + PHI_X*(k1_tt-k1_t));

    // Eq. 9: extra forward lag
    Y[8] = -1.0*k1_t + k_tt;

    // Eq. 10: structural productivity
    Y[9] = -1.0*log(a_tt) + RHOAX * log(a_t);

    // Eq. 11: ratio of trade balance to GDP
    Y[10] = -1.0*log(tby_t) + log(tb_t) / y_t;

    // Eq. 12: current account over GDP
    Y[11] = -1.0*log(cay_t) - (RSTAX * d_t / y_t) + log(tby_t);

    // Eq. 13: structural interest rates
    Y[12] = -1.0*log(r_tt) + (1.0-RHORX)*log(RSTAX) + RHORX*log(r_t);

    // Eq. 14: structural investment-specific productivity shock
    Y[13] = -1.0*log(v_tt) + RHOVX * log(v_t);
    
    // Eq. 15: structural preference shock
    Y[14] = -1.0*log(p_tt) + RHOPX * log(p_t);
    
        
    for (int i = 0; i < nx+ny; i++)
	Y[i] >>= pxy[i];
    //------------------------------------------------------------
    // 6.) clean up and return the tape number
    
    trace_off();
    printf("T:\t\t\t\tTrace off.\n\n");

    delete[] pxy;
    delete[] X;
    delete[] Y;
    delete[] para;
    
    //printf("6: turn trace off.\n");

}
