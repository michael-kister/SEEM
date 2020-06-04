#include "kister_perturbation.h"

// helper macros
#define LN_UTILITY(ln_ct, ln_lt) (alpha*ln_ct + (1.0-alpha)*log(1.0-exp(ln_lt)))
#define LN_PI_STAR(ln_Pt) ((log(1.0-thetx*exp((epsilon-1.0)*ln_Pt))-log(1.0-thetx))/(1.0-epsilon))

#define ALN_UTILITY(ln_ct, ln_lt) (ALPHX*ln_ct + (1.0-ALPHX)*log(1.0-exp(ln_lt)))
#define ALN_PI_STAR(ln_Pt) ((log(1.0-THETX*exp((EPS_X-1.0)*ln_Pt))-log(1.0-THETX))/(1.0-EPS_X))

//========================================//
//                                        //
//         QUADRATIC DECISION RULE        //
//                                        //
//========================================//

void MPK_SGU::decision(const double* states, double* controls)
{
    // this is just manual matrix multiplication
    
    // loop over controls
    for (int i = 0; i < ny; i++) {

	// set the control to zero to start out
	controls[i] = 0.0;

	// loop over the states
	for (int j = 0; j < nx; j++) {
	    
	    controls[i] += gx[nx*i+j]*states[j];

	    // loop over the states again for the kronecker product
	    for (int k = 0; k < nx; k++) {

		controls[i] += 0.5*gxx[nx*nx*i+nx*j+k]*states[j]*states[k];

	    }
	}
    }
}

//========================================//
//                                        //
//          MODEL SPECIFICATION           //
//                                        //
//========================================//

MPK_SGU::MPK_SGU(short int tag_): perturbation(6, 5, 3, 19, 2, tag_)
{
    printf("\t\t\tConstructing 'MPK_SGU' (%p)\n\n", this);
    
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
    
    names[0]   = "ψ  "; descriptions[0]   = "EZ parameter";
    names[1]   = "γ  "; descriptions[1]   = "EZ parameter";
    names[2]   = "β  "; descriptions[2]   = "Intertemporal elasticity of substitution";
    names[3]   = "Πss"; descriptions[3]   = "Inflation target";
    names[4]   = "φ_π"; descriptions[4]   = "Taylor rule: on inflation";
    names[5]   = "φ_y"; descriptions[5]   = "Taylor rule: on output";
    names[6]   = "ρ_R"; descriptions[6]   = "Autoregressive in Taylor rule";
    names[7]   = "ε  "; descriptions[7]   = "Firm aggregation parameter";
    names[8]   = "θ  "; descriptions[8]   = "Price stickiness";
    names[9]   = "α  "; descriptions[9]   = "Consumption elasticity in utility function";
    names[10]  = "ζ  "; descriptions[10]  = "Capital elasticity of production function";
    names[11]  = "δ  "; descriptions[11]  = "Depreciation rate";
    names[12]  = "Ass"; descriptions[12]  = "Tech steady state";
    names[13]  = "ρ_A"; descriptions[13]  = "Serial correlation of productivity shock";
    names[14]  = "σ_A"; descriptions[14]  = "StDev of tech/productivity shock";
    names[15]  = "gss"; descriptions[15]  = "Gov't steady state";
    names[16]  = "ρ_g"; descriptions[16]  = "Serial correlation of gov't shock";
    names[17]  = "σ_g"; descriptions[17]  = "StDev of gov't shock";
    names[18]  = "σ_ξ"; descriptions[18]  = "StDev of monetary shock";


    LB[0]  = 1.0   ;   theta[0]  = 2.0   ;   UB[0]  = 3.0   ;
    LB[1]  = 4.0   ;   theta[1]  = 5.0   ;   UB[1]  = 6.0   ;
    LB[2]  = 0.9   ;   theta[2]  = 0.991 ;   UB[2]  = 0.999 ;
    LB[3]  = 0.004 ;   theta[3]  = 0.005 ;   UB[3]  = 0.006 ;
    LB[4]  = 1.4   ;   theta[4]  = 1.5   ;   UB[4]  = 1.6   ;
    LB[5]  = 0.1   ;   theta[5]  = 0.25  ;   UB[5]  = 0.3   ;
    LB[6]  = 0.01  ;   theta[6]  = 0.1   ;   UB[6]  = 0.2   ;
    LB[7]  = 5.0   ;   theta[7]  = 6.0   ;   UB[7]  = 7.0   ;
    LB[8]  = 0.5   ;   theta[8]  = 0.75  ;   UB[8]  = 0.9   ;
    LB[9]  = 0.3   ;   theta[9]  = 0.357 ;   UB[9]  = 0.4   ;
    LB[10] = 0.2   ;   theta[10] = 0.3   ;   UB[10] = 0.4   ;
    LB[11] = 0.01  ;   theta[11] = 0.0196;   UB[11] = 0.03  ;
    LB[12] =-0.01  ;   theta[12] = 0.0   ;   UB[12] = 0.01  ;
    LB[13] = 0.8   ;   theta[13] = 0.9   ;   UB[13] = 0.99  ;
    LB[14] = 0.002 ;   theta[14] = 0.0025;   UB[14] = 0.003 ;
    LB[15] =-1.7   ;   theta[15] =-1.6094;   UB[15] =-1.5   ;
    LB[16] = 0.7   ;   theta[16] = 0.8   ;   UB[16] = 0.9   ;
    LB[17] = 0.002 ;   theta[17] = 0.0025;   UB[17] = 0.003 ;
    LB[18] = 0.002 ;   theta[18] = 0.0025;   UB[18] = 0.003 ;
    
    perform_autodiff(); 
    verbose = 0;
    
    printf("\t\t\tDone.\n\n");
}

//    perturbation(that.nx, that.ny, that.neps, that.npar, that.deg), Y(that.Y)
// copy constructor
MPK_SGU::MPK_SGU(const MPK_SGU& that): perturbation(that)
{
    printf("\t\t\tCopy constructor for 'MPK_SGU' (%p receiving %p)\n\n", this, &that);

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
void swap(MPK_SGU& first, MPK_SGU& second)
{
    printf("\t\t\tSwapping 'MPK_SGU' (%p <--> %p)\n\n", &first, &second);

    swap(static_cast<SGU_BASE&>(first), static_cast<SGU_BASE&>(second));
    
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


MPK_SGU::~MPK_SGU(void)
{
    printf("\t\t\t-Destructing 'MPK_SGU' (%p)... ", this);

    delete[] LB;
    delete[] UB;
    delete[] theta_0;

    delete[] Rmat;
    delete[] Hmat;
    delete[] Qmat;

    printf("Done.\n\n");
}

void MPK_SGU::print_theta(void)
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


void MPK_SGU::print_descriptions(void)
{
    printf("----------------------------------------------------\n");
    printf("Parameter Descriptions:\n\n");
    for (int i = 0; i < npar; i++)
	    printf(" %3s : %-50s\n", names[i].c_str(), descriptions[i].c_str());
    printf("----------------------------------------------------\n");
}


void MPK_SGU::operator()(const double* theta_unbounded, int bounded)
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


void MPK_SGU::set_steady_state(void)
{
    const double psi      = theta[0];
    const double gama     = theta[1];
    const double beta     = theta[2];
    const double ln_P_ss  = theta[3];
    const double phi_pi   = theta[4];
    const double phi_y    = theta[5];
    const double rho_R    = theta[6];
    const double epsilon  = theta[7];
    const double thetx    = theta[8];
    const double alpha    = theta[9];
    const double zeta     = theta[10];
    const double delta    = theta[11];
    const double ln_A_ss  = theta[12];
    const double rho_A    = theta[13];
    const double sigma_A  = theta[14];
    const double ln_sg_ss = theta[15];
    const double rho_sg   = theta[16];
    const double sigma_sg = theta[17];
    const double sigma_xi = theta[18];

    double ln_Ps_ss= LN_PI_STAR(ln_P_ss);
    double ln_R_ss = ln_P_ss - log(beta);
    double ln_v_ss = log(1.0-thetx) - epsilon*ln_Ps_ss - log(1.0-thetx*exp(epsilon*ln_P_ss));
    double ln_m_ss = log(beta);
    double ln_mc_ss= log(1.0-beta*thetx*exp(epsilon*ln_P_ss)) - log(1.0-beta*thetx*exp((epsilon-1.0)*ln_P_ss)) + log((epsilon-1.0)/epsilon) + ln_Ps_ss + ln_v_ss;
    double ln_rk_ss= log((1.0/beta) - 1.0 + delta);
    double ln_Th_ss= (ln_rk_ss - ln_A_ss - log(zeta) - ln_mc_ss)/(zeta-1.0);
    double ln_Ph_ss= log(alpha*(1.0-zeta)/(1.0-alpha)) + ln_mc_ss + ln_A_ss + zeta*ln_Th_ss - ln_v_ss;
    double ln_l_ss = ln_Ph_ss - log((exp(ln_A_ss-ln_v_ss)*(1.0-exp(ln_sg_ss))*exp(zeta*ln_Th_ss)) - (delta*exp(ln_Th_ss)) + exp(ln_Ph_ss));
    double ln_k_ss = ln_Th_ss + ln_l_ss;
    double ln_i_ss = log(delta) + ln_k_ss;
    double ln_c_ss = ln_Ph_ss + log(1.0-exp(ln_l_ss));
    double ln_V_ss = LN_UTILITY(ln_c_ss, ln_l_ss);
    double ln_y_ss = log(exp(ln_c_ss)+exp(ln_i_ss)) - log(1.0-exp(ln_sg_ss));
    double ln_x_ss = ln_mc_ss + ln_y_ss - ln_v_ss - log(1.0-beta*thetx*exp(epsilon*ln_P_ss));

    // {5} exog : ln_P_ss
    //
    // [4] endo : ln_A_ss
    // [5] endo : ln_sg_ss
    
    // exogneous controls (V, i, l, P, x)
    y_ss[0] = ln_V_ss;
    y_ss[1] = ln_i_ss;
    y_ss[2] = ln_l_ss;
    y_ss[3] = ln_P_ss;
    y_ss[4] = ln_x_ss;
    
    // endogenous states (v, R, k, A, sg, xi)
    x_ss[0] = ln_v_ss;
    x_ss[1] = ln_R_ss;
    x_ss[2] = ln_k_ss;
    x_ss[3] = ln_A_ss;
    x_ss[4] = ln_sg_ss;
    x_ss[5] = 0.0;
    
    // shock matrix :
    //
    // [v]           [ 0 0 0 ]
    // [R]           [ 0 0 0 ]
    // [k] = [...] + [ 0 0 0 ]
    // [A]           [ σ 0 0 ]
    // [g]           [ 0 σ 0 ]
    // [x]           [ 0 0 σ ]
    // 
    sigma = sigma_A;

    eta[neps*3 + 0] = 1.0;
    eta[neps*4 + 1] = sigma_sg / sigma_A;
    eta[neps*5 + 2] = sigma_xi / sigma_A;

    int i_X = 0;
    for (int j = 0; j < 2; j++)
	for (int i = 0; i < nx; i++)
	    X_ss[i_X++] = x_ss[i];
	
    for (int j = 0; j < 2; j++)
	for (int i = 0; i < ny; i++)
	    X_ss[i_X++] = y_ss[i];
    
}

void MPK_SGU::perform_autodiff(void)
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
    // ... well, you don't have to, but I find it more convenient.
    #define PSI_X getparam(para[0])
    #define GAMMX getparam(para[1])
    #define BETAX getparam(para[2])
    #define PSS_X getparam(para[3])
    #define PHPIX getparam(para[4])
    #define PHIYX getparam(para[5])
    #define RHORX getparam(para[6])
    #define EPS_X getparam(para[7])
    #define THETX getparam(para[8])
    #define ALPHX getparam(para[9])
    #define ZETAX getparam(para[10])
    #define DELTX getparam(para[11])
    #define ASS_X getparam(para[12])
    #define RHOAX getparam(para[13])
    #define SGA_X getparam(para[14])
    #define GSS_X getparam(para[15])
    #define RHOGX getparam(para[16])
    #define SGG_X getparam(para[17])
    #define SGX_X getparam(para[18])
    
    //printf("2: set up dynamic parameters.\n");
    //------------------------------------------------------------
    // 3.) transfer steady state values to the tape
    
    for (int i = 0; i < N; i++)
	X[i] <<= X_ss[i];
    

    //------------------------------------------------------------
    // 4.) fix variable names (I know it's a pain, but the alter-
    //     native is just remembering where all of them are...)

    adouble ln_vtm1 = X[0];
    adouble ln_Rtm1 = X[1];
    adouble ln_ktm1 = X[2];
    adouble ln_At   = X[3];
    adouble ln_sgt  = X[4];
    adouble ln_xit  = X[5];
    
    adouble ln_vt    = X[6];
    adouble ln_Rt    = X[7];
    adouble ln_kt    = X[8];
    adouble ln_Atp1  = X[9];
    adouble ln_sgtp1 = X[10];
    adouble ln_xitp1 = X[11];
    
    adouble ln_Vt = X[12];
    adouble ln_it = X[13];
    adouble ln_lt = X[14];
    adouble ln_Pt = X[15];
    adouble ln_xt = X[16];

    adouble ln_Vtp1 = X[17];
    adouble ln_itp1 = X[18];
    adouble ln_ltp1 = X[19];
    adouble ln_Ptp1 = X[20];
    adouble ln_xtp1 = X[21];

    // in the pursuit of ln_y_ss
    adouble ln_Ps_ss= ALN_PI_STAR(PSS_X);
    adouble ln_v_ss = log(1.0-THETX) - EPS_X*ln_Ps_ss - log(1.0-THETX*exp(EPS_X*1.0*PSS_X));
    adouble ln_mc_ss= log(1.0-BETAX*1.0*THETX*exp(EPS_X*1.0*PSS_X)) - log(1.0-BETAX*1.0*THETX*exp((EPS_X-1.0)*PSS_X)) + log((EPS_X-1.0)/EPS_X) + ln_Ps_ss + ln_v_ss;
    adouble ln_rk_ss= log((1.0/BETAX) - 1.0 + DELTX);
    adouble ln_Th_ss= (ln_rk_ss - ASS_X - log(ZETAX) - ln_mc_ss)/(ZETAX-1.0);
    adouble ln_Ph_ss= log(ALPHX*(1.0-ZETAX)/(1.0-ALPHX)) + ln_mc_ss + ASS_X + ZETAX*ln_Th_ss - ln_v_ss;
    adouble ln_l_ss = ln_Ph_ss - log((exp(ASS_X-ln_v_ss)*(1.0-exp(GSS_X))*exp(ZETAX*ln_Th_ss)) - (DELTX*exp(ln_Th_ss)) + exp(ln_Ph_ss));
    adouble ln_k_ss = ln_Th_ss + ln_l_ss;
    adouble ln_i_ss = log(DELTX) + ln_k_ss;
    adouble ln_c_ss = ln_Ph_ss + log(1.0-exp(ln_l_ss));
    adouble ln_y_ss = log(exp(ln_c_ss)+exp(ln_i_ss)) - log(1.0-exp(GSS_X));

    // the rest of the helper variables
    adouble ln_Pst= ALN_PI_STAR(ln_Pt);
    adouble ln_ct = log((1.0-exp(ln_sgt))*exp(ln_At + ZETAX*ln_kt + (1.0-ZETAX)*ln_lt - ln_vt) - exp(ln_it));
    adouble ln_wt = log((1.0-ALPHX)/ALPHX) + ln_ct - log(1.0-exp(ln_lt));
    adouble ln_mct= ln_wt - log(1.0-ZETAX) - ln_At - ZETAX*(ln_kt-ln_lt) + ln_vt;
    adouble ln_Ot = (1.0-PSI_X)*ALN_UTILITY(ln_ct, ln_lt) - ln_ct;
    adouble ln_yt = log(exp(ln_ct)+exp(ln_it)) - log(1.0-exp(ln_sgt));
    
    adouble ln_ktp1 = log(exp(ln_itp1) + (1.0-DELTX)*exp(ln_kt));
    adouble ln_Pstp1= ALN_PI_STAR(ln_Ptp1);
    adouble ln_vtp1 = log((1.0-THETX)*exp(-1.0*EPS_X*ln_Pstp1) + THETX*exp(EPS_X*ln_Ptp1 + ln_vt));
    adouble ln_ctp1 = log((1.0-exp(ln_sgtp1))*exp(ln_Atp1 + ZETAX*ln_ktp1 + (1.0-ZETAX)*ln_ltp1 - ln_vtp1) - exp(ln_itp1));
    adouble ln_wtp1 = log((1.0-ALPHX)/ALPHX) + ln_ctp1 - log(1.0-exp(ln_ltp1));
    adouble ln_mctp1= ln_wtp1 - log(1.0-ZETAX) - ln_Atp1 - ZETAX*(ln_ktp1-ln_ltp1) + ln_vtp1;
    adouble ln_rktp1= ln_mctp1 + log(ZETAX) + ln_Atp1 + (ZETAX-1.0)*(ln_ktp1-ln_ltp1) - ln_vtp1;
    adouble ln_Otp1 = (1.0-PSI_X)*ALN_UTILITY(ln_ctp1, ln_ltp1) - ln_ctp1;

    // implement some powers
    adouble ln_v_g_p = (GAMMX+0.0-PSI_X)*ln_Vtp1;
    adouble ln_p_eps = EPS_X*ln_Ptp1;

    // the integrands
    adouble Int1 = exp((1.0-GAMMX)*ln_Vtp1);
    adouble Int2 = exp(ln_v_g_p + ln_Otp1 - ln_Ptp1);
    adouble Int3 = exp(ln_v_g_p + ln_Otp1 + log(exp(ln_rktp1) + 1.0 - DELTX));
    adouble Int4 = exp(ln_v_g_p + ln_Otp1 + ln_p_eps + ln_xtp1);
    adouble Int5 = exp(ln_v_g_p + ln_Otp1 + ln_p_eps + ln_xtp1 - ln_Ptp1 - ln_Pstp1);
    
    ln_v_g_p = (GAMMX+0.0-PSI_X)*log(Int1)/(1.0-GAMMX);
    
    //------------------------------------------------------------
    // 5.) define your DSGE


    // Eq. 1: Epstein-Zin recursion
    Y[0] = (1.0-PSI_X)*ln_Vt - log((1.0-BETAX)*exp(ln_Ot+ln_ct) + (BETAX*pow(Int1,(1.0-PSI_X)/(1.0-GAMMX))));
    
    // Eq. 2: interest rate expectation
    Y[1] = ln_v_g_p + ln_Ot - log(BETAX) - ln_Rt - log(Int2);
    
    // Eq. 3: rental rate expectation
    Y[2] = ln_v_g_p + ln_Ot - log(BETAX) - log(Int3);
    
    // Eq. 4: x1t recursion (1)
    Y[3] = ln_v_g_p + ln_Ot + log(exp(ln_xt)-exp(ln_mct+ln_yt-ln_vt)) - log(THETX*1.0*BETAX*Int4);
    
    // Eq. 5: x1t recursion (2)
    Y[4] = ln_v_g_p + ln_Ot + log(exp(ln_xt)-((EPS_X-1.0)/EPS_X)*exp(ln_Pst+ln_yt)) - ln_Pst - log(THETX*1.0*BETAX*Int5);

    // Eq. 6: spread transition
    Y[5] = ln_vt - log((1.0-THETX)*exp(-1.0*EPS_X*ln_Pst) + THETX*exp(EPS_X*ln_Pt + ln_vtm1));
    
    // Eq. 7: interest rate transition
    Y[6] = ln_Rt - (RHORX*ln_Rtm1 + (1.0-RHORX)*(PSS_X-log(BETAX) + PHPIX*(ln_Pt-PSS_X) + PHIYX*(ln_yt-ln_y_ss)) + ln_xit);
    
    // Eq. 8: capital transition
    Y[7] = ln_kt - log(exp(ln_it) + (1.0-DELTX)*exp(ln_ktm1));
    
    // Eq. 9: productivity shock
    Y[8] = (1.0-RHOAX)*ASS_X + RHOAX*ln_At - ln_Atp1;

    // Eq. 10: government spending shock
    Y[9] = (1.0-RHOGX)*GSS_X + RHOGX*ln_sgt - ln_sgtp1;

    // Eq. 11: monetary policy shock
    Y[10]= 0.0 - ln_xitp1;

    
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
