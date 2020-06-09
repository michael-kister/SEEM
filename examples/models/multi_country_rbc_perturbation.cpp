#include "multi_country_rbc_perturbation.h"

//========================================//
//                                        //
//          MODEL SPECIFICATION           //
//                                        //
//========================================//

MCRBC_SGU::MCRBC_SGU(data Y_, short int tag_):
    perturbation(20, 62, 11, 9, 2, tag_), Y(Y_)
{
    printf("\t\t\tConstructing 'MCRBC_SGU' (%p)\n\n", this);
    
    LB = new double [npar];
    UB = new double [npar];
    theta_0 = new double [npar];
/*
    Hmat = new double[n_me * n_me]();
    Qmat = new double[neps * neps]();
    Rmat = new double[nx * neps]();

    Rmat[neps*(2+0)+0] = 1.0;
    Rmat[neps*(2+1)+1] = 1.0;
    Rmat[neps*(2+2)+2] = 1.0;
    Rmat[neps*(2+3)+3] = 1.0;
*/  
    names.resize(npar);
    descriptions.resize(npar);

    names[0] = "β  "; descriptions[0] = "Subjective time discount factor";
    names[1] = "α  "; descriptions[1] = "Capital elasticity of production function";
    names[2] = "δ  "; descriptions[2] = "Depreciation rate of capital";
    names[3] = "σ  "; descriptions[3] = "StDev of shocks";
    names[4] = "ρ  "; descriptions[4] = "Serial correlation of productivity shock";
    names[5] = "φ  "; descriptions[5] = "Intensity of friction (adjustment cost)";
    
    names[6] = "γ  "; descriptions[6] = "Degree of risk aversion";
    names[7] = "μ  "; descriptions[6] = "Elasticity of subs. between capital & labor";
    names[8] = "χ  "; descriptions[6] = "Elasticity of subs. between consumption & leisure";

    LB[0] = 0.985;     theta[0] = 0.99 ;     UB[0] = 0.995;
    LB[0] = 0.35 ;     theta[0] = 0.36 ;     UB[0] = 0.37 ;
    LB[0] = 0.024;     theta[0] = 0.025;     UB[0] = 0.026;
    LB[0] = 0.009;     theta[0] = 0.01 ;     UB[0] = 0.011;
    LB[0] = 0.94 ;     theta[0] = 0.95 ;     UB[0] = 0.96 ;
    LB[0] = 0.49 ;     theta[0] = 0.5  ;     UB[0] = 0.51 ;
    
    LB[0] = 0.24 ;     theta[0] = 0.25 ;     UB[0] = 0.26 ;
    LB[0] =-0.21 ;     theta[0] =-0.2  ;     UB[0] =-0.19 ;
    LB[0] = 0.82 ;     theta[0] = 0.83 ;     UB[0] = 0.84 ;

    perform_autodiff(); 
    verbose = 0;

    printf("\t\t\tDone.\n\n");
}

// copy constructor
MCRBC_SGU::MCRBC_SGU(const MCRBC_SGU& that):
    perturbation(that), Y(that.Y)
{
    printf("\t\t\tCopy constructor for 'MCRBC_SGU' (%p receiving %p)\n\n", this, &that);

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
void swap(MCRBC_SGU& first, MCRBC_SGU& second)
{
    printf("\t\t\tSwapping 'MCRBC_SGU' (%p <--> %p)\n\n", &first, &second);

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


MCRBC_SGU::~MCRBC_SGU(void)
{
    printf("\t\t\t-Destructing 'MCRBC_SGU' (%p)... ", this);

    delete[] LB;
    delete[] UB;
    delete[] theta_0;

    delete[] Rmat;
    delete[] Hmat;
    delete[] Qmat;

    printf("Done.\n\n");
}

void MCRBC_SGU::print_theta(void)
{
    printf("------------------------------------------------\n");
    printf("Parameter Values:\n\n");
    for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++)
	    printf(" %3s = %+08.5f ", names[3*i+j].c_str(), theta[3*i+j]);
	printf("\n");
    }
    printf("------------------------------------------------\n");
}


void MCRBC_SGU::print_descriptions(void)
{
    printf("----------------------------------------------------\n");
    printf("Parameter Descriptions:\n\n");
    for (int i = 0; i < npar; i++)
	    printf(" %3s : %-50s\n", names[i].c_str(), descriptions[i].c_str());
    printf("----------------------------------------------------\n");
}


void MCRBC_SGU::operator()(const double* theta_unbounded)
{
    double theta_bounded[npar];

    for (int i = 0; i < npar; i++) {
	theta_bounded[i] = inverse_logit(theta_unbounded[i]);
	theta_bounded[i] = change_bounds(theta_bounded[i], 0.0, 1.0, LB[i], UB[i]);
    }

    load_parameters(theta_bounded, 0);
    
    if (verbose)
	print_theta();
/*
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
*/
}


void MCRBC_SGU::set_steady_state(void)
{
    const double beta  = theta[0];
    const double alpha = theta[1];
    const double delta = theta[2];

    double A = (1.0 - beta)/(alpha * beta);
    
    // controls (c, l, i, lam)

    for (int i = 0; i < num_country; i++) {
	y_ss[i+0*num_country] = A;     // consumption
	y_ss[i+1*num_country] = 1.0;   // labor
	y_ss[i+2*num_country] = delta; // investment
    }
    y_ss[3*num_country] = 1.0;         // lambda


    // states (k, a)

    for (int i = 0; i < num_country; i++) {
	x_ss[i+0*num_country] = 1.0;   // capital
	x_ss[i+1*num_country] = 1.0;   // productivity
    }
    
    // shock matrix
    sigma = theta[3];

    for (int i = 0; i < num_country; i++)
	eta[neps*i] = eta[neps*i+(i+1)] = 1.0;
    
    int i_X = 0;
    for (int j = 0; j < 2; j++)
	for (int i = 0; i < nx; i++)
	    X_ss[i_X++] = x_ss[i];
	
    for (int j = 0; j < 2; j++)
	for (int i = 0; i < ny; i++)
	    X_ss[i_X++] = y_ss[i];
    
}

void MCRBC_SGU::perform_autodiff(void)
{
    //------------------------------------------------------------
    // 0.) Obtain the steady state values
    int N = nx + nx + ny + ny;
    
    set_steady_state();
    
    double* pxy  = new double [nx+ny];
    
    //------------------------------------------------------------
    // 1.) define the tag; start the trace; set up the variables

    printf("T:\t\t\t\tTurning on trace %d\n\n", tag);
    trace_on(tag);

    adouble* X = new adouble [N];
    adouble* Y = new adouble [nx+ny];
    
    //------------------------------------------------------------
    // 2.) some work with the parameters

    locint* para = new locint [npar];
    for (int i = 0; i < npar; i++)
	para[i] = mkparam_idx(theta[i]);


    // you must manually name all the variables here!
    #define BETAX getparam(para[0])
    #define ALPHX getparam(para[1])
    #define DELTX getparam(para[2])
    #define SIGMX getparam(para[3])
    #define RHO_X getparam(para[4])
    #define PHI_X getparam(para[5])
    
    #define GAMMX getparam(para[6])
    #define MU__X getparam(para[7])
    #define CHI_X getparam(para[8])

    //------------------------------------------------------------
    // 3.) transfer steady state values to the tape
    
    for (int i = 0; i < N; i++)
	X[i] <<= X_ss[i];
    

    //------------------------------------------------------------
    // 4.) fix variable names (I know it's a pain, but the alter-
    //     native is just remembering where all of them are...)

    // STATES
    
    adouble k_01_t = exp( X[0] );
    adouble k_02_t = exp( X[1] );
    adouble k_03_t = exp( X[2] );
    adouble k_04_t = exp( X[3] );
    adouble k_05_t = exp( X[4] );
    adouble k_06_t = exp( X[5] );
    adouble k_07_t = exp( X[6] );
    adouble k_08_t = exp( X[7] );
    adouble k_09_t = exp( X[8] );
    adouble k_10_t = exp( X[9] );

    adouble a_01_t = exp( X[10] );
    adouble a_02_t = exp( X[11] );
    adouble a_03_t = exp( X[12] );
    adouble a_04_t = exp( X[13] );
    adouble a_05_t = exp( X[14] );
    adouble a_06_t = exp( X[15] );
    adouble a_07_t = exp( X[16] );
    adouble a_08_t = exp( X[17] );
    adouble a_09_t = exp( X[18] );
    adouble a_10_t = exp( X[19] );

    adouble k_01_tt = exp( X[20] );
    adouble k_02_tt = exp( X[21] );
    adouble k_03_tt = exp( X[22] );
    adouble k_04_tt = exp( X[23] );
    adouble k_05_tt = exp( X[24] );
    adouble k_06_tt = exp( X[25] );
    adouble k_07_tt = exp( X[26] );
    adouble k_08_tt = exp( X[27] );
    adouble k_09_tt = exp( X[28] );
    adouble k_10_tt = exp( X[29] );

    adouble a_01_tt = exp( X[30] );
    adouble a_02_tt = exp( X[31] );
    adouble a_03_tt = exp( X[32] );
    adouble a_04_tt = exp( X[33] );
    adouble a_05_tt = exp( X[34] );
    adouble a_06_tt = exp( X[35] );
    adouble a_07_tt = exp( X[36] );
    adouble a_08_tt = exp( X[37] );
    adouble a_09_tt = exp( X[38] );
    adouble a_10_tt = exp( X[39] );

    // CONTROLS

    adouble c_01_t = exp( X[40] );
    adouble c_02_t = exp( X[41] );
    adouble c_03_t = exp( X[42] );
    adouble c_04_t = exp( X[43] );
    adouble c_05_t = exp( X[44] );
    adouble c_06_t = exp( X[45] );
    adouble c_07_t = exp( X[46] );
    adouble c_08_t = exp( X[47] );
    adouble c_09_t = exp( X[48] );
    adouble c_10_t = exp( X[49] );
    
    adouble l_01_t = exp( X[50] );
    adouble l_02_t = exp( X[51] );
    adouble l_03_t = exp( X[52] );
    adouble l_04_t = exp( X[53] );
    adouble l_05_t = exp( X[54] );
    adouble l_06_t = exp( X[55] );
    adouble l_07_t = exp( X[56] );
    adouble l_08_t = exp( X[57] );
    adouble l_09_t = exp( X[58] );
    adouble l_10_t = exp( X[59] );
    
    adouble i_01_t = exp( X[60] );
    adouble i_02_t = exp( X[61] );
    adouble i_03_t = exp( X[62] );
    adouble i_04_t = exp( X[63] );
    adouble i_05_t = exp( X[64] );
    adouble i_06_t = exp( X[65] );
    adouble i_07_t = exp( X[66] );
    adouble i_08_t = exp( X[67] );
    adouble i_09_t = exp( X[68] );
    adouble i_10_t = exp( X[69] );

    adouble lam_t = exp( X[70] );

    adouble c_01_tt = exp( X[71] );
    adouble c_02_tt = exp( X[72] );
    adouble c_03_tt = exp( X[73] );
    adouble c_04_tt = exp( X[74] );
    adouble c_05_tt = exp( X[75] );
    adouble c_06_tt = exp( X[76] );
    adouble c_07_tt = exp( X[77] );
    adouble c_08_tt = exp( X[78] );
    adouble c_09_tt = exp( X[79] );
    adouble c_10_tt = exp( X[80] );
    
    adouble l_01_tt = exp( X[81] );
    adouble l_02_tt = exp( X[82] );
    adouble l_03_tt = exp( X[83] );
    adouble l_04_tt = exp( X[84] );
    adouble l_05_tt = exp( X[85] );
    adouble l_06_tt = exp( X[86] );
    adouble l_07_tt = exp( X[87] );
    adouble l_08_tt = exp( X[88] );
    adouble l_09_tt = exp( X[89] );
    adouble l_10_tt = exp( X[90] );
    
    adouble i_01_tt = exp( X[91] );
    adouble i_02_tt = exp( X[92] );
    adouble i_03_tt = exp( X[93] );
    adouble i_04_tt = exp( X[94] );
    adouble i_05_tt = exp( X[95] );
    adouble i_06_tt = exp( X[96] );
    adouble i_07_tt = exp( X[97] );
    adouble i_08_tt = exp( X[98] );
    adouble i_09_tt = exp( X[99] );
    adouble i_10_tt = exp( X[100] );

    adouble lam_tt = exp( X[101] );

    
    // useful scalars
    
    adouble Ascal = (1.0 - BETAX)/(ALPHX * 1.0 * BETAX); // a constant

    adouble Lscal = 2.5; // labor endowment of representative agent

    adouble Bscal = (1.0-ALPHX)*pow(Ascal, 1.0-(1.0/CHI_X))*pow(Lscal-1.0, 1.0/CHI_X);

    // partial derivatives

#define DU_DC_(c,l) pow(c, 1.0-(1.0/CHI_X)) + Bscal*pow(Lscal - l, 1.0-(1.0/CHI_X))
//#define DU_DC_(c,l) (c)
    
#define DU_DC(c,l) (pow(c,-1.0/CHI_X)*pow(DU_DC_(c,l), (GAMMX-(0.0+CHI_X))/(GAMMX*(CHI_X-1.0))))

#define DU_DL(c,l) (-1.0*Bscal*pow(c,1.0/CHI_X)*pow(DU_DC_(c,l), (CHI_X*(GAMMX-1.0))/(GAMMX*(CHI_X-1.0))) / \
		    (Bscal*pow(c,1.0/CHI_X)*(Lscal-l) + c*pow(Lscal-l,1.0/CHI_X)))

//#define DU_DL(c,l) (-1.0*Bscal*pow(c,1.0/CHI_X)*pow(DU_DC_(c,l),(CHI_X*(GAMMX-1.0))/(GAMMX*(CHI_X-1.0))))

#define TAU(j) (DU_DC(Ascal, 1.0))

#define PRODFUN_(k,l) (ALPHX*pow(k,MU__X) + (1.0-ALPHX)*pow(l,MU__X))
    
#define PRODFUN(k,l) (Ascal*pow(PRODFUN_(k,l), 1.0-ALPHX))

#define DF_DK(k,l) (Ascal*ALPHX*pow(k,MU__X-1.0)*pow(PRODFUN_(k,l), (1.0/MU__X)-1.0))

#define DF_DL(k,l) (Ascal*(ALPHX-1.0)*pow(l,MU__X-1.0)*pow(PRODFUN_(k,l), (1.0/MU__X)-1.0))

    // first-order condition templates
    
#define FOCOND1(j) TAU(j)*DU_DC(c_##j##_t, l_##j##_t) - lam_t
    
#define FOCOND2(j) TAU(j)*DU_DL(c_##j##_t, l_##j##_t) + lam_t*a_##j##_t*DF_DL(k_##j##_t, l_##j##_t)

#define FOCOND3(j) (lam_t*(1.0+PHI_X*((i_##j##_t/k_##j##_t)-DELTX))) - \
	BETAX*lam_tt*(1.0 + a_##j##_tt*DF_DK(k_##j##_tt, l_##j##_tt) + \
		      PHI_X*(1.0 - DELTX + (i_##j##_tt/k_##j##_tt) -   \
			     0.5*((i_##j##_tt/k_##j##_tt)-DELTX))*     \
		      ((i_##j##_tt/k_##j##_tt)-DELTX))

#define FOCOND4(j) k_##j##_tt - ((1.0 - DELTX)*k_##j##_t + i_##j##_t)

    // useful summation macro
    
#define VARSUM(v,t) (v##_01_##t + v##_02_##t + v##_03_##t + v##_04_##t + v##_05_##t + \
		     v##_06_##t + v##_07_##t + v##_08_##t + v##_09_##t + v##_10_##t)
    
#define SUMGRAND(j) (a_##j##_t*PRODFUN(k_##j##_t, l_##j##_t) - (PHI_X/2.0)* \
		     (k_##j##_t*((i_##j##_t/k_##j##_t)-DELTX)*((i_##j##_t/k_##j##_t)-DELTX)))

    //------------------------------------------------------------
    // 5.) define your DSGE

    // first-order condition #1

    Y[0] = FOCOND1(01);
    Y[1] = FOCOND1(02);
    Y[2] = FOCOND1(03);
    Y[3] = FOCOND1(04);
    Y[4] = FOCOND1(05);
    Y[5] = FOCOND1(06);
    Y[6] = FOCOND1(07);
    Y[7] = FOCOND1(08);
    Y[8] = FOCOND1(09);
    Y[9] = FOCOND1(10);
    
    // first-order condition #2

    Y[10] = FOCOND2(01);
    Y[11] = FOCOND2(02);
    Y[12] = FOCOND2(03);
    Y[13] = FOCOND2(04);
    Y[14] = FOCOND2(05);
    Y[15] = FOCOND2(06);
    Y[16] = FOCOND2(07);
    Y[17] = FOCOND2(08);
    Y[18] = FOCOND2(09);
    Y[19] = FOCOND2(10);
    
    // first-order condition #3

    Y[20] = FOCOND3(01);
    Y[21] = FOCOND3(02);
    Y[22] = FOCOND3(03);
    Y[23] = FOCOND3(04);
    Y[24] = FOCOND3(05);
    Y[25] = FOCOND3(06);
    Y[26] = FOCOND3(07);
    Y[27] = FOCOND3(08);
    Y[28] = FOCOND3(09);
    Y[29] = FOCOND3(10);
    
    // first-order condition #4

    Y[30] = FOCOND4(01);
    Y[31] = FOCOND4(02);
    Y[32] = FOCOND4(03);
    Y[33] = FOCOND4(04);
    Y[34] = FOCOND4(05);
    Y[35] = FOCOND4(06);
    Y[36] = FOCOND4(07);
    Y[37] = FOCOND4(08);
    Y[38] = FOCOND4(09);
    Y[39] = FOCOND4(10);

    // summation equation

    Y[40] = VARSUM(c,t) + VARSUM(i,t) - DELTX*VARSUM(k,t) -
	(SUMGRAND(01) + SUMGRAND(02) + SUMGRAND(03) + SUMGRAND(04) + SUMGRAND(05) +
	 SUMGRAND(06) + SUMGRAND(07) + SUMGRAND(08) + SUMGRAND(09) + SUMGRAND(10));
        
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

#undef DU_DC_
#undef DU_DC
#undef DU_DL
#undef TAU
#undef PRODFUN_
#undef PRODFUN
#undef DU_DK
#undef DU_DL
#undef FOCOND1
#undef FOCOND2
#undef FOCOND3
#undef FOCOND4
#undef VARSUM
#undef SUMGRAND

}
