/*------------------------------------------------------------------------------
 * This is a function to shift the particles according to the transition equa-
 * tion. We will hard-code this stuff. We have singular transitions for:
 *  - v (spread)
 *  - R (Rates)
 *  - k (capital)
 * and we have stochastic transitions for:
 *  - A (productivity)
 *  - s (government share)
 *  - ξ (monetary policy shock)
 *
 * A key regarding the inputs of functions:
 *  * Parameter
 *  @ State series
 *  ! State series that cannot be used again
 *  $ Observable series
 *----------------------------------------------------------------------------*/
#define ln_v_tm2  x_tm1[NUM_STATE*id+0]
#define ln_v_tm1  x_tm1[NUM_STATE*id+0]
#define ln_R_tm2  x_tm1[NUM_STATE*id+1]
#define ln_R_tm1  x_tm1[NUM_STATE*id+1]
#define ln_k_tm2  x_tm1[NUM_STATE*id+2]
#define ln_k_tm1  x_tm1[NUM_STATE*id+2]
#define ln_A_tm1  x_tm1[NUM_STATE*id+3]
#define ln_A_t    x_tm1[NUM_STATE*id+3]
#define ln_sg_tm1 x_tm1[NUM_STATE*id+4]
#define ln_sg_t   x_tm1[NUM_STATE*id+4]
#define ln_xi_tm1 x_tm1[NUM_STATE*id+5]
#define ln_xi_t   x_tm1[NUM_STATE*id+5]

#define ln_y_tm1 y_tm1[NUM_MEASR*id+0]
#define ln_i_tm1 y_tm1[NUM_MEASR*id+2]
#define ln_P_tm1 y_tm1[NUM_MEASR*id+4]

#define ln_P_ss  parameters[3]
#define phi_pi   parameters[4]
#define phi_y    parameters[5]
#define rho_R    parameters[6]
#define epsilon  parameters[7]
#define theta    parameters[8]
#define delta    parameters[11]
#define ln_A_ss  parameters[12]
#define rho_A    parameters[13]
#define ln_sg_ss parameters[15]
#define rho_sg   parameters[16]

__global__ void transition_equation
(int N, double* x_tm1, const double* y_tm1, const double* scaled_shocks_t,
 const double* parameters)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < N) {

	// Lagged Pi star
	// * theta
	// * epsilon
	// $ ln_P_tm1
	double ln_Pst_tm1 = (log(1.0-theta*exp((epsilon-1.0)*ln_P_tm1)) -
			      log(1.0-theta)
			     )/(1.0-epsilon);

	// Lagged Spread............ No [ln_v_tm2] below here!!!
	// * theta
	// * epsilon
	// @ ln_Pst_tm1
	// $ ln_P_tm1
	// ! ln_v_tm2
	ln_v_tm1 = log((1.0-theta)*exp(-1.0*epsilon*ln_Pst_tm1) +
		       theta*exp(epsilon*ln_P_tm1 + ln_v_tm2));

	// Lagged Interest Rates.... No [ln_R_tm2] below here!!!
	// * rho_R
	// * ln_R_ss
	// * phi_pi
	// * ln_P_ss
	// * phi_y
	// * ln_y_ss
	// ! ln_R_tm2
	// $ ln_P_tm1
	// $ ln_y_tm1
	ln_R_tm1 = (rho_R*ln_R_tm2 + ln_xi_tm1 +
		    (1.0-rho_R)*(ln_R_ss + phi_pi*(ln_P_tm1-ln_P_ss) +
				 phi_y*(ln_y_tm1-ln_y_ss)));
	ln_R_tm1 = (ln_R_tm1 < 0.0) ? 0.0 : ln_R_tm1;

	// Lagged Capital........... No [ln_k_tm2] below here!!!
	// * delta
	// @ ln_i_tm1
	// ! ln_k_tm2
	ln_k_tm1 = log(exp(ln_i_tm1) + (1.0-delta)*exp(ln_k_tm2));
	
	// Productivity shock....... No [ln_A_tm1] below here!!!
	// * rho_A
	// * ln_A_ss
	// ! ln_A_tm1
	ln_A_t = rho_A * ln_A_tm1 + (1.0 - rho_A) * ln_A_ss + eps_A_t;

	// Government shock......... No [ln_sg_tm1] below here!!!
	// * rho_sg
	// * ln_sg_ss
	// ! ln_sg_tm1
	ln_sg_t = rho_sg * ln_sg_tm1 + (1.0 - rho_sg) * ln_sg_ss + eps_sg_t;

	// Policy shock............. No [ln_xi_tm1] below here!!!
	// ! ln_xi_tm2
	ln_xi_t = eps_xi_t;
	
    }
}
#undef ln_v_tm2
#undef ln_v_tm1
#undef ln_R_tm2
#undef ln_R_tm1
#undef ln_k_tm2
#undef ln_k_tm1
#undef ln_A_tm1
#undef ln_A_t
#undef ln_sg_tm1
#undef ln_sg_t
#undef ln_xi_tm1
#undef ln_xi_t

#undef ln_y_tm1
#undef ln_i_tm1
#undef ln_P_tm1

#undef ln_P_ss
#undef phi_pi
#undef phi_y
#undef rho_R
#undef epsilon
#undef theta
#undef delta
#undef ln_A_ss
#undef rho_A
#undef ln_sg_ss
#undef rho_sg
