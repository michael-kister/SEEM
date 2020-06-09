
#include "helper_structs.h"

Parameters::Parameters(const double* P_in)
{
    psi      = P_in[0];
    gama     = P_in[1];
    beta     = P_in[2];
    ln_P_ss  = P_in[3];
    phi_pi   = P_in[4];
    phi_y    = P_in[5];
    rho_R    = P_in[6];
    epsilon  = P_in[7];
    theta    = P_in[8];
    alpha    = P_in[9];
    zeta     = P_in[10];
    delta    = P_in[11];
    ln_A_ss  = P_in[12];
    rho_A    = P_in[13];
    sigma_A  = P_in[14];
    ln_sg_ss = P_in[15];
    rho_sg   = P_in[16];
    sigma_sg = P_in[17];
    sigma_xi = P_in[18];
}

Parameters::Parameters(const Parameters& that)
{
    psi      = that.psi;
    gama     = that.gama;
    beta     = that.beta;
    ln_P_ss  = that.ln_P_ss;
    phi_pi   = that.phi_pi;
    phi_y    = that.phi_y;
    rho_R    = that.rho_R;
    epsilon  = that.epsilon;
    theta    = that.theta;
    alpha    = that.alpha;
    zeta     = that.zeta;
    delta    = that.delta;
    ln_A_ss  = that.ln_A_ss;
    rho_A    = that.rho_A;
    sigma_A  = that.sigma_A;
    ln_sg_ss = that.ln_sg_ss;
    rho_sg   = that.rho_sg;
    sigma_sg = that.sigma_sg;
    sigma_xi = that.sigma_xi;
}

void swap(Parameters& first, Parameters& second)
{
    std::swap( first.psi ,      second.psi      );
    std::swap( first.gama ,     second.gama     );
    std::swap( first.beta ,     second.beta     );
    std::swap( first.ln_P_ss ,  second.ln_P_ss  );
    std::swap( first.phi_pi ,   second.phi_pi   );
    std::swap( first.phi_y ,    second.phi_y    );
    std::swap( first.rho_R ,    second.rho_R    );
    std::swap( first.epsilon ,  second.epsilon  );
    std::swap( first.theta ,    second.theta    );
    std::swap( first.alpha ,    second.alpha    );
    std::swap( first.zeta ,     second.zeta     );
    std::swap( first.delta ,    second.delta    );
    std::swap( first.ln_A_ss ,  second.ln_A_ss  );
    std::swap( first.rho_A ,    second.rho_A    );
    std::swap( first.sigma_A ,  second.sigma_A  );
    std::swap( first.ln_sg_ss , second.ln_sg_ss );
    std::swap( first.rho_sg ,   second.rho_sg   );
    std::swap( first.sigma_sg , second.sigma_sg );
    std::swap( first.sigma_xi , second.sigma_xi );
}

Parameters& Parameters::operator=(Parameters other)
{
    swap(*this, other);
    return *this;
}

Parameters::operator std::vector<double>() const
{
    std::vector<double> vec;
    vec.push_back(psi);
    vec.push_back(gama);
    vec.push_back(beta);
    vec.push_back(ln_P_ss);
    vec.push_back(phi_pi);
    vec.push_back(phi_y);
    vec.push_back(rho_R);
    vec.push_back(epsilon);
    vec.push_back(theta);
    vec.push_back(alpha);
    vec.push_back(zeta);
    vec.push_back(delta);
    vec.push_back(ln_A_ss);
    vec.push_back(rho_A);
    vec.push_back(sigma_A);
    vec.push_back(ln_sg_ss);
    vec.push_back(rho_sg);
    vec.push_back(sigma_sg);
    vec.push_back(sigma_xi);
    return vec;
}


SteadyStates::SteadyStates(const Parameters& P)
{
    ln_Ps_ss = (log(1.0-P.theta*exp((P.epsilon-1.0)*P.ln_P_ss))-log(1.0-P.theta))/(1.0-P.epsilon);
    ln_R_ss  = P.ln_P_ss - log(P.beta);
    ln_v_ss  = log(1.0-P.theta) - P.epsilon*ln_Ps_ss - log(1.0-P.theta*exp(P.epsilon*P.ln_P_ss));
    ln_m_ss  = log(P.beta);
    ln_mc_ss = log(1.0-P.beta*P.theta*exp(P.epsilon*P.ln_P_ss)) - log(1.0-P.beta*P.theta*exp((P.epsilon-1.0)*P.ln_P_ss)) + log((P.epsilon-1.0)/P.epsilon) + ln_Ps_ss + ln_v_ss;
    ln_rk_ss = log((1.0/P.beta) - 1.0 + P.delta);
    
    double ln_Th_ss = (ln_rk_ss - P.ln_A_ss - log(P.zeta) - ln_mc_ss)/(P.zeta-1.0);
    double ln_Ph_ss = log(P.alpha*(1.0-P.zeta)/(1.0-P.alpha)) + ln_mc_ss + P.ln_A_ss + P.zeta*ln_Th_ss - ln_v_ss;

    ln_l_ss  = ln_Ph_ss - log((exp(P.ln_A_ss-ln_v_ss)*(1.0-exp(P.ln_sg_ss))*exp(P.zeta*ln_Th_ss)) - (P.delta*exp(ln_Th_ss)) + exp(ln_Ph_ss));
    ln_k_ss  = ln_Th_ss + ln_l_ss;
    ln_i_ss  = log(P.delta) + ln_k_ss;
    ln_c_ss  = ln_Ph_ss + log(1.0-exp(ln_l_ss));
    ln_V_ss  = P.alpha*ln_c_ss + (1.0-P.alpha)*log(1.0-exp(ln_l_ss));    
    ln_y_ss  = log(exp(ln_c_ss)+exp(ln_i_ss)) - log(1.0-exp(P.ln_sg_ss));
    ln_x_ss  = ln_mc_ss + ln_y_ss - ln_v_ss - log(1.0-P.beta*P.theta*exp(P.epsilon*P.ln_P_ss));

    ln_P_ss  = P.ln_P_ss;
    ln_A_ss  = P.ln_A_ss;
    ln_sg_ss = P.ln_sg_ss;
}

SteadyStates::SteadyStates(const SteadyStates& that)
{
    ln_Ps_ss = that.ln_Ps_ss;
    ln_R_ss  = that.ln_R_ss;
    ln_v_ss  = that.ln_v_ss;
    ln_m_ss  = that.ln_m_ss;
    ln_mc_ss = that.ln_mc_ss;
    ln_rk_ss = that.ln_rk_ss;
    ln_l_ss  = that.ln_l_ss;
    ln_k_ss  = that.ln_k_ss;
    ln_i_ss  = that.ln_i_ss;
    ln_c_ss  = that.ln_c_ss;
    ln_V_ss  = that.ln_V_ss;
    ln_y_ss  = that.ln_y_ss;
    ln_x_ss  = that.ln_x_ss;

    ln_P_ss  = that.ln_P_ss;
    ln_A_ss  = that.ln_A_ss;
    ln_sg_ss = that.ln_sg_ss;
}

void swap(SteadyStates& first, SteadyStates& second)
{
    std::swap( first.ln_Ps_ss , second.ln_Ps_ss );
    std::swap( first.ln_R_ss ,  second.ln_R_ss  );
    std::swap( first.ln_v_ss ,  second.ln_v_ss  );
    std::swap( first.ln_m_ss ,  second.ln_m_ss  );
    std::swap( first.ln_mc_ss , second.ln_mc_ss );
    std::swap( first.ln_rk_ss , second.ln_rk_ss );
    std::swap( first.ln_l_ss ,  second.ln_l_ss  );
    std::swap( first.ln_k_ss ,  second.ln_k_ss  );
    std::swap( first.ln_i_ss ,  second.ln_i_ss  );
    std::swap( first.ln_c_ss ,  second.ln_c_ss  );
    std::swap( first.ln_V_ss ,  second.ln_V_ss  );
    std::swap( first.ln_y_ss ,  second.ln_y_ss  );
    std::swap( first.ln_x_ss ,  second.ln_x_ss  );

    std::swap( first.ln_P_ss ,  second.ln_P_ss  );
    std::swap( first.ln_A_ss ,  second.ln_A_ss  );
    std::swap( first.ln_sg_ss , second.ln_sg_ss );
}

SteadyStates& SteadyStates::operator=(SteadyStates other)
{
    swap(*this, other);
    return *this;
}

void SteadyStates::print(void)
{
    printf("\n");
    printf("----------------------------\n");
    printf(" Steady States:\n");
    printf("----------------------------\n");
    printf(" States");
    printf("\t\tv : % 7.4f\n", ln_v_ss);
    printf("\t\tR : % 7.4f\n", ln_R_ss);
    printf("\t\tk : % 7.4f\n", ln_k_ss);
    printf("\t\tA : % 7.4f\n", ln_A_ss);
    printf("\t\tsg: % 7.4f\n", ln_sg_ss);
    printf("\t\tξ : % 7.4f\n", 0.0);
    printf(" Decisions");
    printf("\tV : % 7.4f\n", ln_V_ss);
    printf("\t\ti : % 7.4f\n", ln_i_ss);
    printf("\t\tl : % 7.4f\n", ln_l_ss);
    printf("\t\tΠ : % 7.4f\n", ln_P_ss);
    printf("\t\tx : % 7.4f\n", ln_x_ss);
    printf(" Observations");
    printf("\ty : % 7.4f\n", ln_y_ss);
    printf("\t\tc : % 7.4f\n", ln_c_ss);
    printf("\t\tl : % 7.4f\n", ln_l_ss);
    printf("\t\tΠ : % 7.4f\n", ln_P_ss);
    printf("\t\tR : % 7.4f\n", ln_R_ss);
    printf("----------------------------\n");
    printf("\n");
}
