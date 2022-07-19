functions {
    /*
    Determine the parameters nu and tau for the scaled inverse chi squared
    distribution from the distribution's desired mode and variance
    */
    vector get_scaled_inv_chi2_nu_tau(real mode, real v){
        if (fabs(mode*v)<1e-50) {
            // prevent improper input
            reject("mode x variance != 0");
        }
        // internal variables
        real inner_term;
        vector[2] nu_tau;
        // some helper expressions
        real mode2 = mode*mode;
        real mode3 = mode2 * mode;
        real mode4 = mode2 * mode2;
        real v2 = v*v;
        real v3 = v2 * v;
        real v4 = v2 * v2;
        inner_term = pow(mode3 * v3 + 75 * mode2 * v2 + 6*sqrt(3) * sqrt(mode3*mode2 * v3*v2 + 47*mode4*v4 + 3*mode3*v3) + 21*mode*v + 1, 0.333333333);
        // determine nu, then tau
        nu_tau[1] = 2 * inner_term / (3*mode*v) * (-4*mode2*v2 - 56*mode*v-4) / (6*mode*v * inner_term) + 2*(4*mode*v+1)/(3*mode*v);
        nu_tau[2] = sqrt(mode*(nu_tau[1]+2)/nu_tau[1]);
        return nu_tau;
    }

    real get_scaled_inv_chi2_mode(real nu, real tau){
        return nu * tau^2 / (nu + 2);
    }
}


data{
    int<lower=0> N_tot;
    vector[N_tot] t;
    vector[N_tot] inv_a;
    vector[N_tot] inv_a_err;
}

transformed data {
    vector[N_tot] inv_a_recentre = inv_a - mean(inv_a);
    //vector<lower=0>[2] reg_nu_tau = get_scaled_inv_chi2_nu_tau(1, 10);
    //vector<lower=0>[2] H_nu_tau = get_scaled_inv_chi2_nu_tau(1, 10);
}

parameters{
    real alpha;
    real beta;
    real HGp_s;
    real c;
    real<lower=0> sigma2_reg;
    real<lower=0> sigma2_H;
}

transformed parameters {
    real sigma_reg_mode = get_scaled_inv_chi2_mode(10, 10);
    real<lower=0> sigma_reg_1 = sqrt(sigma2_reg-sigma_reg_mode);
    real<lower=0> sigma_H_1 = sqrt(sigma2_H-sigma_reg_mode);
}

model {
    // priors
    alpha ~ normal(1, 1000);
    beta ~ normal(1, 1000);
    HGp_s ~ normal(0, 10);
    c ~ normal(0, 100);
    sigma2_reg ~ scaled_inv_chi_square(10, 10);
    sigma2_H ~ scaled_inv_chi_square(10, 10);


    inv_a_recentre[2:N_tot] ~ normal(beta+alpha * inv_a_recentre[1:(N_tot-1)], sigma_reg_1);
    HGp_s ~ normal((inv_a_recentre-c)./t, sigma_H_1);
}