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
}

data {
    int N_c;    // number of child runs
    int N_tot;    // sum(N_c x obs/child), [total number of ALL observations]
    array[N_tot] int<lower=0> child_id;   // id of the child run
    vector[N_tot] t;    // observed time
    vector[N_tot] inv_a;    // observed inverse semimajor axis
    vector[N_tot] t_sigma;    // observation error in time
    vector[N_tot] inv_a_sigma;   // observation error in 1/a
}


transformed data {
    vector[2] HGp_s_nu_tau;
    vector[2] intercept_nu_tau;

    HGp_s_nu_tau = get_scaled_inv_chi2_nu_tau(1, 50);
    intercept_nu_tau = get_scaled_inv_chi2_nu_tau(1, 50);
}


parameters {
    // parameters of the distributions for the linear fitting parameters
    // gradient
    real HGp_s_mu;
    real<lower=0> HGp_s_tau;
    // intercept
    real intercept_mu;
    real<lower=0> intercept_tau;

    // regression parameters for each child
    vector[N_c] HGp_s;
    vector[N_c] intercept;
}

transformed parameters {
    vector[N_tot] inv_a_true;

    // inv-a -- t relation
    inv_a_true = HGp_s[child_id] .* t + intercept[child_id];
}

model {
    // priors
    HGp_s_mu ~ normal(0, 10);
    HGp_s_tau ~ scaled_inv_chi_square(HGp_s_nu_tau[1], HGp_s_nu_tau[2]);
    intercept_mu ~ normal(0, 100);
    intercept_tau ~ scaled_inv_chi_square(intercept_nu_tau[1], intercept_nu_tau[2]);

    //t_true ~ normal(t, t_sigma);

    // connection to latent parameters
    HGp_s ~ normal(HGp_s_mu, HGp_s_tau);
    intercept ~ normal(intercept_mu, intercept_tau);

    // connection to data
    inv_a ~ normal(inv_a_true, inv_a_sigma);
}