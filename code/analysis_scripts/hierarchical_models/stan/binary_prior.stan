functions {
    real gamma_alpha(real mu, real sigma2){
        return pow(mu, 2.0) * inv(sigma2);
    }

    real gamma_beta(real mu, real sigma2){
        return mu * inv(sigma2);
    }

    real beta_alpha(real mu, real sigma2){
        return ((mu - pow(mu, 2.0))*inv(sigma2) - 1.0) * mu;
    }

    real beta_beta(real mu, real sigma2){
        return ((mu - pow(mu, 2.0))*inv(sigma2) - 1.0) * (1.0 - mu);
    }
}

data {
    int<lower=0> N_child;
    // total mass
    vector<lower=0.0>[N_child] M;
    // reduced mass
    vector<lower=0.0>[N_child] M_reduced;
}

generated quantities {
    /****** parameters of the parent distributions ******/
    // semimajor axis at time binary is hard
    real a_hard;
    real a_hard_a;
    real a_hard_b;
    real a_hard_mu;
    real a_hard_sigma2;
    real a_hard_mu_a;
    real a_hard_mu_b;

    // eccentricity at time binary is hard
    real e_hard;
    real e_hard_a;
    real e_hard_b;
    real e_hard_mu;
    real e_hard_sigma2;
    real e_hard_mu_a;
    real e_hard_mu_b;

    // general scatter
    real sigma2;

    /****** forward folded values ******/
    array[N_child] real ang_mom;

    /****** sample the parameters ******/
    // semimajor axis
    a_hard_mu = gamma_rng(gamma_alpha(10, 100.0), gamma_beta(10, 100.0));
    a_hard_sigma2 = cauchy_rng(0.0, 10.0);
    //while(a_hard_sigma2 < 0.0){
    //    a_hard_sigma2 = cauchy_rng(0.0, 10.0);
    //}
    a_hard_a = gamma_alpha(a_hard_mu, a_hard_sigma2);
    a_hard_b = gamma_beta(a_hard_mu, a_hard_sigma2);
    a_hard = gamma_rng(a_hard_a, a_hard_b);

    // eccentricity
    e_hard_mu = beta_rng(beta_alpha(0.7, 0.2), beta_beta(0.7, 0.2));
    e_hard_sigma2 = cauchy_rng(0.0, 0.2);
    //while(e_hard_sigma2 < 0.0 || e_hard_sigma2 > e_hard_mu * (1.0 - e_hard_mu)){
    //    e_hard_sigma2 = cauchy_rng(0.0, 0.2);
    //}
    e_hard_a = beta_alpha(e_hard_mu, e_hard_sigma2);
    e_hard_b = beta_beta(e_hard_mu, e_hard_sigma2);
    e_hard = beta_rng(e_hard_a, e_hard_b);

    sigma2 = gamma_rng(10.0, 10.0);
    //while(sigma2 < 0.0){
    //    sigma2 = cauchy_rng(0.0, 10.0);
    //}

    ang_mom = normal_rng(M_reduced .* sqrt(M .* a_hard * (1.0 - pow(e_hard, 2.0))), sqrt(sigma2));

}