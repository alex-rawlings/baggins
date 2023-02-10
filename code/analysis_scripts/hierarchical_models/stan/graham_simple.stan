functions{
    #include funcs_graham.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    array[N_tot] real<lower=0.0> R;
    // array of surface density values
    array[N_tot] real<lower=0.0> log10_surf_rho;
}


parameters {
    // no hierarchy: assume all observations from the same set
    real<lower=0.0> r_b;
    real<lower=0.0> Re;
    real<lower=0.0> I_b;
    real<lower=0.0> g;
    real<lower=0.0> n;
    real<lower=0.0> a;
    
    // model variance: same for all
    real<lower=0.0> err;
}


transformed parameters {
    array[N_tot] real log10_surf_rho_calc;

    log10_surf_rho_calc = log10_I(N_tot, R, I_b, g, a, r_b, Re, n);
}


model{
    // density at model parameters
    target += normal_lpdf(r_b | 0.0, 2.0);
    target += normal_lpdf(Re | 0.0, 14.0);
    target += normal_lpdf(I_b | 0.0, 10.0);
    target += normal_lpdf(g | 0.0, 0.5);
    target += normal_lpdf(n | 0.0, 8.0);
    target += normal_lpdf(a | 0.0, 20.0);

    // density at error
    target += normal_lpdf(err | 0.0, 1.0);

    // likelihood
    target += normal_lpdf(log10_surf_rho | log10_surf_rho_calc, err);


}


generated quantities {
    // posterior predictive: for loop to ensure each radial bin positive with
    // rejection sampling
    array[N_tot] real log10_surf_rho_posterior;
    array[1] real Ri;
    array[1] real Ii;
    for(i in 1:N_tot){
        Ri[1] = R[i];
        Ii = normal_rng(log10_I(1, Ri, I_b, g, a, r_b, Re, n), err);
        while(log10_surf_rho_posterior[i] < 0.0){
            Ii = normal_rng(log10_I(1, Ri, I_b, g, a, r_b, Re, n), err);
        }
        log10_surf_rho_posterior[i] = Ii[1];
    }

    /****** determine log likelihood function ******/
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_calc[i], err);
    }
}