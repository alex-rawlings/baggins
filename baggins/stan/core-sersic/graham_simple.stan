functions{
    #include funcs_graham.stan
    #include ../custom_rngs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    vector<lower=0.001>[N_tot] R;
    // array of surface density values
    array[N_tot] real log10_surf_rho;

    // Out of Sample points
    // total number of OOS points
    int<lower=1> N_OOS;
    // OOS radii values
    vector<lower=min(R), upper=max(R)>[N_OOS] R_OOS;
}


transformed data {
    int N_GQ = N_tot + N_OOS;
    vector<lower=min(R), upper=max(R)>[N_GQ] R_GQ = append_row(R, R_OOS);
}


parameters {
    // no hierarchy: assume all observations from the same set
    real<lower=0, upper=5> rb;
    real<lower=0, upper=20> Re;
    real<lower=-5, upper=15> log10densb;
    real<lower=0, upper=1> g;
    real<lower=0, upper=20> n;
    real<lower=0, upper=15> a;

    // model variance, function of radius
    real<lower=0> err;
}


transformed parameters {
    array[7] real lprior;
    lprior[1] = rayleigh_lpdf(rb | 1.);
    lprior[2] = rayleigh_lpdf(Re | 8.);
    lprior[3] = normal_lpdf(n | 4., 1.);
    lprior[4] = exponential_lpdf(g | 2.);
    lprior[5] = normal_lpdf(log10densb | 10., 1.);
    lprior[6] = rayleigh_lpdf(a | 7.);
    lprior[7] = normal_lpdf(err | 0., 1.);
}


model{
    // density at model parameters
    target += sum(lprior);

    // likelihood
    {
        real b = sersic_b_parameter(n);
        real pt = graham_preterm(g, a, n, b, rb, Re);
        target += normal_lpdf(log10_surf_rho | graham_surf_density_vec(R, pt, g, a, rb, n, b, Re, log10densb), err);
    }
}


generated quantities {
    // generate data replication
    vector[N_GQ] log10_surf_rho_posterior;
    vector[N_GQ] surf_rho_posterior;

    {
        // some helper quantities
        real b = sersic_b_parameter(n);
        real pt = graham_preterm(g, a, n, b, rb, Re);

        // push forward data
        vector[N_GQ] mean_gsd = graham_surf_density_vec(
                                        R_GQ,
                                        pt,
                                        g,
                                        a,
                                        rb,
                                        n,
                                        b,
                                        Re,
                                        log10densb);
        for(i in 1:N_GQ){
            log10_surf_rho_posterior[i] = trunc_normal_rng(mean_gsd[i], err, -5, 15);
        }
    }
    surf_rho_posterior = pow(10., log10_surf_rho_posterior);
}
