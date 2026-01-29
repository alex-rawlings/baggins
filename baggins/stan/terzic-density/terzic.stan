functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}


data {
    int<lower=1> N;
    vector<lower=0>[N] r;
    vector<lower=0>[N] density;

    // OOS points
    int<lower=1> N_OOS;
    vector<lower=0, upper=max(r)>[N_OOS] r_OOS;
}


transformed data {
    vector[N] log10_density = log10(density);
    int N_GQ = N + N_OOS;
    vector<lower=0, upper=max(r)>[N_GQ] r_GQ = append_row(r, r_OOS);
}


parameters {
    real<upper=1> log10rb;
    real<upper=1.5> log10Re;
    real<lower=-5, upper=15> log10rhob;
    real<lower=-2, upper=2> g;
    real<lower=0, upper=20> n;
    real<lower=0, upper=15> a;

    // model variance, function of radius
    real<lower=0> err;
}


transformed parameters {
    array[7] real lprior;
    lprior[1] = lognormal_lpdf(log10rb | 0, 1.);
    lprior[2] = lognormal_lpdf(log10Re | 0, 1);
    lprior[3] = normal_lpdf(n | 4., 2.);
    lprior[4] = normal_lpdf(g | 0., 0.5);
    lprior[5] = normal_lpdf(log10rhob | 3., 1.);
    lprior[6] = rayleigh_lpdf(a | 0.5);
    lprior[7] = normal_lpdf(err | 0., 1.);

    real rb = pow(10., log10rb);
    real Re = pow(10., log10Re);
}


model{
    // density at model parameters
    target += sum(lprior);

    // likelihood
    {
        real b = sersic_b_parameter(n);
        real p = p_parameter(n);
        real pt = terzic_preterm(g, a, rb, Re, n, b, p);
        target += normal_lpdf(log10_density | terzic_density_vec(r, pt, g, a, rb, n, b, p, Re, log10rhob), err);
    }
}


generated quantities {
    // generate data replication
    vector[N_GQ] log10_rho_posterior;
    vector[N_GQ] rho_posterior;

    {
        // some helper quantities
        real b = sersic_b_parameter(n);
        real p = p_parameter(n);
        real pt = terzic_preterm(g, a, rb, Re, n, b, p);

        // push forward data
        vector[N_GQ] mean_gsd = terzic_density_vec(r_GQ, pt, g, a, rb, n, b, p, Re, log10rhob);
        for(i in 1:N_GQ){
            log10_rho_posterior[i] = trunc_normal_rng(mean_gsd[i], err, -5, 15);
        }
    }
    rho_posterior = pow(10., log10_rho_posterior);
}