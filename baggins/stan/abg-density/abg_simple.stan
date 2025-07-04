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
    real<lower=0> rS;
    real<lower=0> a;
    real<lower=0> b;
    real<lower=0> g;
    real<lower=-5, upper=15> log10rhoS;
    real<lower=0> err;
}


transformed parameters {
    array[6] real lprior;
    lprior[1] = rayleigh_lpdf(rS | 1);
    lprior[2] = normal_lpdf(a | 0, 8);
    lprior[3] = normal_lpdf(b | 0, 8);
    lprior[4] = normal_lpdf(g | 0, 8);
    lprior[5] = normal_lpdf(log10rhoS | 3, 2);
    lprior[6] = normal_lpdf(err | 0, 1);
}


model {
    target += sum(lprior);
    target += normal_lpdf(log10_density | abg_density_vec(r, log10rhoS, rS, a, b, g), err);
}


generated quantities {
    // generate data replication
    vector[N_GQ] log10_rho_posterior;
    vector[N_GQ] rho_posterior;

    // push forward data
    vector[N_GQ] mean_gsd = abg_density_vec(r_GQ, log10rhoS, rS, a, b, g);
    for(i in 1:N_GQ){
        log10_rho_posterior[i] = trunc_normal_rng(mean_gsd[i], err, -5, 15);
    }

    rho_posterior = pow(10., log10_rho_posterior);
}