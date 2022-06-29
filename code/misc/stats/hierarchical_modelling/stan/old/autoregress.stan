data{
    int<lower=0> N_tot;
    vector[N_tot] inv_a;
}

transformed data {
    vector[N_tot] inv_a_recentre;
    inv_a_recentre = inv_a - mean(inv_a);
}

parameters{
    real alpha;
    real beta;
    real<lower=0> sigma;
}

model {
    inv_a_recentre[2:N_tot] ~ normal(beta+alpha * inv_a_recentre[1:(N_tot-1)], sigma);
}