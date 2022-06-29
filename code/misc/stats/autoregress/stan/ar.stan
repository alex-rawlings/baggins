data{
    // number of observations
    int<lower=2> N_tot;
    // data
    vector[N_tot] observed_data;
}

parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}

transformed parameters {
    vector[(N_tot-1)] mu = alpha + beta * observed_data[1:(N_tot-1)];
}

model {
    // priors
    mu ~ normal(1, 10);

    observed_data[2:N_tot] ~ normal(mu, sigma);
}

generated quantities {
    array[(N_tot-1)] real posterior_pred = normal_rng(mu, sigma);
}