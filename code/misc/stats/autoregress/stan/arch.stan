data{
    // number of observations
    int<lower=2> N_tot;
    // data
    vector[N_tot] observed_data;
}

parameters {
    real mu; // average return
    real<lower=0> alpha0; // noise intercept
    real<lower=0, upper=1> alpha1; // noise slope

}

transformed parameters {
    vector[(N_tot-1)] sigma = sqrt(alpha0 + alpha1 * pow(observed_data[1:(N_tot-1)] - mu, 2));
}

model {
    // priors
    mu ~ normal(1, 10);

    observed_data[2:N_tot] ~ normal(mu, sigma);
}

generated quantities {
    array[(N_tot-1)] real posterior_pred = normal_rng(mu, sigma);
}