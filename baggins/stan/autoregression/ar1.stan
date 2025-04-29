data {
    int<lower=0> N;
    vector<lower=-1, upper=1>[N] y;
}

parameters {
    real a1;
    real a2;
    real<lower=0> sigma;
}

transformed parameters {
    array[3] real lprior;
    lprior[1] = normal_lpdf(a1 | 0, 1);
    lprior[2] = normal_lpdf(a2 | 0, 1);
    lprior[3] = normal_lpdf(sigma | 0, 1);
}

model {
    target += sum(lprior);
    target += normal_lpdf(y[2:N] | a1 + a2 * y[1:(N-1)], sigma);
}

generated quantities {
    array[N] real y_posterior;
    y_posterior[1] = y[1];
    y_posterior[2:N] = normal_rng(a1 + a2 * y[1:(N-1)], sigma);
}