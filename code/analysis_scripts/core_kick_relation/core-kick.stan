functions {
    vector relation(vector x, real nu, real mu, real sigma, real k, real b, real c){
        int N = size(x);
        vector[N] term1;
        vector[N] term2;
        vector[N] sigmoidfun;
        term1 = sqrt((nu-2.)./nu) .* pow(1. + pow(((x - mu) ./ (sigma .* sqrt((nu-2.)./nu))), 2.), (-(nu + 1)./2));
        term2 = sqrt(nu./(nu-2.)) .* pow(1. + pow(((x - mu) ./ (sigma .* sqrt(nu./(nu-2.)))), 2.), (-(nu + 1)./2));
        sigmoidfun  = k ./ (1 + exp(-b .* (x-mu)));
        return (term1 + term2) ./ (2 * pi() .* sigma) - sigmoidfun + c;
    }

    real partial_sum(array[] real y_slice, int start, int end, vector vk, real nu, real mu, real sigma, real k, real b, real c, real err){
        return normal_lpdf( y_slice | relation(vk[start:end], nu, mu, sigma, k, b, c), err);
    }

    #include "../hierarchical_models/stan/custom_rngs.stan"
}


data {
    // total number of points
    int<lower=1> N_tot;
    // array of observed velocity values in units of v_esc
    vector<lower=0>[N_tot] vkick;
    // array of observed core radii in units of rb0
    vector<lower=0>[N_tot] rb;

    // Out of Sample points
    // follows same structure as above
    // total number of OOS points
    int<lower=1> N_OOS;
    // OOS kick velocities
    vector<lower=0>[N_OOS] vkick_OOS;
}


transformed data {
    int N_GQ = N_tot + N_OOS;
    vector<lower=0>[N_GQ] vkick_GQ = append_row(vkick, vkick_OOS);
}


parameters {
    real<lower=0.> mu;
    real<lower=0.> nu;
    real<lower=0.> sigma;
    real<lower=0.> k;
    real<upper=0.> b;
    real<lower=0.> c;
    real<lower=0.> err;
}


transformed parameters {
    array[7] real lprior;
    lprior[1] = normal_lpdf(mu | 0.3, 0.2);
    lprior[2] = normal_lpdf(nu | 5, 10);
    lprior[3] = normal_lpdf(sigma | 0, 0.5);
    lprior[4] = normal_lpdf(k | 1, 0.4);
    lprior[5] = normal_lpdf(b | -30, 20);
    lprior[6] = normal_lpdf(c | 1, 2);
    lprior[7] = normal_lpdf(err | 0, 9);
}


model {
    // density at priors
    target += sum(lprior);

    target += reduce_sum(partial_sum, to_array_1d(rb), 1, vkick, nu, mu, sigma, k, b, c, err);

}


generated quantities {
    // generate data replication
    vector[N_GQ] rb_posterior;
    vector[N_GQ] rel_mean = relation(vkick_GQ, nu, mu, sigma, k, b, c);
    for(i in 1:N_GQ){
        rb_posterior[i] = lower_trunc_normal_rng(rel_mean[i], err, 0.);
    }
}