/*
RNG for a lower-bounded normal variate
TODO add support for upper, upper and lower truncations

Parameters
----------
mu: location parameter (mean)
sigma: scale parameter (standard deviation)
t: lower trunction boundary

Return
------
truncated normal variate
*/
real LB_normal_rng(real mu, real sigma, real t){
    if (is_nan(mu) || is_inf(mu)){
        reject("normal_rng_LB: mu must be finite; ", 
        "found mu = ", mu);
    }
    if (is_nan(sigma) || is_inf(sigma) || sigma < 0){
        reject("normal_rng_LB: sigma must be finite and non-negative; ",
        "found sigma = ", sigma);
    }
    // cdf for lower bound
    real p = normal_cdf(t | mu, sigma);
    real u = uniform_rng(p, 1.0);
    // protect against extreme quantiles where inv_Phi breaks down
    while (u < 1e-16 || u > 0.999999999){
        u = uniform_rng(p, 1.0);
    }
    // sample from inverse cdf
    return inv_Phi(u);

}