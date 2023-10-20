/*
Truncated normal distribution RNG
Parameters
----------
mu : real
    location parameter
sigma : real
    scale parameter
low : real
    lower bound of truncation
up : real
    upper bound of truncation

Returns
-------
: real
    random variate in truncation region
*/
real trunc_normal_rng(real mu, real sigma, real low, real up){
    if(is_inf(abs(mu)) || is_inf(sigma)){
        reject("Location and scale arguments must be finite!");
    }
    if(sigma < 0){
        reject("Scale parameter must be positive!");
    }
    if(low > up){
        reject("Lower bound must be less than up bound!");
    }
    // handle instances where only one truncation bound is given
    real p_lb;
    real p_ub;
    if(is_inf(low)){
        p_lb = 0.0;
    }else{
        p_lb = normal_cdf(low | mu, sigma);
    }
    if(is_inf(up)){
        p_ub = 1.0;
    }else{
        p_ub = normal_cdf(up | mu, sigma);
    }
    real u;
    if(p_lb >= 1){
        u = 1;
    }else if (p_ub <= 0) {
        u = 0;
    }else{
        u = uniform_rng(p_lb, p_ub);
    }
    return mu + sigma * inv_Phi(u);
}


/*
Lower truncated normal distribution RNG
Parameters
----------
mu : real
    location parameter
sigma : real
    scale parameter
low : real
    lower bound of truncation

Returns
-------
: real
    random variate in truncation region
*/
real lower_trunc_normal_rng(real mu, real sigma, real low){
    return trunc_normal_rng(mu, sigma, low, positive_infinity());
}


/*
Truncated Weibull distribution RNG
Parameters
----------
a : real
    shape parameter
sigma : real
    scale parameter
low : real
    lower bound of truncation
up : real
    upper bound of truncation

Returns
-------
: real
    random variate in truncation region
*/
real trunc_weibull_rng(real a, real sigma, real low, real up){
    if(is_inf(abs(a)) || is_inf(sigma)){
        reject("SCale and shape arguments must be finite!");
    }
    if(a <= 0 || sigma <= 0){
        reject("Parameters must be positive!");
    }
    if(low > up){
        reject("Lower bound must be less than up bound!");
    }
    // handle instances where only one truncation bound is given
    real p_lb;
    real p_ub;
    if(is_inf(low)){
        p_lb = 0.0;
    }else{
        p_lb = weibull_cdf(low | a, sigma);
    }
    if(is_inf(up)){
        p_ub = 1.0;
    }else{
        p_ub = weibull_cdf(up | a, sigma);
    }
    real u;
    if(p_lb >= 1){
        u = 1;
    }else if (p_ub <= 0) {
        u = 0;
    }else{
        u = uniform_rng(p_lb, p_ub);
    }
    return sigma * (-log1m(u))^inv(a);
}


/*
Truncated Rayleigh distribution RNG
Parameters
----------
sigma : real
    scale parameter
low : real
    lower bound of truncation
up : real
    upper bound of truncation

Returns
-------
: real
    random variate in truncation region
*/
real trunc_rayleigh_rng(real sigma, real low, real up){
    if(is_inf(sigma)){
        reject("Shape arguments must be finite!");
    }
    if(sigma <= 0){
        reject("Parameter must be positive!");
    }
    if(low > up){
        reject("Lower bound must be less than up bound!");
    }
    // handle instances where only one truncation bound is given
    real p_lb;
    real p_ub;
    if(is_inf(low)){
        p_lb = 0.0;
    }else{
        p_lb = rayleigh_cdf(low | sigma);
    }
    if(is_inf(up)){
        p_ub = 1.0;
    }else{
        p_ub = rayleigh_cdf(up | sigma);
    }
    real u;
    if(p_lb >= 1){
        u = 1;
    }else if (p_ub <= 0) {
        u = 0;
    }else{
        u = uniform_rng(p_lb, p_ub);
    }
    return sigma * sqrt(-2 * log1m(u));
}


/*
Truncated exponential distribution RNG
Parameters
----------
lambda : real
    shape parameter
low : real
    lower bound of truncation
up : real
    upper bound of truncation

Returns
-------
: real
    random variate in truncation region
*/
real trunc_exponential_rng(real lambda, real low, real up){
    if(is_inf(lambda)){
        reject("Shape arguments must be finite!");
    }
    if(lambda <= 0){
        reject("Parameter must be positive!");
    }
    if(low > up){
        reject("Lower bound must be less than up bound!");
    }
    // handle instances where only one truncation bound is given
    real p_lb;
    real p_ub;
    if(is_inf(low)){
        p_lb = 0.0;
    }else{
        p_lb = exponential_cdf(low | lambda);
    }
    if(is_inf(up)){
        p_ub = 1.0;
    }else{
        p_ub = exponential_cdf(up | lambda);
    }
    real u;
    if(p_lb >= 1){
        u = 1;
    }else if (p_ub <= 0) {
        u = 0;
    }else{
        u = uniform_rng(p_lb, p_ub);
    }
    return -log1m(u)/lambda;
}


/*
Truncated Frechet distribution RNG
Parameters
----------
a : real
    shape parameter
sigma : real
    scale parameter
low : real
    lower bound of truncation
up : real
    upper bound of truncation

Returns
-------
: real
    random variate in truncation region
*/
real trunc_frechet_rng(real a, real sigma, real low, real up){
    real yw = trunc_weibull_rng(a, inv(sigma), low, up);
    return inv(yw);
}