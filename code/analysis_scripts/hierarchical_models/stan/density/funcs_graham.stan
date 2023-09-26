/*
Calculate the Sersic b parameter

Parameters
----------
n: sersic index

Returns
-------
b value (approximate)
*/
real sersic_b_parameter(real n){
    return 2.0 * n - 0.33333333 + 0.009876 * inv(n);
}


/*
Calculate log10 of core-Sersic profile as a function of radius

Parameters
----------
N: number of radial bins
R: radii to evaluate at
log_I_b: natural log of intensity at break radius
g: slope index of inner core
a: transition index
r_b: break radius
Re: effective radius
n: sersic index

Returns
-------
log10_I_arr: log10 of profile
*/
array[] real log10_I(int N, array[] real R, real log10_I_b, real g, real a, real r_b, real Re, real n){
    real temp_val;
    real b = sersic_b_parameter(n);
    real inv_n = inv(n);
    real inv_a = inv(a);
    real denom = log(10.0);
    array[N] real log10_I_arr;
    //
    real pre_term = - g/a*log(2.0) + b * pow((pow(2.0, inv_a)*r_b/Re), inv_n);

    for(i in 1:N){
        temp_val = pre_term + g/a*log(pow(R[i],a) + pow(r_b,a)) - a*log(R[i]) - b * pow(Re, -inv_n) * pow(pow(R[i],a) + pow(r_b,a), inv_a*inv_n);
        // change base to log10
        log10_I_arr[i] = log10_I_b + temp_val / denom;

    }
    return log10_I_arr;
}



real graham_preterm(real g, real a, real n, real b, real r_b, real Re){
    return - g/a*log(2.0) + b * pow((pow(2.0, inv(a))*r_b/Re), inv(n));
}

real graham_surf_density(real R, real preterm, real g, real a, real r_b, real n, real b, real Re, real log10_I_b){
    real inv_n = inv(n);
    return log10_I_b + (preterm + g/a*log(pow(R,a) + pow(r_b,a)) - a*log(R) - b * pow(Re, -inv_n) * pow((pow(R,a) + pow(r_b,a)), (inv(a)*inv_n))) / log(10.0);
}



real core_radius(real vk, real r0, real p, real q){
    return r0 * (p * pow(vk, q) + 1);
}



real trunc_norm_rng(real mu, real sigma, real low, real up){
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
    real y = mu + sigma * inv_Phi(u);
    return y;
}



real lower_trunc_norm_rng(real mu, real sigma, real low){
    real y = trunc_norm_rng(mu, sigma, low, positive_infinity());
    return y;
}

