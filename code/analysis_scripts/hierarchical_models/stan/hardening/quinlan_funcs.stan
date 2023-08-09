real quinlan_inva(real t, real HGps, real inva0){
    return HGps * t + inva0;
}


real quinlan_e(real inva, real K, real inva0, real e0){
    return K * log(inva/inva0) + e0;
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