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
I_b: intensity at break radius
g: slope index of inner core
a: transition index
r_b: break radius
Re: effective radius
n: sersic index

Returns
-------
log10_I_arr: log10 of profile
*/
array[] real log10_I(int N, array[] real R, real I_b, real g, real a, real r_b, real Re, real n){
    real temp_val;
    real b = sersic_b_parameter(n);
    real inv_n = inv(n);
    real denom = log(10.0);
    array[N] real log10_I_arr;
    
    real pre_term = log(I_b) - g/a*log(2.0) + b * pow((pow(2.0, inv(a))*r_b/Re), inv_n);

    for(i in 1:N){
        temp_val = pre_term + g/a*log(pow(R[i],a) + pow(r_b,a)) - a*log(R[i]) - b * pow(Re, -inv_n) * pow(pow(R[i],a) + pow(r_b,a), inv(a)*inv_n);
        // change base to log10
        log10_I_arr[i] = temp_val / denom;

    }
    return log10_I_arr;
}

