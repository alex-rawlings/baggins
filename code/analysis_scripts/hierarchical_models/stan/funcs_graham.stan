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
Calculate the Sersic p parameter

Parameters
----------
n: sersic index

Returns
-------
p value (approximate)
*/
real p_parameter(real n){
    return 1.0 - 0.6097*inv(n) + 0.05563*inv(square(n));
}


/*
Calculate normalising coefficient of core-Sersic model

Parameters
----------
I_b: intensity at break radius
g: slope index of inner core
a: transition index
r_b: break radius
Re: effective radius
n: sersic index

Returns
-------
normalising coefficient
*/
real I_dash(real I_b, real g, real a, real r_b, real Re, real n){
    real b = sersic_b_parameter(n);
    return I_b * pow(2.0, -g*inv(a)) * exp(b * pow((pow(2.0, inv(a)) * r_b/Re), inv(n)));
}


/*
Calculate core-Sersic profile as a function of radius

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
I_arr: profile
*/
array[] real I(int N, array[] real R, real I_b, real g, real a, real r_b, real Re, real n){
    real I_ = I_dash(I_b, g, a, r_b, Re, n);
    real b = sersic_b_parameter(n);
    array[N] real I_arr;

    for(i in 1:N){
        I_arr[i] = I_ * pow((1.0 + pow((r_b*inv(R[i])), a)), g*inv(a)) * exp(-b * pow(((pow(R[i], a) + pow(r_b, a)) / pow(Re, a)), inv(a * n)));
    }
    return I_arr;
}


/*
Calculate log10 of core-Sersic profile as a function of radius

Parameters
----------
N: number of radial bins
R: radii to evaluate at
I_b: intensity at break radius in normalised by 1e9
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
    real b = sersic_b_parameter(n);
    array[N] real log10_I_arr;
    real denom = log(10.0);
    real pre_term = log(I_b) - g/a*log(2.0) + b*pow((pow(2.0, inv(a))*r_b/Re), inv(n));

    for(i in 1:N){
        log10_I_arr[i] = pre_term + g/a * log(1.0 + pow((r_b/R[i]), a)) - b*pow(((pow(R[i], a) + pow(r_b, a)) / pow(Re, a)), inv(a*n));
        // change base to log10
        log10_I_arr[i] = log10_I_arr[i] / denom + 9.0;
    }
    return log10_I_arr;
}

