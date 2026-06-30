/*
Calculate the Sersic b parameter, vectorised

Parameters
----------
n: sersic index

Returns
-------
b value (approximate)
*/
vector sersic_b_parameter(vector n){
    return 2.0 * n - 0.33333333 + 0.009876 * inv(n);
}

/*
Calculated Sersic b parameter, serial
*/
real sersic_b_parameter(real n){
    return 2.0 * n - 0.33333333 + 0.009876 * inv(n);
}

/*
Calculate the radius-independent term of the core Sersic function, vectorised

Parameters
----------
g: inner slope
a: transition index
n: sersic index
rb: core radius
Re: effective radius

Returns
-------
pre term value
*/
vector graham_preterm(vector g, vector a, vector n, vector b, vector rb, vector Re){
    return - g./a*log(2.0) + b .* pow((pow(2.0, inv(a)).*rb./Re), inv(n));
}

/*
Calculate the radius-independent term of the core Sersic function, serial
*/
real graham_preterm(real g, real a, real n, real b, real rb, real Re){
    return - g/a*log(2.0) + b * pow((pow(2.0, inv(a))*rb/Re), inv(n));
}


/*
Calculate the core Sersic profile, vectorised over parameters

Parameters
----------
R: radius to evaluate at
preterm: radius-independent factor, output of graham_preterm()
g: inner slope
a: transition index
rb: core radius
n: sersic index
b: sersic b parameter
Re: effective radius
log10densb: density at core radius

Returns
-------
projected density at specified radial values
*/
vector graham_surf_density_vec(
                             vector R,
                             vector log10densb,
                             vector log10rb,
                             vector log10g,
                             vector log10n,
                             vector log10a,
                             vector log10Re,
                             ){
    int N = size(log10densb);
    vector[N] rb = pow(10., log10rb);
    vector[N] g = pow(10., log10g);
    vector[N] n = pow(10., log10n);
    vector[N] a = pow(10., log10a);
    vector[N] Re = pow(10., log10Re);
    vector[N] b = sersic_b_parameter(n);
    vector[N] preterm = graham_preterm(g, a, n, b, rb, Re);
    return log10densb + (preterm + g./a.*log(pow(R,a) + pow(rb,a)) - g.*log(R) - b .* pow(Re, -inv(n)) .* pow((pow(R,a) + pow(rb,a)), inv(a.*n))) ./ log(10.0);
}


/*
Calculate the core Sersic profile, serial over parameters
*/
vector graham_surf_density_vec(
                             vector R,
                             real log10densb,
                             real log10rb,
                             real log10g,
                             real log10n,
                             real log10a,
                             real log10Re){
    real rb = pow(10., log10rb);
    real g = pow(10., log10g);
    real n = pow(10., log10n);
    real a = pow(10., log10a);
    real Re = pow(10., log10Re);
    real b = sersic_b_parameter(n);
    real preterm = graham_preterm(g, a, n, b, rb, Re);
    return log10densb + (preterm + g/a*log(pow(R,a) + pow(rb,a)) - g*log(R) - b * pow(Re, -inv(n)) * pow((pow(R,a) + pow(rb,a)), inv(a*n))) / log(10.0);
}

/*
Partial summation function for reduce_sum() capability

Parameters
----------
yslice: observed projected density values
start: start index for partial sum
end: end index for partial sum
nc: number of contexts (groups) in the hierarchy
R: radius to evaluate at
g: inner slope
a: transition index
rb: core radius
n: sersic index
re: effective radius
ld: density at core radius
s: standard deviation for projected density
fidx: factor index
cidx: context index

Returns
-------
likelihood evaluation at radius R
*/
real partial_sum_factor(array[] real y_slice, int start, int end, vector R, vector g, vector a, vector rb, vector n, vector re, vector ld, vector s, array[] int fidx, array[] int cidx){
    vector[size(n)] b = sersic_b_parameter(n);
    vector[size(n)] pt = graham_preterm(g, a, n, b, rb, re);

    return normal_lpdf(y_slice | graham_surf_density_vec(
                    R[start:end],
                    pt[cidx[start:end]],
                    g[cidx[start:end]],
                    a[cidx[start:end]],
                    rb[cidx[start:end]],
                    n[cidx[start:end]],
                    b[cidx[start:end]],
                    re[cidx[start:end]],
                    ld[cidx[start:end]]),
                    s[fidx[cidx[start:end]]]);
}


/*
Partial summation function for reduce_sum() capability

Parameters
----------
yslice: observed projected density values
start: start index for partial sum
end: end index for partial sum
R: radius to evaluate at
log10densb: logarithmic break density
log10rb: logarithmic break radius
log10g: logarithmic inner slope
log10n: logarithmic Sersic index
log10a: logarithmic transition index
log10Re: logarithmic effective radius
s: error for projected density
cidx: context index

Returns
-------
likelihood evaluation at radius R
*/
real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector R, vector log10densb, vector log10rb, vector log10g, vector log10n, vector log10a, vector log10Re, vector s, array[] int cidx){
    return normal_lpdf(y_slice | graham_surf_density_vec(
                    R[start:end],
                    log10densb[cidx[start:end]],
                    log10rb[cidx[start:end]],
                    log10g[cidx[start:end]],
                    log10n[cidx[start:end]],
                    log10a[cidx[start:end]],
                    log10Re[cidx[start:end]]),
                    s);
}
