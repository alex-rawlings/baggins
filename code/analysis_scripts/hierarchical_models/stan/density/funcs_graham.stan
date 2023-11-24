/*
Calculate the Sersic b parameter

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
/*array[] real log10_I(int N, array[] real R, real log10_I_b, real g, real a, real r_b, real Re, real n){
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
*/


vector graham_preterm(vector g, vector a, vector n, vector b, vector r_b, vector Re){
    return - g./a*log(2.0) + b .* pow((pow(2.0, inv(a)).*r_b./Re), inv(n));
}

real graham_surf_density(real R, real preterm, real g, real a, real r_b, real n, real b, real Re, real log10_I_b){
    real inv_n = inv(n);
    return log10_I_b + (preterm + g/a*log(pow(R,a) + pow(r_b,a)) - g*log(R) - b * pow(Re, -inv_n) * pow((pow(R,a) + pow(r_b,a)), (inv(a)*inv_n))) / log(10.0);
}


vector graham_surf_density_vec(
                             vector R,
                             vector preterm,
                             vector g,
                             vector a,
                             vector rb,
                             vector n,
                             vector b,
                             vector Re,
                             vector log10densb){
    return log10densb + (preterm + g./a.*log(pow(R,a) + pow(rb,a)) - g.*log(R) - b .* pow(Re, -inv(n)) .* pow((pow(R,a) + pow(rb,a)), inv(a.*n))) ./ log(10.0);
}


// TODO parse each array used in log_rho_calc to this function??
real partial_sum_factor(array[] real y_slice, int start, int end, int nc, vector R, vector g, vector a, vector rb, vector n, vector re, vector ld, vector s, array[] int fidx, array[] int cidx){
    vector[nc] b = sersic_b_parameter(n);
    vector[nc] pt = graham_preterm(g, a, n, b, rb, re);

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



real partial_sum_hierarchy(array[] real y_slice, int start, int end, int nc, vector R, vector g, vector a, vector rb, vector n, vector re, vector ld, vector s, array[] int cidx){
    vector[nc] b = sersic_b_parameter(n);
    vector[nc] pt = graham_preterm(g, a, n, b, rb, re);

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
                    s[start:end]);
}

