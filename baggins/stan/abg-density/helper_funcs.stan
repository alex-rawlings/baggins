vector abg_density_vec(
    vector r,
    real log10rhoS,
    real log10rS,
    real log10a,
    real b,
    real g
){
    real a = pow(10, log10a);
    // work in log10(x) throughout to avoid pow(x, a) overflow
    vector[size(r)] log10x = log10(fmax(r / pow(10, log10rS), 1e-20));

    // log10(1 + x^a): stable via log1p in natural log
    // When a*log(x) > 20, log(1+x^a) ~= a*log(x) (asymptote, avoids exp overflow)
    // When a*log(x) < -20, log(1+x^a) ~= x^a      (avoids catastrophic cancellation)
    vector[size(r)] log10_1pxa;
    for (i in 1:size(r)) {
        real exponent = a * log10x[i] * log(10);   // = a * ln(x)
        if (exponent > 20)
            log10_1pxa[i] = a * log10x[i];          // asymptote: log10(x^a)
        else
            log10_1pxa[i] = log1p(exp(exponent)) / log(10);
    }

    return log10rhoS - g * log10x + (g - b) / a * log10_1pxa;
}


vector abg_density_vec(
    vector r,
    vector log10rhoS,
    vector log10rS,
    vector log10a,
    vector b,
    vector g
){
    vector[size(log10a)] a = pow(10, log10a);
    // same stabilisation for the vectorised overload
    vector[size(r)] log10x = log10(fmax(r ./ pow(10, log10rS), 1e-20));

    vector[size(r)] log10_1pxa;
    for (i in 1:size(r)) {
        real exponent = a[i] * log10x[i] * log(10);
        if (exponent > 20)
            log10_1pxa[i] = a[i] * log10x[i];
        else
            log10_1pxa[i] = log1p(exp(exponent)) / log(10);
    }

    return log10rhoS - g .* log10x + (g - b) ./ a .* log10_1pxa;
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10a, vector b, vector g, vector s, array[] int cidx){
    return normal_lpdf(y_slice | abg_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10a[cidx[start:end]],
                    b[cidx[start:end]],
                    g[cidx[start:end]]),
                    s[start:end]);
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10a, vector b, vector g, real s, array[] int cidx){
    return normal_lpdf(y_slice | abg_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10a[cidx[start:end]],
                    b[cidx[start:end]],
                    g[cidx[start:end]]),
                    s);
}