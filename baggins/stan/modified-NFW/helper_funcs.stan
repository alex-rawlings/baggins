vector mNFW_density_vec(
    vector r,
    real log10rhoS,
    real log10rS,
    real log10g
){
    vector[size(r)] x = r ./ pow(10, log10rS);
    real g = pow(10, log10g);
    return log10rhoS - g * log10(x) + (g - 3.) * log10(1 + x);
}


vector mNFW_density_vec(
    vector r,
    vector log10rhoS,
    vector log10rS,
    vector log10g
){
    vector[size(r)] x = r ./ pow(10, log10rS);
    vector[size(log10g)] g = pow(10, log10g);
    return log10rhoS - g .* log10(x) + (g - 3.) .* log10(1 + x);
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10g, vector s, array[] int cidx){
    return normal_lpdf(y_slice | mNFW_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10g[cidx[start:end]]),
                    s[start:end]);
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10g, real s, array[] int cidx){
    return normal_lpdf(y_slice | mNFW_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10g[cidx[start:end]]),
                    s);
}