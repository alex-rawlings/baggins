vector quinlan_ODE(real t, vector inva, real const){
    vector[1] dinva_dt;
    dinva_dt[1] = const * t;
    return dinva_dt;
}


real quinlan_inva(real t, real slope, real const){
    return slope * t + const;
}