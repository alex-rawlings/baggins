real quinlan_inva(real t, real HGps, real inva0){
    return HGps * t + inva0;
}

real quinlan_e(real inva, real K, real inva0, real e0){
    return K * log(inva/inva0) + e0;
}

