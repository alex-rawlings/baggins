/*
Define the Quinlan hardening relation
@param: t: independent variable, time
@param: grad: gradient of the linear relation
@param: intercept: y-intercept of the linear relation
@return: inverse semimajor axis
*/
quinlan_relation(t, grad, intercept){
    return grad .* t + intercept;
}

