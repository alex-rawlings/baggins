/*
Integrand of semimajor axis integral
@param t: unscale time value
@param xc:
@param theta: arguments to intregrand function
@param x_r:
@param x_i:
@return: integrand value
*/
real a_integrand(real t, real xc, array[] real theta, array[] real x_r, array[] int x_i){
        return 1/(theta[1] * t + theta[2]);
    }
