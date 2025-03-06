functions {
    real sersic_profile(real x, real y, real x0, real y0, real log_Re, real n, real q, real theta, real log_Ie) {
        real Re = exp(log_Re);
        
        real bn = 1.9992 * n - 0.3271;  // Approximation of bn for Sersic index

        // Coordinate transformation
        real x_prime = (x - x0) * cos(theta) + (y - y0) * sin(theta);
        real y_prime = -(x - x0) * sin(theta) + (y - y0) * cos(theta);

        real r = sqrt(x_prime^2 + (y_prime^2 / (q^2)));  // Elliptical radius
        return log_Ie - bn * (pow(r / Re, 1 / n) - 1);
    }

    real sersic_profile_boxy(real x, real y, real x0, real y0, real log_Re, real n, real q, real theta, real log_Ie, real b) {
        real Re = exp(log_Re);
        
        real bn = 1.9992 * n - 0.3271;  // Approximation of bn for Sersic index

        real Rt = sqrt((x-x0)^2 + (y-y0)^2);
        real thetat = atan2(x-x0, y-y0) + theta;
        real Rm = pow(
            pow(abs(Rt * sin(thetat) * q), 2 + b) +
            pow(abs(Rt * cos(thetat)), 2 + b),
            inv(2 + b)
        );
        return log_Ie - bn * (pow(Rt / Re, 1 / n) - 1);
    }
}

data {
    int<lower=1> N;  // Image height
    int<lower=1> M;  // Image width
    matrix[N, M] log_image;  // Logarithm of the image data
}

parameters {
    real log_Ie;  // Log effective intensity
    real<lower=1, upper=2.5> log_Re;  // Log effective radius
    real<lower=0, upper=15> n;   // Sersic index
    real<lower=0, upper=1> q;  // Axis ratio
    real<lower=0, upper=pi()> theta;  // Position angle in radians
    real<lower=1, upper=M> x0;  // X center
    real<lower=1, upper=N> y0;  // Y center
    real<lower=0> sigma;  // Noise standard deviation
}

transformed parameters {
    matrix[N, M] log_model_image;

    array[8] real lprior;
    lprior[1] = normal_lpdf(log_Ie | 0, 1);
    lprior[2] = rayleigh_lpdf(log_Re | 4);
    lprior[3] = normal_lpdf(n | 4, 4);
    lprior[4] = beta_lpdf(q | 2, 2);
    lprior[5] = uniform_lpdf(theta | 0, pi());
    lprior[6] = normal_lpdf(x0 | M/2, 2);
    lprior[7] = normal_lpdf(y0 | N/2, 2);
    lprior[8] = normal_lpdf(sigma | 0, 1);
    
    // Vectorized computation of the model
    for (i in 1:N) {
        for (j in 1:M) {
            log_model_image[i, j] = sersic_profile(j, i, x0, y0, log_Re, n, q, theta, log_Ie);
        }
    }
}

model {
    // Priors (weakly informative)
    target += sum(lprior);
    
    // Likelihood (in log space)
    for (j in 1:M) {
        for (i in 1:N) {
            target += normal_lpdf(log_image[i, j] | log_model_image[i, j], sigma);
        }
    }
}
