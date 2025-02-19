import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
from baggins.general import sersic_b_param


data = np.loadtxt("mag_data.txt")
data = np.log(data)

xcen = int(data.shape[0]/2)
ycen = int(data.shape[1]/2)

XX, YY = np.meshgrid(np.arange(data.shape[0])-xcen, np.arange(data.shape[1])-ycen)

R = np.sqrt(XX**2 + YY**2)

def log_sersic(R, logI0, Re, n):
    return logI0 - sersic_b_param(n) * ((R/Re)**(1/n) - 1)


# fit the params
params, pcov, *_ = so.curve_fit(log_sersic, R.ravel(), data.ravel(), bounds=([0, 0, 0], [10, 10, 10]))

print(params)
print(pcov)

rs = np.linspace(R.min(), R.max(), 400)
plt.loglog(rs, np.exp(log_sersic(rs, *params)))
plt.scatter(R, np.exp(data), marker=".", c="tab:red")
plt.savefig("sersic.png", dpi=300)