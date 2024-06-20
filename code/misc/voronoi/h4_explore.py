import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import baggins as bgs


x = np.linspace(-5, 5, 500)
h4s = np.linspace(-0.5, 0.5, 21)

cmapper, sm = bgs.plotting.create_normed_colours(h4s[0], h4s[-1], cmap="icefire")

for h4 in tqdm(h4s):
    y = bgs.analysis.gauss_hermite_function(x, 0, 1, 0, h4)
    plt.plot(x, y, c=cmapper(h4), lw=2)
plt.colorbar(sm, ax=plt.gca(), label="h4")
plt.xlabel("LOS velocity")
plt.ylabel("PDF")
bgs.plotting.savefig("h4_vary.png")
plt.show()