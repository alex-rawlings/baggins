import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter
from skimage.filters import gaussian
import baggins as bgs


data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/IFU_bt/ifu_bt_0600.pickle"

data = bgs.utils.load_data(data_file)

img = data["voronoi_stats_all"]["img_h4"][40:70,30:70]

sigma_blur = 0.5

circles = bgs.analysis.fit_circle_to_image(img, 1, sigma_blur=sigma_blur, hough_radii=np.arange(6, 12, 1), sigma_canny=4)

print(f"There are {len(circles)} circles found")
img = bgs.analysis.rescale_image(img)
img = gaussian(img, sigma_blur)

# Draw the detected inner circle
circle_colour = 1.5
for i, c in enumerate(circles):
    circ_y, circ_x = circle_perimeter(c.cy, c.cx, c.r)
    # Draw the circle with a different color intensity
    try:
        img[circ_y, circ_x] = circle_colour
    except IndexError:
        print("Circle goes over image edge")
        for _cy, _cx in zip(circ_y, circ_x):
            try:
                img[_cy, _cx] = circle_colour
            except IndexError:
                continue
    print(f"Radius of circle {i} is {c.r}")

fig, ax = plt.subplots()
p = ax.imshow(img, cmap="Reds", vmin=0, vmax=circle_colour)
plt.colorbar(p)
plt.show()