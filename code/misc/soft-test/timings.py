from os import linesep
import numpy as np
import matplotlib.pyplot as plt


files = [
    "/scratch/pjohanss/arawling/collisionless_merger/softening-test/0-0035/output/info.txt",
    "/scratch/pjohanss/arawling/collisionless_merger/softening-test/0-00475/output/info.txt",
    "/scratch/pjohanss/arawling/collisionless_merger/softening-test/0-006/output/info.txt"
]

fig, ax = plt.subplots(1,1, figsize=(5,3))
for ind, this_file in enumerate(files):
    with open(this_file, 'r') as f:
        #get the number of lines
        num_lines = int(sum(1 for _ in f)/2)+1
    with open(this_file, "r") as f:
        times = np.full(num_lines, np.nan)
        timesteps = np.full(num_lines, np.nan)
        #extract text after Time and Systemstep
        counter = 0
        for line in f:
            print("Completed {:.3f}%".format(((counter+1)/num_lines)*100), end='\r')
            if line == "\n":
                continue
            line_split = line.split(" ")
            times[counter] = float(line_split[3].rstrip(","))
            timesteps[counter] = float(line_split[8])
            counter += 1
        print("Complete file {}                                 ".format(ind))
    plt.plot(times, timesteps, label=this_file.split("/")[-3])
ax.set_xlabel("Time")
ax.set_ylabel("Timestep")
plt.legend()
plt.tight_layout()
plt.savefig("/users/arawling/figures/soft-test/timestep.png", dpi=300)
plt.show()