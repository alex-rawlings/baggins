import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Count columns per line in file")
parser.add_argument(help="file", type=str, dest="file")
parser.add_argument("-s", "--sep", type=str, help="separator", dest="sep", default=",")
parser.add_argument("-j", "--jump", type=int, help="jump this many lines from start", dest="jump", default=0)
args = parser.parse_args()


count = 0
colcount = []
with open(args.file, "r") as f:
    while True:
        count += 1
        line  = f.readline()
        if count < args.jump: continue
        if not line:
            break
        colcount.append(line.count(args.sep))

print(f"Unique column lengths: {np.unique(colcount)}")

plt.plot(np.arange(len(colcount))+args.jump, colcount)
plt.show()