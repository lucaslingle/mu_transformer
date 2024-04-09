import argparse
import math

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--group")
args = parser.parse_args()


df = pd.read_csv("main_results.csv", sep="\t", header=0)
df = df[df["Group"] == args.group]
df["log2_LR"] = df["LR"].map(lambda y: int(math.log2(y)))

smallmin = float("inf")
mediummin = float("inf")
largemin = float("inf")
smallargmin = None
mediumargmin = None
largeargmin = None

lines = []
for i, width in enumerate([128, 2048, 512], 1):
    if i == 3:
        s = args.group[0].upper() + args.group[1:].lower()
    else:
        s = ""
    s += f" & {width}"
    for log2lr in [-10, -8, -6, -4, -2]:
        val = df["Loss"][(df["Width"] == width) & (df["log2_LR"] == log2lr)]
        val = val.to_numpy()[0]
        s += " & {0:.3f}".format(val)
        if width == 128:
            smallmin = min(smallmin, val)
            if smallmin == val:
                smallargmin = log2lr
        if width == 512:
            mediummin = min(mediummin, val)
            if mediummin == val:
                mediumargmin = log2lr
        if width == 2048:
            largemin = min(largemin, val)
            if largemin == val:
                largeargmin = log2lr
    s += " & "
    if (i == 3) and (smallmin < float("inf")):
        if smallargmin == mediumargmin == largeargmin:
            s += " Yes "
        else:
            s += " No "
    s += " \\\\"
    lines.append(s)

print(lines[0])
print(lines[2])
print(lines[1])
print("\\midrule")
