import numpy as np
import pathlib as pl

def get_mean_stddev(f: str) -> tuple[float, float]:
    with open(f) as file:
        lines = file.readlines()
        data = [float(res.split()[4]) for res in lines]
        return (np.mean(data), np.std(data))


files = [str(f) for f in pl.Path(".").iterdir() if f.suffix == ".log"]
for f in files:
    print(f)
    r = get_mean_stddev(f)
    print("{:.3f} +- {:.3f}".format(r[0], r[1]))

