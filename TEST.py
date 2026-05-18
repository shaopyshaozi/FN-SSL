import os
import numpy as np
from tqdm import tqdm

root = r"E:\RealMAN\val_gen"

bad = []
for dirpath, _, files in os.walk(root):
    for f in tqdm(files):
        if f.endswith(".npy"):
            p = os.path.join(dirpath, f)
            try:
                x = np.load(p)
                if x.dtype == object:
                    bad.append(p)
            except ValueError as e:
                if "pickled" in str(e):
                    bad.append(p)

print("bad files:", len(bad))
for p in bad[:50]:
    print(p)