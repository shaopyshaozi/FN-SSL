import os
import glob
import numpy as np

data_dir = r"E:\RealMAN\test_gen"

vad_files = sorted(glob.glob(os.path.join(data_dir, "vad_*.npy")))

count_files = 0
count_changed = 0

for vad_path in vad_files:
    base = os.path.basename(vad_path).replace("vad_", "")
    targets_path = os.path.join(data_dir, f"targets_{base}")

    if not os.path.exists(targets_path):
        print(f"Missing targets file for {vad_path}")
        continue

    vad = np.load(vad_path)
    targets = np.load(targets_path)

    old_vad = vad.copy()

    vad[targets == 0] = 0

    if not np.array_equal(vad, old_vad):
        np.save(vad_path, vad)
        count_changed += 1

    count_files += 1

print(f"Processed {count_files} files")
print(f"Changed {count_changed} vad files")