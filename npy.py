import numpy as np

for i in range(9,10):
    print(i)
    data1 = np.load(f"E:\\RealMAN\\test_gen\\vad_sample_00000{i}.npy")
    print(data1)
    data2 = np.load(f"E:\\RealMAN\\test_gen\\targets_sample_00000{i}.npy")
    print(data2)
    data3 = data1.copy()
    data3[data2 == 0] = 0
    print(data3)