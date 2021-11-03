import os
import sys
sys.path.append(os.getcwd())
import numpy as np


data = np.load("product/data/pre_data.npy", allow_pickle = True).tolist()
x = data["x"]
y = data["y"]
id = data["id"]
len_x = []
lg_512 = 0
lg_256 = 0
lg_128 = 0
lg_64 = 0
lg_32 = 0
for i in x:
    if len(i) > 10000:
        pass
    if len(i) > 512:
        lg_512 += 1
    if len(i) > 256:
        lg_256 += 1
    if len(i) > 128:
        lg_128 += 1
    if len(i) > 64:
        lg_64 += 1
    if len(i) > 32:
        lg_32 += 1
    len_x.append(len(i))

print(max(len_x))
print(">512", lg_512)
print(">256", lg_256)
print(">128", lg_128)
print(">64", lg_64)
print(">32", lg_32)

len_y = []
for l in y:
    len_y.append(len(l))
print("1 num", len_y.count(1))
print("2 num", len_y.count(2))
print("3 num", len_y.count(3))
print("length", len(x), len(y), len(id))