import os
import sys
sys.path.append(os.getcwd())
import csv
import numpy as np


reader = csv.reader(open("product/data/test-5000.csv", "r"))
sentences = []

for i, item in enumerate(reader):
    if i == 0:
        pass
    else:
        sentences.append(item[1])

data = {"x": sentences}
np.save("product/data/pre_data.npy", data, allow_pickle = True)
print("data_length", i)