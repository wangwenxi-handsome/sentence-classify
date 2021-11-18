import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np


labels = np.load("product/experiments/sentence1/results.npy", allow_pickle = True).tolist()
raw_data = pd.read_csv("product/data/test-5000.csv")
raw_data["label"] = labels
raw_data.to_csv("product/experiments/sentence1/test-5000-labeled.csv", encoding = "utf_8_sig")