import os
import sys
sys.path.append(os.getcwd())
import csv
import numpy as np


reader = csv.reader(open("product/data/raw_data.csv", "r"))
sentences = []
labels = []
ids = []
lack_labels_id = []
for i, item in enumerate(reader):
    if i == 0:
        tag = item
        tag2id = dict(zip(item, range(len(item))))
    else:
        if isinstance(eval(item[tag2id["object_data"]])["content"], str):
            try:
                labels.append(eval(item[tag2id["verify_data_map"]])["comment_label"])
                sentences.append(eval(item[tag2id["object_data"]])["content"])
                ids.append(str(item[tag2id["task_id"]]))
            except KeyError:
                lack_labels_id.append(i)

new_labels = []
for l in labels:
    new_labels.append(l.split(","))

data = {"x": sentences, "y": new_labels, "id": ids}
np.save("product/data/pre_data.npy", data, allow_pickle = True)
print("data_length", i)
print("lack_length", len(lack_labels_id))

# print(lack_labels_id)
# [1904, 6892, 13135, 14985, 15115, 27957, 34188, 46410, 49774, 59030, 59437, 61934, 80522, 82850, 
# 86109, 87103, 89203, 92393, 94925, 95985, 97021, 99648, 101100, 101617, 117763, 120784, 126504, 130099, 
# 147269, 147889, 151179, 157325, 167116, 170321, 172036, 178123, 182160, 189150, 190204, 202034, 209490, 
# 221582, 225069, 236417, 239059, 260480, 265687, 270605, 271547, 281225, 282623, 287597, 289344, 298132, 
# 311737, 312450, 326153, 334296, 335752, 339407, 341214, 353215, 359386, 365363, 366086, 369600, 387703, 
# 387734, 394045, 399360, 419256, 419775, 423746, 424859, 429828, 434882, 438357, 460304, 466570, 467127, 
# 477316, 478420, 482197, 482872, 485438, 486163, 486540, 487780, 491004, 491355, 496438, 499751, 506854, 
# 510798, 512820, 513778, 515037, 520356, 521067, 534062, 540799, 544523, 557636, 559802, 567259, 583502, 
# 585488, 586943, 595433]


