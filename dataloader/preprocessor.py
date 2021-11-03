import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from utils.torch_related import MyDataSet, dict_to_list_by_max_len
        

class SentencePre:
    def __init__(
        self,
        model_name = "bert-base-chinese",
        dataloader_name = ["train", "dev", "test"],
        split_rate = [0.1, 0.1],
        max_length = 64,
    ):
        # token relate
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case = True)
        print(self.tokenizer)
        self.tag2id = dict(zip(self.tag, range(len(self.tag))))
        self.id2tag = self.tag

        # dataloader related
        self.dataloader_name = dataloader_name
        self.dataloader_name2id = dict(zip(dataloader_name, range(len(self.dataloader_name))))
        self.split_rate = split_rate

        # init max length
        if max_length is None or isinstance(max_length, (int, float)):
            self.max_length = [max_length] * len(self.dataloader_name)
        else:
            # 保证训练集，验证集和测试集都有最大长度参数
            assert len(max_length) == len(dataloader_name)
            self.max_length = max_length

    def init_data(self, data_path):
        if data_path[-4: ] == ".pth":
            self.data = torch.load(data_path)
        else:
            # data_path must be npy with format of 
            # {"x": [sentence1, sentence2, ...], "y": [[normal, unknow], [normal]...]}
            raw_data = np.load(data_path, allow_pickle = True).tolist()
            if "y" in raw_data:
                raw_data["y"] = self._convert_raw_data_y_to_mutilabels(raw_data["y"])

            # split data
            data_list = self._split_data(raw_data)

            # 将分好的数据对应到dataloader_name上 
            assert len(data_list) == len(self.dataloader_name)
            data_list = {self.dataloader_name[i]: data_list[i] for i in range(len(data_list))}
            # to tensor
            data_tensor = {}
            for i in data_list:
                data_tensor[i] = self._convert_data_with_tensor_format(
                    data_list[i], 
                    max_length = self.max_length[self.dataloader_name2id[i]],
                )
            self.data = {"list": data_list, "tensor": data_tensor}
            # save, 取第一个文件的文件名作为名字，但后缀名为.pth
            torch.save(self.data, os.path.join(os.path.dirname(data_path), "data.pth"))

    def _convert_raw_data_y_to_mutilabels(self, data_y):
        new_data_y = []
        for d in data_y:
            tmp_label = [0] * len(self.tag)
            for i in d:
                tmp_label[self.tag2id[i]] = 1
            new_data_y.append(tmp_label)
        return new_data_y

    def _split_data(self, data):
        """        
        in: {"x": , "y": , "id": }
        out: [{"x": , "y": , "id": }, ...]
        """
        if len(self.split_rate) == 0:
            splited_data = [data]
        else:
            # first split
            raw_rate = 1
            data_keys = list(data.keys())
            data1 = {}
            data2 = {}
            split_data1 = (train_test_split(
                *(data[i] for i in data),
                test_size = self.split_rate[0], 
                random_state = 42,
            ))
            for i in range(len(split_data1)):
                if i % 2 == 0:
                    data1[data_keys[int(i / 2)]] = split_data1[i]
                else:
                    data2[data_keys[int(i / 2)]] = split_data1[i]
            raw_rate -= self.split_rate[0]

            if len(self.split_rate) == 1:
                splited_data = [data1, data2]
            elif len(self.split_rate) == 2:
                # second split
                data3 = {}
                split_data2 = (train_test_split(
                    *(data1[i] for i in data1),
                    test_size = self.split_rate[1] / raw_rate, 
                    random_state = 42,
                ))
                for i in range(len(split_data2)):
                    if i % 2 == 0:
                        data1[data_keys[int(i / 2)]] = split_data2[i]
                    else:
                        data3[data_keys[int(i / 2)]] = split_data2[i]
                splited_data = [data1, data2, data3]
            else:
                raise ValueError(f"len(split_rate) must <= 2")
        return splited_data

    def _convert_data_with_tensor_format(
        self, 
        data,  
        max_length = None, 
        padding = "max_length", 
        truncation = True,
        return_tensors = "pt",
    ):
        """get data with tensor format
        in: {"x": , "y": , "id": }
        out: {"input_ids":, "token_type_ids":, "attention_mask":, "offset_mapping":, "labels":, "length":}
        """
        data_x = [[i] for i in data["x"]]
        data_x = self.tokenizer(
            data_x,
            is_split_into_words = True, 
            return_offsets_mapping= True,
            padding = padding,
            truncation = truncation,
            max_length = max_length,
            return_tensors = return_tensors,
        )
        data_len = [len(i[i == 1]) for i in data_x["attention_mask"]]
        new_data = {
            "input_ids": data_x["input_ids"],
            "token_type_ids": data_x["token_type_ids"],
            "attention_mask": data_x["attention_mask"],
            "offset_mapping": data_x["offset_mapping"],
            "length": torch.tensor(data_len, dtype=torch.long),
        }
        # if there is y
        if "y" in data:
            new_data["labels"] = torch.tensor(data["y"], dtype=torch.float)
        return new_data
    
    @property
    def tag(self):
        return [
            "normal", "unknow", "politics", "violent", "illegal", "abuse", "national", 
            "poorguidance", "advertise", "religious", "gongxuliangsu", "porn", "cult", 
            "minors",
        ]

    def get_dataloader(
        self, 
        batch_size, 
        num_workers = 0, 
        collate_fn = dict_to_list_by_max_len,
    ):
        dataloader = {}
        sampler = None
        for i in self.data["tensor"]:
            dataset = MyDataSet(**self.data["tensor"][i])
            if i == "train":
                shuffle = True
                if dist.is_initialized():
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                shuffle = False

            dataloader[i] = DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle = shuffle, 
                num_workers = num_workers,
                collate_fn = collate_fn,
                sampler = sampler,
            )
        return dataloader

    def get_raw_data_x(self, name):
        return self.data["list"][name]["x"]

    def get_raw_data_y(self, name):
        return self.data["list"][name]["y"]

    def get_raw_data_id(self, name):
        return self.data["list"][name]["id"]

    def get_tokenize_length(self, name):
        return self.data["tensor"][name]["length"]

    def decode(self, model_output, threshold = 0):
        # process outputs
        outputs = model_output.numpy().tolist()
        new_outputs = []
        for i in outputs:
            new_outputs.extend(i)
        label_output = []
        for i in new_outputs:
            tmp = []
            for j in i:
                if j >= threshold:
                    tmp.append(1)
                else:
                    tmp.append(0)
            label_output.append(tmp)
        return label_output