import os
import shutil
import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from dataloader.preprocessor import SentencePre
from model.bert_base import SentenceMutilabel
from worker.worker import Worker
from utils.torch_related import setup_seed, get_torch_model
from utils.torch_related import get_linear_schedule_with_warmup
from utils.logger import init_logger


# defalut kwargs
default_config = {
    # path
    "data_cls": SentencePre,   # 模型训练用到的预处理类
    "data_folder_name": "product/data/data.pth",   # 原始数据位置 
    "folder_path": "product/experiments/sentence1/",    # 本次实验log，checkpoint的保存位置
    # model
    "model_name": "bert-base-chinese",  # 用于选择使用哪一款bert模型
    "label_num": 14,    # NER标签的数目
    # train
    "epoch": 3,     # epoch数
    "lr": 3e-05,    # 学习率
    "batch_size_per_gpu": 48,   # 每张显卡上的batch_size
    "save_step_rate": 0.1,  # 每训练多少百分比保存一个checkpoint
    # main
    "if_train": False,
    "if_select": False,
    "if_test": True,
    "if_save_result": True,
}


def train(logger, config, data_gen, train_dataloader, dev_dataloader = None):
    # model
    device, model = get_torch_model(
        SentenceMutilabel, 
        model_config = {"model_name": config["model_name"], "label_num": config["label_num"]},
        load_checkpoint_path = None,
    )

    # opt
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    sum_steps = config["epoch"] * len(train_dataloader)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr = config["lr"], eps = 1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.1 * sum_steps, num_training_steps = sum_steps)

    # worker
    trainer = Worker(
        device = device,
        model = model, 
        folder_path = config["folder_path"],
        epoch = config["epoch"],
        optimizer = optimizer, 
        scheduler = scheduler,
        save_step_rate = config["save_step_rate"],
    )
    trainer.train(train_dataloader, dev_dataloader)
    del trainer


def select(logger, config, data_gen, dataloader, name):
    metrics = []
    
    all_checkpoints = os.listdir(os.path.join(config["folder_path"], "model/"))
    if ".ipynb_checkpoints" in all_checkpoints:
        all_checkpoints.remove(".ipynb_checkpoints")   
    if "best_model.pth" in all_checkpoints:
        all_checkpoints.remove("best_model.pth")  
    all_checkpoints.sort(key = lambda x: int(x.split(".")[0]))
    all_checkpoints = [os.path.join(config["folder_path"], "model/", i) for i in all_checkpoints]
    logger.info("all checkpoints %s", all_checkpoints)
    
    best_checkpoint = None
    best_precision = None
    
    for i, checkpoint in enumerate(all_checkpoints):
        logger.info("select by %s start with %s", name, checkpoint)
        device, model = get_torch_model(
            SentenceMutilabel, 
            model_config = {"model_name": config["model_name"], "label_num": config["label_num"]},
            load_checkpoint_path = checkpoint,
        )

        trainer = Worker(
            device = device,
            model = model, 
        )

        outputs, _ = trainer.rollout(dataloader)
        outputs = data_gen.decode(outputs)
        metric = classification_report(outputs, data_gen.get_raw_data_y(name), output_dict = True)
        metrics.append(metric)
        logger.info("i result %s", metric)
        
        # update best_checkpoint
        if best_checkpoint is None or metric["weighted avg"]["precision"] > best_precision:
            best_checkpoint = checkpoint
            best_precision = metric["weighted avg"]["precision"]
    
    # select best model
    save_best_model = os.path.join(config["folder_path"], "model/best_model.pth")
    shutil.copyfile(best_checkpoint, save_best_model)

    # plot
    acc = []
    recall = []
    f1 = []
    for m in metrics:
        acc.append(m["weighted avg"]["precision"])
        recall.append(m["weighted avg"]["recall"])
        f1.append(m["weighted avg"]["f1-score"])
    
    logger.info("show all metrics(weighted avg)")
    logger.info("acc %s", acc)
    logger.info("recall %s", recall)
    logger.info("f1 %s", f1)
    
    
def test(logger, config, data_gen, dataloader, name, checkpoint):
    device, model = get_torch_model(
        SentenceMutilabel, 
        model_config = {"model_name": config["model_name"], "label_num": config["label_num"]},
        load_checkpoint_path = checkpoint,
    )

    trainer = Worker(
        device = device,
        model = model, 
    )

    outputs, _ = trainer.rollout(dataloader)
    outputs = data_gen.decode(outputs)
    if data_gen.get_raw_data_y(name) is not None:
        metric = classification_report(outputs, data_gen.get_raw_data_y(name), output_dict = True)
        logger.info("test metric is %s", metric)
    
    if config["if_save_result"]:
        data_gen.save_results(
            data_gen.get_raw_data_x(name), 
            outputs, 
            os.path.join(config["folder_path"], "result.txt"), 
            data_gen.get_raw_data_y(name),
        )


def run_sentence_classify(config):
    # the answer to the world
    setup_seed(42)
    if not os.path.exists(config["folder_path"]):
        os.makedirs(config["folder_path"])
    logger = init_logger(log_path = os.path.join(config["folder_path"], "output.log"))
    logger.info("global config %s", config)

    # data
    logger.info("prepare data")
    n_gpus = max(torch.cuda.device_count(), 1)
    data_gen = config["data_cls"](model_name = config["model_name"])
    data_gen.init_data(data_path = config["data_folder_name"])
    dataloader = data_gen.get_dataloader(batch_size = config["batch_size_per_gpu"] * n_gpus)
    logger.info("dataloader down")

    # train
    if config["if_train"]:
        logger.info("train start")
        train(logger, config, data_gen, dataloader["train"])
        logger.info("train end")

    # dev
    if config["if_select"]:
        logger.info("select start")
        select(logger, config, data_gen, dataloader["dev"], "dev")
        logger.info("select end")
    
    # test
    if config["if_test"]:
        logger.info("test start")
        test(logger, config, data_gen, dataloader["test"], "test", os.path.join(config["folder_path"], "model/best_model.pth"))
        logger.info("test end")


if __name__ == "__main__":
    run_sentence_classify(default_config)