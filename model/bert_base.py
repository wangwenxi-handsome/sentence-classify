import torch.nn as nn
from transformers import BertModel, BertConfig

class SentenceMutilabel(nn.Module):
    def __init__(
        self, 
        model_name = "bert-base-chinese",
        label_num = 14,
        dropout_rate = 0.1,
    ):
        super().__init__()
        self.label_num = label_num
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, label_num)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels = None):
        # bert returns a special class, the second item is pool(sentence) output.
        output = self.bert(input_ids, token_type_ids, attention_mask)[1]
        output = self.dropout(output)
        output = self.classifier(output)
        loss = None
        if labels is not None:
            loss = self.loss_func(output, labels).mean()
        return output, loss