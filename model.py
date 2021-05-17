import torch
from util import get_metrics, compute_metrics
from transformers import BertModel


class BERTClassifierModel(torch.nn.Module):
    def __init__(self, BERT_model, num_labels, task=None):
        super(BERTClassifierModel, self).__init__()
        self.task = task
        self.bert = BERT_model
        self.linear = torch.nn.Linear(768, num_labels)
        if task == "stsb":
            self.loss = torch.nn.MSELoss()
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.metric, self.metric_1 = get_metrics(task)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        if self.task == "stsb":
            linear_output = torch.clip((self.sigmoid(linear_output) * 5.5), min=0.0, max=5.0)
        output = {"hidden_layer": bert_output, "logits":linear_output}
        return output

    def compute_metrics(self, predictions, references):
        metric, metric_1 = None, None
        if self.metric is not None: 
            metric = compute_metrics(predictions=predictions, references=references, metric=self.metric)
        if self.metric_1 is not None: 
            metric_1 = compute_metrics(predictions=predictions, references=references, metric=self.metric_1)
        return metric, metric_1

class CustomBERTModel(BERTClassifierModel):
    def __init__(self, model_checkpoint, num_labels, task=None):
        BERT_model = BertModel.from_pretrained(model_checkpoint)
        super(CustomBERTModel, self).__init__(BERT_model, num_labels, task)