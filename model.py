import torch
from transformers import BertModel

class CustomBERTModel(torch.nn.Module):
    def __init__(self, model_checkpoint, num_labels, task=None):
        super(CustomBERTModel, self).__init__()
        self.task = task
        if task == "stsb":
            self.loss = torch.nn.MSELoss()
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.bert = BertModel.from_pretrained(model_checkpoint)
        self.linear = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        if self.task == "stsb":
            linear_output = torch.clip((self.sigmoid(linear_output) * 5.5), min=0.0, max=5.0)
        output = {"loss": self.loss, "logits":linear_output}
        return output 

class BERTClassifierModel(torch.nn.Module):
    def __init__(self, BERT_model, num_labels, task=None):
        super(BERTClassifierModel, self).__init__()
        self.task = task
        if task == "stsb":
            self.loss = torch.nn.MSELoss()
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.bert = BERT_model
        self.linear = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        if self.task == "stsb":
            linear_output = torch.clip((self.sigmoid(linear_output) * 5.5), min=0.0, max=5.0)
        output = {"loss": self.loss, "logits":linear_output}
        return output 