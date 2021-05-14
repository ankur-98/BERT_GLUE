from torch import nn
from transformers import BertModel

class CustomBERTModel(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_checkpoint)
        self.linear = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        output = {"loss": None, "logits":linear_output}
        return output 

class BERTClassifierModel(nn.Module):
    def __init__(self, BERT_model, num_labels):
        super(BERTClassifierModel, self).__init__()
        self.bert = BERT_model
        self.linear = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        output = {"loss": None, "logits":linear_output}
        return output 