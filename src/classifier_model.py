import torch
import torch.nn as nn

from .bert import BERT


class BERTForClassification(nn.Module):
    """
        Fine-tune Task Classifier Model
    """

    def __init__(self, bert: BERT, vocab_size, n_labels):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size
        :param n_labels: number of labels for the task
        """
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden, n_labels)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.linear(x[:, 0])
    
class BERTForClassificationWithFeats(nn.Module):
    """
        Fine-tune Task Classifier Model 
        BERT embeddings concatenated with features
    """

    def __init__(self, bert: BERT, n_labels, feat_size=9):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size
        :param n_labels: number of labels for the task
        """
        super().__init__()
        self.bert = bert
        # self.linear1 = nn.Linear(self.bert.hidden+feat_size, 128)
        self.linear = nn.Linear(self.bert.hidden+feat_size, n_labels)
        # self.RELU = nn.ReLU()
        # self.linear2 = nn.Linear(128, n_labels)

    def forward(self, x, segment_label, feat):
        x = self.bert(x, segment_label)
        x = torch.cat((x[:, 0], feat), dim=-1)
        # x = self.linear1(x)
        # x = self.RELU(x)
        # return self.linear2(x)
        return self.linear(x)