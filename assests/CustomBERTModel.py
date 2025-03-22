import torch
import torch.nn as nn
from src.bert import BERT

class CustomBERTModel(nn.Module):
    def __init__(self, vocab_size, output_dim, pre_trained_model_path):
        super(CustomBERTModel, self).__init__()
        hidden_size = 768
        self.bert = BERT(vocab_size=vocab_size, hidden=hidden_size, n_layers=4, attn_heads=8, dropout=0.1)

        # Load the pre-trained model's state_dict
        checkpoint = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            self.bert.load_state_dict(checkpoint)
        else:
            raise TypeError(f"Expected state_dict, got {type(checkpoint)} instead.")

        # Fully connected layer with input size 768 (matching BERT hidden size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, sequence, segment_info):
        sequence = sequence.to(next(self.parameters()).device)
        segment_info = segment_info.to(sequence.device)

        x = self.bert(sequence, segment_info)
        print(f"BERT output shape: {x.shape}") 

        cls_embeddings = x[:, 0]  # Extract CLS token embeddings
        print(f"CLS Embeddings shape: {cls_embeddings.shape}") 

        logits = self.fc(cls_embeddings)  # Pass tensor of size (batch_size, 768) to the fully connected layer
        
        return logits
