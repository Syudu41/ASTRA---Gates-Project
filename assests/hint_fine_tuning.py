import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from src.dataset import TokenizerDataset
from src.bert import BERT
from src.pretrainer import BERTFineTuneTrainer1
from src.vocab import Vocab
import pandas as pd


# class CustomBERTModel(nn.Module):
#     def __init__(self, vocab_size, output_dim, pre_trained_model_path):
#         super(CustomBERTModel, self).__init__()
#         hidden_size = 768
#         self.bert = BERT(vocab_size=vocab_size, hidden=hidden_size, n_layers=12, attn_heads=12, dropout=0.1)
#         checkpoint = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))
#         if isinstance(checkpoint, dict):
#             self.bert.load_state_dict(checkpoint)
#         elif isinstance(checkpoint, BERT):
#             self.bert = checkpoint
#         else:
#             raise TypeError(f"Expected state_dict or BERT instance, got {type(checkpoint)} instead.")
#         self.fc = nn.Linear(hidden_size, output_dim)

#     def forward(self, sequence, segment_info):
#         sequence = sequence.to(next(self.parameters()).device)
#         segment_info = segment_info.to(sequence.device)

#         if sequence.size(0) == 0 or sequence.size(1) == 0:
#             raise ValueError("Input sequence tensor has 0 elements. Check data preprocessing.")

#         x = self.bert(sequence, segment_info)
#         print(f"BERT output shape: {x.shape}")

#         if x.size(0) == 0 or x.size(1) == 0:
#             raise ValueError("BERT output tensor has 0 elements. Check input dimensions.")

#         cls_embeddings = x[:, 0]
#         logits = self.fc(cls_embeddings)
#         return logits

# class CustomBERTModel(nn.Module):
#     def __init__(self, vocab_size, output_dim, pre_trained_model_path):
#         super(CustomBERTModel, self).__init__()
#         hidden_size = 764  # Ensure this is 768
#         self.bert = BERT(vocab_size=vocab_size, hidden=hidden_size, n_layers=12, attn_heads=12, dropout=0.1)
        
#         # Load the pre-trained model's state_dict
#         checkpoint = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))
#         if isinstance(checkpoint, dict):
#             self.bert.load_state_dict(checkpoint)
#         else:
#             raise TypeError(f"Expected state_dict, got {type(checkpoint)} instead.")
        
#         # Fully connected layer with input size 768
#         self.fc = nn.Linear(hidden_size, output_dim)

#     def forward(self, sequence, segment_info):
#         sequence = sequence.to(next(self.parameters()).device)
#         segment_info = segment_info.to(sequence.device)

#         x = self.bert(sequence, segment_info)
#         print(f"BERT output shape: {x.shape}")  # Should output (batch_size, seq_len, 768)

#         cls_embeddings = x[:, 0]  # Extract CLS token embeddings
#         print(f"CLS Embeddings shape: {cls_embeddings.shape}")  # Should output (batch_size, 768)

#         logits = self.fc(cls_embeddings)  # Should now pass a tensor of size (batch_size, 768) to `fc`
        
#         return logits


# for test
class CustomBERTModel(nn.Module):
    def __init__(self, vocab_size, output_dim, pre_trained_model_path):
        super(CustomBERTModel, self).__init__()
        self.hidden = 764  # Ensure this is defined correctly
        self.bert = BERT(vocab_size=vocab_size, hidden=self.hidden, n_layers=12, attn_heads=12, dropout=0.1)

        # Load the pre-trained model's state_dict
        checkpoint = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            self.bert.load_state_dict(checkpoint)
        else:
            raise TypeError(f"Expected state_dict, got {type(checkpoint)} instead.")

        self.fc = nn.Linear(self.hidden, output_dim)

    def forward(self, sequence, segment_info):
        x = self.bert(sequence, segment_info)
        cls_embeddings = x[:, 0]  # Extract CLS token embeddings
        logits = self.fc(cls_embeddings)  # Pass to fully connected layer
        return logits

def preprocess_labels(label_csv_path):
    try:
        labels_df = pd.read_csv(label_csv_path)
        labels = labels_df['last_hint_class'].values.astype(int)
        return torch.tensor(labels, dtype=torch.long)
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return None


def preprocess_data(data_path, vocab, max_length=128):
    try:
        with open(data_path, 'r') as f:
            sequences = f.readlines()
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None, None

    if len(sequences) == 0:
        raise ValueError(f"No sequences found in data file {data_path}. Check the file content.")

    tokenized_sequences = []

    for sequence in sequences:
        sequence = sequence.strip()
        if sequence:
            encoded = vocab.to_seq(sequence, seq_len=max_length)
            encoded = encoded[:max_length] + [vocab.vocab.get('[PAD]', 0)] * (max_length - len(encoded))
            segment_label = [0] * max_length

            tokenized_sequences.append({
                'input_ids': torch.tensor(encoded),
                'segment_label': torch.tensor(segment_label)
            })

    if not tokenized_sequences:
        raise ValueError("Tokenization resulted in an empty list. Check the sequences and tokenization logic.")

    tokenized_sequences = [t for t in tokenized_sequences if len(t['input_ids']) == max_length]

    if not tokenized_sequences:
        raise ValueError("All tokenized sequences are of unexpected length. This suggests an issue with the tokenization logic.")

    input_ids = torch.cat([t['input_ids'].unsqueeze(0) for t in tokenized_sequences], dim=0)
    segment_labels = torch.cat([t['segment_label'].unsqueeze(0) for t in tokenized_sequences], dim=0)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Segment labels shape: {segment_labels.shape}")

    return input_ids, segment_labels


def collate_fn(batch):
    inputs = []
    labels = []
    segment_labels = []

    for item in batch:
        if item is None:
            continue

        if isinstance(item, dict):
            inputs.append(item['input_ids'].unsqueeze(0))
            labels.append(item['label'].unsqueeze(0))
            segment_labels.append(item['segment_label'].unsqueeze(0))

    if len(inputs) == 0 or len(segment_labels) == 0:
        print("Empty batch encountered. Returning None to skip this batch.")
        return None

    try:
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)
        segment_labels = torch.cat(segment_labels, dim=0)
    except Exception as e:
        print(f"Error concatenating tensors: {e}")
        return None

    return {
        'input': inputs,
        'label': labels,
        'segment_label': segment_labels
    }

def custom_collate_fn(batch):
    processed_batch = collate_fn(batch)
    
    if processed_batch is None or len(processed_batch['input']) == 0:
        # Return a valid batch with at least one element instead of an empty one
        return {
            'input': torch.zeros((1, 128), dtype=torch.long),
            'label': torch.zeros((1,), dtype=torch.long),
            'segment_label': torch.zeros((1, 128), dtype=torch.long)
        }
    
    return processed_batch


def train_without_progress_status(trainer, epoch, shuffle):
    for epoch_idx in range(epoch):
        print(f"EP_train:{epoch_idx}:")
        for batch in trainer.train_data:
            if batch is None:
                continue

            # Check if batch is a string (indicating an issue)
            if isinstance(batch, str):
                print(f"Error: Received a string instead of a dictionary in batch: {batch}")
                raise ValueError(f"Unexpected string in batch: {batch}")

            # Validate the batch structure before passing to iteration
            if isinstance(batch, dict):
                # Verify that all expected keys are present and that the values are tensors
                if all(key in batch for key in ['input_ids', 'segment_label', 'labels']):
                    if all(isinstance(batch[key], torch.Tensor) for key in batch):
                        try:
                            print(f"Batch Structure: {batch}")  # Debugging batch before iteration
                            trainer.iteration(epoch_idx, batch)
                        except Exception as e:
                            print(f"Error during batch processing: {e}")
                            sys.stdout.flush()
                            raise e  # Propagate the exception for better debugging
                    else:
                        print(f"Error: Expected all values in batch to be tensors, but got: {batch}")
                        raise ValueError("Batch contains non-tensor values.")
                else:
                    print(f"Error: Batch missing expected keys. Batch keys: {batch.keys()}")
                    raise ValueError("Batch does not contain expected keys.")
            else:
                print(f"Error: Expected batch to be a dictionary but got {type(batch)} instead.")
                raise ValueError(f"Invalid batch structure: {batch}")

# def main(opt):
#     # device = torch.device("cpu")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     vocab = Vocab(opt.vocab_file)
#     vocab.load_vocab()

#     input_ids, segment_labels = preprocess_data(opt.data_path, vocab, max_length=128)
#     labels = preprocess_labels(opt.dataset)

#     if input_ids is None or segment_labels is None or labels is None:
#         print("Error in preprocessing data. Exiting.")
#         return

#     dataset = TensorDataset(input_ids, segment_labels, torch.tensor(labels, dtype=torch.long))
#     val_size = len(dataset) - int(0.8 * len(dataset))
#     val_dataset, train_dataset = random_split(dataset, [val_size, len(dataset) - val_size])

#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=32,
#         shuffle=True,
#         collate_fn=custom_collate_fn
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=32,
#         shuffle=False,
#         collate_fn=custom_collate_fn
#     )

#     custom_model = CustomBERTModel(
#         vocab_size=len(vocab.vocab),
#         output_dim=2,
#         pre_trained_model_path=opt.pre_trained_model_path
#     ).to(device)

#     trainer = BERTFineTuneTrainer1(
#         bert=custom_model.bert,
#         vocab_size=len(vocab.vocab),
#         train_dataloader=train_dataloader,
#         test_dataloader=val_dataloader,
#         lr=5e-5,
#         num_labels=2,
#         with_cuda=torch.cuda.is_available(),
#         log_freq=10,
#         workspace_name=opt.output_dir,
#         log_folder_path=opt.log_folder_path 
#     )

#     trainer.train(epoch=20)

#     # os.makedirs(opt.output_dir, exist_ok=True)
#     # output_model_file = os.path.join(opt.output_dir, 'fine_tuned_model.pth')
#     # torch.save(custom_model.state_dict(), output_model_file)
#     # print(f'Model saved to {output_model_file}')
    
#     os.makedirs(opt.output_dir, exist_ok=True)
#     output_model_file = os.path.join(opt.output_dir, 'fine_tuned_model_2.pth')
#     torch.save(custom_model, output_model_file)
#     print(f'Model saved to {output_model_file}')


def main(opt):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(torch.cuda.is_available())  # Should return True if GPU is available
    print(torch.cuda.device_count())  

    # Load vocabulary
    vocab = Vocab(opt.vocab_file)
    vocab.load_vocab()

    # Preprocess data and labels
    input_ids, segment_labels = preprocess_data(opt.data_path, vocab, max_length=128)
    labels = preprocess_labels(opt.dataset)

    if input_ids is None or segment_labels is None or labels is None:
        print("Error in preprocessing data. Exiting.")
        return

    # Transfer tensors to the correct device (GPU/CPU)
    input_ids = input_ids.to(device)
    segment_labels = segment_labels.to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    # Create TensorDataset and split into train and validation sets
    dataset = TensorDataset(input_ids, segment_labels, labels)
    val_size = len(dataset) - int(0.8 * len(dataset))
    val_dataset, train_dataset = random_split(dataset, [val_size, len(dataset) - val_size])

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Initialize custom BERT model and move it to the device
    custom_model = CustomBERTModel(
        vocab_size=len(vocab.vocab),
        output_dim=2,
        pre_trained_model_path=opt.pre_trained_model_path
    ).to(device)

    # Initialize the fine-tuning trainer
    trainer = BERTFineTuneTrainer1(
        bert=custom_model.bert,
        vocab_size=len(vocab.vocab),
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        lr=5e-5,
        num_labels=2,
        with_cuda=torch.cuda.is_available(),
        log_freq=10,
        workspace_name=opt.output_dir,
        log_folder_path=opt.log_folder_path 
    )

    # Train the model
    trainer.train(epoch=20)

    # Save the model to the specified output directory
    # os.makedirs(opt.output_dir, exist_ok=True)
    # output_model_file = os.path.join(opt.output_dir, 'fine_tuned_model_2.pth')
    # torch.save(custom_model.state_dict(), output_model_file)
    # print(f'Model saved to {output_model_file}')
    os.makedirs(opt.output_dir, exist_ok=True)
    output_model_file = os.path.join(opt.output_dir, 'fine_tuned_model_2.pth')
    torch.save(custom_model, output_model_file)
    print(f'Model saved to {output_model_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune BERT model.')
    parser.add_argument('--dataset', type=str, default='/home/jupyter/bert/dataset/hint_based/ratio_proportion_change_3/er/er_train.csv', help='Path to the dataset file.')
    parser.add_argument('--data_path', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/gt/er.txt', help='Path to the input sequence file.')
    parser.add_argument('--output_dir', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/output/hint_classification', help='Directory to save the fine-tuned model.')
    parser.add_argument('--pre_trained_model_path', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/output/pretrain:1800ms:64hs:4l:8a:50s:64b:1000e:-5lr/bert_trained.seq_encoder.model.ep68', help='Path to the pre-trained BERT model.')
    parser.add_argument('--vocab_file', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/pretraining/vocab.txt', help='Path to the vocabulary file.')
    parser.add_argument('--log_folder_path', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/logs/oct_logs', help='Path to the folder for saving logs.')


    opt = parser.parse_args()
    main(opt)