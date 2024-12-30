import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from src.dataset import TokenizerDataset
from src.bert import BERT
from src.pretrainer import BERTFineTuneTrainer1
from src.vocab import Vocab
import pandas as pd

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

    input_ids = torch.cat([t['input_ids'].unsqueeze(0) for t in tokenized_sequences], dim=0)
    segment_labels = torch.cat([t['segment_label'].unsqueeze(0) for t in tokenized_sequences], dim=0)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Segment labels shape: {segment_labels.shape}")

    return input_ids, segment_labels

def custom_collate_fn(batch):
    inputs = [item['input_ids'].unsqueeze(0) for item in batch]
    labels = [item['label'].unsqueeze(0) for item in batch]
    segment_labels = [item['segment_label'].unsqueeze(0) for item in batch]

    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)
    segment_labels = torch.cat(segment_labels, dim=0)

    return {
        'input': inputs,
        'label': labels,
        'segment_label': segment_labels
    }

def main(opt):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    vocab = Vocab(opt.vocab_file)
    vocab.load_vocab()

    # Preprocess data and labels
    input_ids, segment_labels = preprocess_data(opt.data_path, vocab, max_length=50)  # Using sequence length 50
    labels = preprocess_labels(opt.dataset)

    if input_ids is None or segment_labels is None or labels is None:
        print("Error in preprocessing data. Exiting.")
        return

    # Create TensorDataset and split into train and validation sets
    dataset = TensorDataset(input_ids, segment_labels, labels)
    val_size = len(dataset) - int(0.8 * len(dataset))
    val_dataset, train_dataset = random_split(dataset, [val_size, len(dataset) - val_size])

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize custom BERT model and move it to the device
    custom_model = CustomBERTModel(
        vocab_size=len(vocab.vocab),
        output_dim=2,
        pre_trained_model_path=opt.pre_trained_model_path
    ).to(device)

    # Initialize the fine-tuning trainer
    trainer = BERTFineTuneTrainer1(
        bert=custom_model,
        vocab_size=len(vocab.vocab),
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        lr=1e-5,  # Using learning rate 10^-5 as specified
        num_labels=2,
        with_cuda=torch.cuda.is_available(),
        log_freq=10,
        workspace_name=opt.output_dir,
        log_folder_path=opt.log_folder_path
    )

    # Train the model
    trainer.train(epoch=20)

    # Save the model
    os.makedirs(opt.output_dir, exist_ok=True)
    output_model_file = os.path.join(opt.output_dir, 'fine_tuned_model_3.pth')
    torch.save(custom_model, output_model_file)
    print(f'Model saved to {output_model_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune BERT model.')
    parser.add_argument('--dataset', type=str, default='/home/jupyter/bert/dataset/hint_based/ratio_proportion_change_3/er/er_train.csv', help='Path to the dataset file.')
    parser.add_argument('--data_path', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/gt/er.txt', help='Path to the input sequence file.')
    parser.add_argument('--output_dir', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/output/hint_classification', help='Directory to save the fine-tuned model.')
    parser.add_argument('--pre_trained_model_path', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/output/pretrain:1800ms:64hs:4l:8a:50s:64b:1000e:-5lr/bert_trained.seq_encoder.model.ep68', help='Path to the pre-trained BERT model.')
    parser.add_argument('--vocab_file', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/pretraining/vocab.txt', help='Path to the vocabulary file.')
    parser.add_argument('--log_folder_path', type=str, default='/home/jupyter/bert/ratio_proportion_change3_1920/logs/oct', help='Path to the folder for saving logs.')


    opt = parser.parse_args()
    main(opt)
