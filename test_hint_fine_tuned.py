import torch
from torch.utils.data import DataLoader
from src.vocab import Vocab
from src.dataset import TokenizerDataset
from hint_fine_tuning import CustomBERTModel
import argparse

def test_model(opt):
    print(f"Loading Vocab {opt.vocab_path}")
    vocab = Vocab(opt.vocab_path)
    vocab.load_vocab()

    print(f"Vocab Size: {len(vocab.vocab)}")

    test_dataset = TokenizerDataset(opt.test_dataset, opt.test_label, vocab, seq_len=50)  # Using sequence length 50
    print(f"Creating Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # Load the entire fine-tuned model (including both architecture and weights)
    print(f"Loading Model from {opt.finetuned_bert_checkpoint}")
    model = torch.load(opt.finetuned_bert_checkpoint, map_location="cpu")

    print(f"Number of Labels: {opt.num_labels}")

    model.eval()
    for batch_idx, data in enumerate(test_data_loader):
        inputs = data["input"].to("cpu")
        segment_info = data["segment_label"].to("cpu")

        with torch.no_grad():
            logits = model(inputs, segment_info)

        print(f"Batch {batch_idx} logits: {logits}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--test_dataset", type=str, default="/home/jupyter/bert/dataset/hint_based/ratio_proportion_change_3/er/er_test_dataset.csv", help="test set for evaluating fine-tuned model")
    parser.add_argument("-tlabel", "--test_label", type=str, default="/home/jupyter/bert/dataset/hint_based/ratio_proportion_change_3/er/test_infos_only.csv", help="label set for evaluating fine-tuned model")
    parser.add_argument("-c", "--finetuned_bert_checkpoint", type=str, default="/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/output/hint_classification/fine_tuned_model_2.pth", help="checkpoint of the saved fine-tuned BERT model") 
    parser.add_argument("-v", "--vocab_path", type=str, default="/home/jupyter/bert/ratio_proportion_change3_1920/_Aug23/pretraining/vocab.txt", help="built vocab model path")
    parser.add_argument("-num_labels", type=int, default=2, help="Number of labels")
    
    opt = parser.parse_args()
    test_model(opt)
