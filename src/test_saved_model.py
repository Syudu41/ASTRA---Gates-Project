# import torch.nn as nn
# import torch

import argparse
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,roc_curve, auc
import pickle
# from pretrainer import BERTFineTuneTrainer
from dataset import TokenizerDataset
from vocab import Vocab

import tqdm
import numpy as np

import time
from bert import BERT
# from vocab import Vocab

# class BERTForSequenceClassification(nn.Module):
#     """
#     Since its classification,
#     n_labels = 2
#     """

#     def __init__(self, vocab_size, n_labels, layers=None, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
#         super().__init__()
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         print(device)
#         # model_ep0 = torch.load("output_1/bert_trained.model.ep0", map_location=device)
#         self.bert = torch.load("output_1/bert_trained.model.ep0", map_location=device)
#         self.dropout = nn.Dropout(dropout)
#         # add an output layer
#         self.
        
#     def forward(self, x, segment_info):
        

#         return x

def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


class BERTFineTunedTrainer:
    
    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, workspace_name=None, num_labels=2):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        self.device = "cpu"
        self.model = bert
        self.test_data = test_dataloader
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-9)
        
        if num_labels == 1:
            self.criterion = nn.MSELoss()
        elif num_labels == 2:
            self.criterion = nn.CrossEntropyLoss()
        elif num_labels > 2:
            self.criterion = nn.BCEWithLogitsLoss()
                
        self.log_freq = log_freq
        self.workspace_name = workspace_name
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"
        
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        plabels = []
        tlabels = []
        logits_list = []
        labels_list = []
        positive_class_probs = []
        self.model.eval()
        
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            
            with torch.no_grad():
                h_rep, logits = self.model.forward(data["bert_input"], data["segment_label"])
            # print(logits, logits.shape)
            logits_list.append(logits.cpu())
            labels_list.append(data["progress_status"].cpu())
            probs=F.softmax(logits,dim=-1)
            # probs = F.softmaxoftmax(dim=-1)(logits)
            predicted_labels = torch.argmax(probs, dim=-1)
            true_labels = torch.argmax(data["progress_status"], dim=-1)
            positive_class_probs.extend(probs[:, 1])
            plabels.extend(predicted_labels.cpu().numpy())
            tlabels.extend(true_labels.cpu().numpy())

            # print(">>>>>>>>>>>>>>", predicted_labels, true_labels)
            # Compare predicted labels to true labels and calculate accuracy
            correct = (predicted_labels == true_labels).sum().item()
            total_correct += correct
            total_element += data["progress_status"].nelement()

        precisions = precision_score(tlabels, plabels, average="binary")
        recalls = recall_score(tlabels, plabels, average="binary")
        f1_scores = f1_score(tlabels, plabels, average="binary")
        accuracy = total_correct * 100.0 / total_element
        auc_score = roc_auc_score(tlabels, plabels)

        final_msg = {
            "epoch": f"EP{epoch}_{str_code}",
            "accuracy": accuracy,
            "avg_loss": avg_loss / len(data_iter),
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores,
            "auc_score":auc_score
        }
        with open("result.txt", 'w') as file:
            for key, value in final_msg.items():
                file.write(f"{key}: {value}\n")
        print(final_msg)
        fpr, tpr, thresholds = roc_curve(tlabels, plabels)
        with open("roc_data.pkl", "wb") as f:
            pickle.dump((fpr, tpr, thresholds), f)
            # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=", total_correct * 100.0 / total_element)
    

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # is_model = torch.load("ratio_proportion_change4/output/bert_fine_tuned.IS.model.ep40", map_location=device)
#     learned_parameters = model_ep0.state_dict()
    
#     for param_name, param_tensor in learned_parameters.items():
#         print(param_name)
#         print(param_tensor)
    # # print(model_ep0.state_dict())
    # # model_ep0.add_module("out", nn.Linear(10,2))
    # # print(model_ep0)
    # seq_vocab = Vocab("pretraining/vocab_file.txt")
    # seq_vocab.load_vocab()
    # classifier = BERTForSequenceClassification(len(seq_vocab.vocab), 2)
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-workspace_name', type=str, default="ratio_proportion_change3")
    parser.add_argument("-t", "--test_dataset", type=str, default="../train.txt", help="test set for evaluate fine tune train set")
    parser.add_argument("-tlabel", "--test_label", type=str, default="../train_label.txt", help="test set for evaluate fine tune train set")
    ##### change Checkpoint
    parser.add_argument("-c", "--finetuned_bert_checkpoint", type=str, default="ratio_proportion_change3/output/FS/bert_fine_tuned.model.ep32", help="checkpoint of saved pretrained bert model") 
    parser.add_argument("-v", "--vocab_path", type=str, default="pretraining/vocab.txt", help="built vocab model path with bert-vocab")

    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=100, help="maximum sequence length")

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="number of epochs")
    # Use 50 for pretrain, and 10 for fine tune
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

    # Later run with cuda
    parser.add_argument("--with_cuda", type=bool, default=False, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout of network")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    
    args = parser.parse_args()
    for k,v in vars(args).items():
        if ('dataset' in k) or ('path' in k) or ('label' in k):
            if v:
                setattr(args, f"{k}", args.workspace_name+"/"+v)
                print(f"args.{k} : {getattr(args, f'{k}')}")
                
    print("Loading Vocab", args.vocab_path)
    vocab_obj = Vocab(args.vocab_path)
    vocab_obj.load_vocab()
    print("Vocab Size: ", len(vocab_obj.vocab))
    print("Loading Test Dataset", args.test_dataset)
    test_dataset = TokenizerDataset(args.test_dataset, args.test_label, vocab_obj, seq_len=args.seq_len, train=False)
    print("Creating Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    bert = torch.load(args.finetuned_bert_checkpoint, map_location="cpu")
        
    if args.workspace_name == "ratio_proportion_change4":
        num_labels = 7
    elif args.workspace_name == "ratio_proportion_change3":
        num_labels = 7
    elif args.workspace_name == "scale_drawings_3":
        num_labels = 7
    elif args.workspace_name == "sales_tax_discounts_two_rates":
        num_labels = 3
        
    print(f"Number of Labels : {num_labels}")
    print("Creating BERT Fine Tune Trainer")
    trainer = BERTFineTunedTrainer(bert, len(vocab_obj.vocab), train_dataloader=None, test_dataloader=test_data_loader, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, workspace_name = args.workspace_name, num_labels=num_labels)
    
    print("Testing Start....")
    start_time = time.time()
    for epoch in range(args.epochs):
        trainer.test(epoch)

    end_time = time.time()

    print("Time Taken to fine tune dataset = ", end_time - start_time)