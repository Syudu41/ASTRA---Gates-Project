import argparse

from torch.utils.data import DataLoader
import torch

from bert import BERT
from pretrainer import BERTTrainer, BERTFineTuneTrainer
from dataset import PretrainerDataset, TokenizerDataset
from vocab import Vocab

import time


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('-workspace_name', type=str, default=None)
    parser.add_argument("-p", "--pretrain_dataset", type=str, default="pretraining/pretrain.txt", help="pretraining dataset for bert")
    parser.add_argument("-pv", "--pretrain_val_dataset", type=str, default="pretraining/test.txt", help="pretraining validation dataset for bert")
# default="finetuning/test.txt",
    parser.add_argument("-f", "--train_dataset", type=str, default="finetuning/test_in.txt", help="fine tune train dataset for progress classifier")
    parser.add_argument("-t", "--test_dataset", type=str, default="finetuning/train_in.txt", help="test set for evaluate fine tune train set")
    parser.add_argument("-flabel", "--train_label", type=str, default="finetuning/test_in_label.txt", help="fine tune train dataset for progress classifier")
    parser.add_argument("-tlabel", "--test_label", type=str, default="finetuning/train_in_label.txt", help="test set for evaluate fine tune train set")
    ##### change Checkpoint
    parser.add_argument("-c", "--pretrained_bert_checkpoint", type=str, default="output_feb09/bert_trained.model.ep40", help="checkpoint of saved pretrained bert model") # output_1: output_1/bert_trained.model.ep3
    parser.add_argument("-v", "--vocab_path", type=str, default="pretraining/vocab.txt", help="built vocab model path with bert-vocab")

    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=100, help="maximum sequence length")

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=301, help="number of epochs")
    # Use 50 for pretrain, and 10 for fine tune
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

    # Later run with cuda
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout of network")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    # These two need to be changed for fine tuning
    # parser.add_argument("--pretrain", type=bool, default=True, help="pretraining: true, or false")
    # parser.add_argument("-o", "--output_path", type=str, default="output/bert_trained.seq_encoder.model", help="ex)output/bert.model")
    # parser.add_argument("--same_student_prediction", type=bool, default=False, help="predict sequences by same student: true, or false")
    
    #clear;python3 src/main.py --output_path output/masked/bert_trained.model
    #clear;python3 src/main.py --output_path output/masked_prediction/bert_trained.model --same_student_prediction True

    parser.add_argument("--pretrain", type=bool, default=False, help="pretraining: true, or false")
    parser.add_argument("-o", "--output_path", type=str, default="output/bert_fine_tuned.FS.model", help="ex)output/bert.model")
    # python3 src/main.py
    
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
    
    if args.pretrain:
        
        print("Pre-training......")
        print("Loading Pretraining Dataset", args.pretrain_dataset)
        print(f"Workspace: {args.workspace_name}")
        pretrain_dataset = PretrainerDataset(args.pretrain_dataset, vocab_obj, seq_len=args.seq_len, select_next_seq=args.same_student_prediction)
        
        print("Loading Pretraining validation Dataset", args.pretrain_val_dataset)
        pretrain_valid_dataset = PretrainerDataset(args.pretrain_val_dataset, vocab_obj, seq_len=args.seq_len, select_next_seq=args.same_student_prediction) \
            if args.pretrain_val_dataset is not None else None

        print("Creating Dataloader")
        pretrain_data_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        pretrain_val_data_loader = DataLoader(pretrain_valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)\
            if pretrain_valid_dataset is not None else None
        
        print("Building BERT model")
        # a = 5/0
        # hidden = pretrain_dataset.seq_len if pretrain_dataset.seq_len > args.hidden else args.hidden
        # print("hidden: ", hidden)
        bert = BERT(len(vocab_obj.vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)

        print(f"Creating BERT Trainer .... masking: True, prediction: {args.same_student_prediction}")
        trainer = BERTTrainer(bert, len(vocab_obj.vocab), train_dataloader=pretrain_data_loader, test_dataloader=pretrain_val_data_loader, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, same_student_prediction = args.same_student_prediction, workspace_name = args.workspace_name)

        print("Training Start")
        start_time = time.time()
        for epoch in range(args.epochs):
            trainer.train(epoch)

            if pretrain_val_data_loader is not None:
                trainer.test(epoch)
            
            if epoch > 19 and trainer.save_model: #  or epoch%10 == 0
                trainer.save(epoch, args.output_path)
        end_time = time.time()
        print("Time Taken to pretrain dataset = ", end_time - start_time)
    else:
        print("Fine Tuning......")
        print("Loading Train Dataset", args.train_dataset)            
        train_dataset = TokenizerDataset(args.train_dataset, args.train_label, vocab_obj, seq_len=args.seq_len, train=True)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = TokenizerDataset(args.test_dataset, args.test_label, vocab_obj, seq_len=args.seq_len, train=False) \
            if args.test_dataset is not None else None

        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
            if test_dataset is not None else None

        print("Load Pre-trained BERT model")
        # bert = BERT(len(vocab_obj.vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        device = torch.device("cuda:0" if cuda_condition else "cpu")
        bert = torch.load(args.pretrained_bert_checkpoint, map_location=device)
        
        if args.workspace_name == "ratio_proportion_change4":
            num_labels = 7
        elif args.workspace_name == "ratio_proportion_change3":
            num_labels = 7
        elif args.workspace_name == "scale_drawings_3":
            num_labels = 7
        elif args.workspace_name == "sales_tax_discounts_two_rates":
            num_labels = 3
        # num_labels = 1
        print(f"Number of Labels : {num_labels}")
        print("Creating BERT Fine Tune Trainer")
        trainer = BERTFineTuneTrainer(bert, len(vocab_obj.vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, workspace_name = args.workspace_name, num_labels=num_labels)

        print("Training Start....")
        start_time = time.time()
        for epoch in range(args.epochs):
            trainer.train(epoch)
            
            if epoch > 4 and trainer.save_model:
                trainer.save(epoch, args.output_path)
                
            if test_data_loader is not None:
                trainer.test(epoch)
                   
        end_time = time.time()
        print("Time Taken to fine tune dataset = ", end_time - start_time)
 

if __name__ == "__main__":
    train()