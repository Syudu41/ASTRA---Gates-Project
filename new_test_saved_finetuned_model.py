import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pickle
print("here1",os.getcwd())
from src.dataset import TokenizerDataset, TokenizerDatasetForCalibration
from src.vocab import Vocab
print("here3",os.getcwd())
from src.bert import BERT
from src.seq_model import BERTSM
from src.classifier_model import BERTForClassification, BERTForClassificationWithFeats
# from src.new_finetuning.optim_schedule import ScheduledOptim
import metrics, recalibration, visualization
from recalibration import ModelWithTemperature
import tqdm
import sys
import time
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
print("here3",os.getcwd())
class BERTFineTuneTrainer:
    
    def __init__(self, bertFinetunedClassifierwithFeats: BERT, #BERTForClassificationWithFeats
                 vocab_size: int, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, workspace_name=None, 
                 num_labels=2, log_folder_path: str = None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        # cuda_condition = torch.cuda.is_available() and with_cuda
        # self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.device = torch.device("cpu") #torch.device("cuda:0" if cuda_condition else "cpu")
        # print(cuda_condition, " Device used = ", self.device)
        print(" Device used = ", self.device)
        
        # available_gpus = list(range(torch.cuda.device_count()))

        # This BERT model will be saved every epoch
        self.model = bertFinetunedClassifierwithFeats.to("cpu")
        print(self.model.parameters())
        for param in self.model.parameters():
            param.requires_grad = False
        # Initialize the BERT Language Model, with BERT model
        # self.model = BERTForClassification(self.bert, vocab_size, num_labels).to(self.device)
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 8).to(self.device)
        # self.model = bertFinetunedClassifierwithFeats
        # print(self.model.bert.parameters())
        # for param in self.model.bert.parameters():
        #     param.requires_grad = False
        # BERTForClassificationWithFeats(self.bert, num_labels, 18).to(self.device)
        
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 1).to(self.device)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=available_gpus)

        # Setting the train, validation and test data loader
        # self.train_data = train_dataloader
        # self.val_data = val_dataloader
        self.test_data = test_dataloader
    
        # self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) #, eps=1e-9
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.model.bert.hidden, n_warmup_steps=warmup_steps)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
        self.criterion = nn.CrossEntropyLoss()

        # if num_labels == 1:
        #     self.criterion = nn.MSELoss()
        # elif num_labels == 2:
        #     self.criterion = nn.BCEWithLogitsLoss()
        #     # self.criterion = nn.CrossEntropyLoss()
        # elif num_labels > 2:
            # self.criterion = nn.CrossEntropyLoss()
            # self.criterion = nn.BCEWithLogitsLoss()
        
        
        self.log_freq = log_freq
        self.log_folder_path = log_folder_path
        # self.workspace_name = workspace_name
        # self.finetune_task = finetune_task
        # self.save_model = False
        # self.avg_loss = 10000
        self.start_time = time.time()
        # self.probability_list = []
        for fi in ['test']: #'val', 
            f = open(self.log_folder_path+f"/log_{fi}_finetuned.txt", 'w')
            f.close()
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    # def train(self, epoch):
    #     self.iteration(epoch, self.train_data)

    # def val(self, epoch):
    #     self.iteration(epoch, self.val_data, phase="val")
        
    def test(self, epoch):
        # if epoch == 0:
        #     self.avg_loss = 10000
        self.iteration(epoch, self.test_data, phase="test")

    def iteration(self, epoch, data_loader, phase="train"):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (phase, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        plabels = []
        tlabels = []
        probabs = []
        positive_class_probs=[]
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()
        # self.probability_list = []
        
        with open(self.log_folder_path+f"/log_{phase}_finetuned.txt", 'a') as f:
            sys.stdout = f
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}
                if phase == "train":
                    logits = self.model.forward(data["input"], data["segment_label"], data["feat"])
                else:
                    with torch.no_grad():
                        logits = self.model.forward(data["input"].cpu(), data["segment_label"].cpu(), data["feat"].cpu())

                logits = logits.cpu()
                loss = self.criterion(logits, data["label"])
                # if torch.cuda.device_count() > 1:
                #     loss = loss.mean()

                # 3. backward and optimization only in train
                # if phase == "train":
                #     self.optim_schedule.zero_grad()
                #     loss.backward()
                #     self.optim_schedule.step_and_update_lr()

                # prediction accuracy
                probs = nn.Softmax(dim=-1)(logits) # Probabilities
                probabs.extend(probs.detach().cpu().numpy().tolist())
                predicted_labels = torch.argmax(probs, dim=-1) #correct
                # self.probability_list.append(probs)
                # true_labels = torch.argmax(data["label"], dim=-1)
                plabels.extend(predicted_labels.cpu().numpy())
                tlabels.extend(data['label'].cpu().numpy())
                positive_class_probs = [prob[1] for prob in probabs]
                # Compare predicted labels to true labels and calculate accuracy
                correct = (data['label'] == predicted_labels).sum().item()
                
                avg_loss += loss.item()
                total_correct += correct
                # total_element += true_labels.nelement()
                total_element += data["label"].nelement()
                # print(">>>>>>>>>>>>>>", predicted_labels, true_labels, correct, total_correct, total_element)
                
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100 if total_element != 0 else 0,
                    "loss": loss.item()
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
            
            precisions = precision_score(tlabels, plabels, average="weighted", zero_division=0)
            recalls = recall_score(tlabels, plabels, average="weighted")
            f1_scores = f1_score(tlabels, plabels, average="weighted")
            cmatrix = confusion_matrix(tlabels, plabels)
            end_time = time.time()
            auc_score = roc_auc_score(tlabels, positive_class_probs)
            final_msg = {
                "avg_loss": avg_loss / len(data_iter),
                "total_acc": total_correct * 100.0 / total_element,
                "precisions": precisions,
                "recalls": recalls,
                "f1_scores": f1_scores,
                # "confusion_matrix": f"{cmatrix}",
                # "true_labels": f"{tlabels}",
                # "predicted_labels": f"{plabels}",
                "time_taken_from_start": end_time - self.start_time,
                "auc_score":auc_score
            }
            with open("result.txt", 'w') as file:
                for key, value in final_msg.items():
                    file.write(f"{key}: {value}\n")
            print(final_msg)
            # print(type(plabels),type(tlabels),plabels,tlabels) 
            fpr, tpr, thresholds = roc_curve(tlabels, positive_class_probs)
            with open("roc_data.pkl", "wb") as f:
                pickle.dump((fpr, tpr, thresholds), f)
            with open("roc_data2.pkl", "wb") as f:
                pickle.dump((tlabels,positive_class_probs), f)                
            print(final_msg)
            f.close()
            with open(self.log_folder_path+f"/log_{phase}_finetuned_info.txt", 'a') as f1:
                sys.stdout = f1
                final_msg = {
                "epoch": f"EP{epoch}_{phase}",
                "confusion_matrix": f"{cmatrix}",
                "true_labels": f"{tlabels if epoch == 0 else ''}",
                "predicted_labels": f"{plabels}",
                "probabilities": f"{probabs}",
                "time_taken_from_start": end_time - self.start_time
                }
                print(final_msg)
                f1.close()
            sys.stdout = sys.__stdout__
        sys.stdout = sys.__stdout__
        

        
class BERTFineTuneCalibratedTrainer:
    
    def __init__(self, bertFinetunedClassifierwithFeats: BERT, #BERTForClassificationWithFeats
                 vocab_size: int, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, workspace_name=None, 
                 num_labels=2, log_folder_path: str = None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print(cuda_condition, " Device used = ", self.device)
        
        # available_gpus = list(range(torch.cuda.device_count()))

        # This BERT model will be saved every epoch
        self.model = bertFinetunedClassifierwithFeats
        print(self.model.parameters())
        for param in self.model.parameters():
            param.requires_grad = False
        # Initialize the BERT Language Model, with BERT model
        # self.model = BERTForClassification(self.bert, vocab_size, num_labels).to(self.device)
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 8).to(self.device)
        # self.model = bertFinetunedClassifierwithFeats
        # print(self.model.bert.parameters())
        # for param in self.model.bert.parameters():
        #     param.requires_grad = False
        # BERTForClassificationWithFeats(self.bert, num_labels, 18).to(self.device)
        
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 1).to(self.device)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=available_gpus)

        # Setting the train, validation and test data loader
        # self.train_data = train_dataloader
        # self.val_data = val_dataloader
        self.test_data = test_dataloader
    
        # self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) #, eps=1e-9
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.model.bert.hidden, n_warmup_steps=warmup_steps)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
        self.criterion = nn.CrossEntropyLoss()

        # if num_labels == 1:
        #     self.criterion = nn.MSELoss()
        # elif num_labels == 2:
        #     self.criterion = nn.BCEWithLogitsLoss()
        #     # self.criterion = nn.CrossEntropyLoss()
        # elif num_labels > 2:
            # self.criterion = nn.CrossEntropyLoss()
            # self.criterion = nn.BCEWithLogitsLoss()
        
        
        self.log_freq = log_freq
        self.log_folder_path = log_folder_path
        # self.workspace_name = workspace_name
        # self.finetune_task = finetune_task
        # self.save_model = False
        # self.avg_loss = 10000
        self.start_time = time.time()
        # self.probability_list = []
        for fi in ['test']: #'val', 
            f = open(self.log_folder_path+f"/log_{fi}_finetuned.txt", 'w')
            f.close()
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    # def train(self, epoch):
    #     self.iteration(epoch, self.train_data)

    # def val(self, epoch):
    #     self.iteration(epoch, self.val_data, phase="val")
        
    def test(self, epoch):
        # if epoch == 0:
        #     self.avg_loss = 10000
        self.iteration(epoch, self.test_data, phase="test")

    def iteration(self, epoch, data_loader, phase="train"):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (phase, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        plabels = []
        tlabels = []
        probabs = []

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()
        # self.probability_list = []
        
        with open(self.log_folder_path+f"/log_{phase}_finetuned.txt", 'a') as f:
            sys.stdout = f
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # print(data_pair[0])
                data = {key: value.to(self.device) for key, value in data[0].items()}
                # print(f"data : {data}")
                # data = {key: value.to(self.device) for key, value in data.items()}
                
                # if phase == "train":
                #     logits = self.model.forward(data["input"], data["segment_label"], data["feat"])
                # else:
                with torch.no_grad():
                    # logits = self.model.forward(data["input"], data["segment_label"], data["feat"])
                    logits = self.model.forward(data)

                loss = self.criterion(logits, data["label"])
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                # 3. backward and optimization only in train
                # if phase == "train":
                #     self.optim_schedule.zero_grad()
                #     loss.backward()
                #     self.optim_schedule.step_and_update_lr()

                # prediction accuracy
                probs = nn.Softmax(dim=-1)(logits) # Probabilities
                probabs.extend(probs.detach().cpu().numpy().tolist())
                predicted_labels = torch.argmax(probs, dim=-1) #correct
                # self.probability_list.append(probs)
                # true_labels = torch.argmax(data["label"], dim=-1)
                plabels.extend(predicted_labels.cpu().numpy())
                tlabels.extend(data['label'].cpu().numpy())
                positive_class_probs = [prob[1] for prob in probabs]

                # Compare predicted labels to true labels and calculate accuracy
                correct = (data['label'] == predicted_labels).sum().item()
                
                avg_loss += loss.item()
                total_correct += correct
                # total_element += true_labels.nelement()
                total_element += data["label"].nelement()
                # print(">>>>>>>>>>>>>>", predicted_labels, true_labels, correct, total_correct, total_element)
                
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100 if total_element != 0 else 0,
                    "loss": loss.item()
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
            
            precisions = precision_score(tlabels, plabels, average="weighted", zero_division=0)
            recalls = recall_score(tlabels, plabels, average="weighted")
            f1_scores = f1_score(tlabels, plabels, average="weighted")
            cmatrix = confusion_matrix(tlabels, plabels)
            auc_score = roc_auc_score(tlabels, positive_class_probs)
            end_time = time.time()
            final_msg = {
                "this one":"this one",
                "avg_loss": avg_loss / len(data_iter),
                "total_acc": total_correct * 100.0 / total_element,
                "precisions": precisions,
                "recalls": recalls,
                "f1_scores": f1_scores,
                "auc_score":auc_score,
                # "confusion_matrix": f"{cmatrix}",
                # "true_labels": f"{tlabels}",
                # "predicted_labels": f"{plabels}",
                "time_taken_from_start": end_time - self.start_time
            }
            with open("result.txt", 'w') as file:
                for key, value in final_msg.items():
                    file.write(f"{key}: {value}\n")
            with open("plabels.txt","w") as file:
                file.write(plabels)          
            print(final_msg)
            fpr, tpr, thresholds = roc_curve(tlabels, positive_class_probs)
            f.close()
            with open(self.log_folder_path+f"/log_{phase}_finetuned_info.txt", 'a') as f1:
                sys.stdout = f1
                final_msg = {
                
                "confusion_matrix": f"{cmatrix}",
                "true_labels": f"{tlabels if epoch == 0 else ''}",
                "predicted_labels": f"{plabels}",
                "probabilities": f"{probabs}",
                "time_taken_from_start": end_time - self.start_time
                }
                print(final_msg)
                f1.close()
            sys.stdout = sys.__stdout__
        sys.stdout = sys.__stdout__
        

    
def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('-workspace_name', type=str, default=None)
    parser.add_argument('-code', type=str, default=None, help="folder for pretraining outputs and logs")
    parser.add_argument('-finetune_task', type=str, default=None, help="folder inside finetuning")
    parser.add_argument("-attention", type=bool, default=False, help="analyse attention scores")
    parser.add_argument("-diff_test_folder", type=bool, default=False, help="use for different test folder")
    parser.add_argument("-embeddings", type=bool, default=False, help="get and analyse embeddings")
    parser.add_argument('-embeddings_file_name', type=str, default=None, help="file name of embeddings")
    parser.add_argument("-pretrain", type=bool, default=False, help="pretraining: true, or false")
    # parser.add_argument('-opts', nargs='+', type=str, default=None, help='List of optional steps')
    parser.add_argument("-max_mask", type=int, default=0.15, help="% of input tokens selected for masking") 
    # parser.add_argument("-p", "--pretrain_dataset", type=str, default="pretraining/pretrain.txt", help="pretraining dataset for bert")
    # parser.add_argument("-pv", "--pretrain_val_dataset", type=str, default="pretraining/test.txt", help="pretraining validation dataset for bert")
# default="finetuning/test.txt",
    parser.add_argument("-vocab_path", type=str, default="pretraining/vocab.txt", help="built vocab model path with bert-vocab")

    parser.add_argument("-train_dataset_path", type=str, default="train.txt", help="fine tune train dataset for progress classifier")
    parser.add_argument("-val_dataset_path", type=str, default="val.txt", help="test set for evaluate fine tune train set")
    parser.add_argument("-test_dataset_path", type=str, default="test.txt", help="test set for evaluate fine tune train set")
    parser.add_argument("-num_labels", type=int, default=2, help="Number of labels") 
    parser.add_argument("-train_label_path", type=str, default="train_label.txt", help="fine tune train dataset for progress classifier")
    parser.add_argument("-val_label_path", type=str, default="val_label.txt", help="test set for evaluate fine tune train set")
    parser.add_argument("-test_label_path", type=str, default="test_label.txt", help="test set for evaluate fine tune train set")
    ##### change Checkpoint for finetuning
    parser.add_argument("-pretrained_bert_checkpoint", type=str, default=None, help="checkpoint of saved pretrained bert model") 
    parser.add_argument("-finetuned_bert_classifier_checkpoint", type=str, default=None, help="checkpoint of saved finetuned bert model")  #."output_feb09/bert_trained.model.ep40"
    #."output_feb09/bert_trained.model.ep40"
    parser.add_argument('-check_epoch', type=int, default=None)

    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model") #64
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers") #4
    parser.add_argument("-a", "--attn_heads", type=int, default=4, help="number of attention heads") #8
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence length")

    parser.add_argument("-b", "--batch_size", type=int, default=500, help="number of batch_size") #64
    parser.add_argument("-e", "--epochs", type=int, default=1)#1501, help="number of epochs") #501
    # Use 50 for pretrain, and 10 for fine tune
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    # Later run with cuda
    parser.add_argument("--with_cuda", type=bool, default=False, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    # parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    # parser.add_argument("--on_memory", type=bool, default=False, help="Loading on memory: true or false")
    
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout of network")
    parser.add_argument("--lr", type=float, default=1e-05, help="learning rate of adam") #1e-3
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="adam first beta value") #0.999

    parser.add_argument("-o", "--output_path", type=str, default="bert_trained.seq_encoder.model", help="ex)output/bert.model")    
    # parser.add_argument("-o", "--output_path", type=str, default="output/bert_fine_tuned.model", help="ex)output/bert.model")
    
    args = parser.parse_args()
    for k,v in vars(args).items():
        if 'path' in k:
            if v:
                if k == "output_path":
                    if args.code:
                        setattr(args, f"{k}", args.workspace_name+f"/output/{args.code}/"+v)
                    elif args.finetune_task:
                        setattr(args, f"{k}", args.workspace_name+f"/output/{args.finetune_task}/"+v)
                    else:
                        setattr(args, f"{k}", args.workspace_name+"/output/"+v)
                elif k != "vocab_path":
                    if args.pretrain:
                        setattr(args, f"{k}", args.workspace_name+"/pretraining/"+v)
                    else:
                        if args.code:
                            setattr(args, f"{k}", args.workspace_name+f"/{args.code}/"+v)
                        elif args.finetune_task:
                            if args.diff_test_folder and "test" in k:
                                setattr(args, f"{k}", args.workspace_name+f"/finetuning/"+v)
                            else:
                                setattr(args, f"{k}", args.workspace_name+f"/finetuning/{args.finetune_task}/"+v)
                        else:
                            setattr(args, f"{k}", args.workspace_name+"/finetuning/"+v)
                else:
                    setattr(args, f"{k}", args.workspace_name+"/"+v)
                
                print(f"args.{k} : {getattr(args, f'{k}')}")

    print("Loading Vocab", args.vocab_path)
    vocab_obj = Vocab(args.vocab_path)
    vocab_obj.load_vocab()
    print("Vocab Size: ", len(vocab_obj.vocab))
    
    
    print("Testing using finetuned model......")
    print("Loading Test Dataset", args.test_dataset_path)            
    test_dataset = TokenizerDataset(args.test_dataset_path, args.test_label_path, vocab_obj, seq_len=args.seq_len)
    # test_dataset = TokenizerDatasetForCalibration(args.test_dataset_path, args.test_label_path, vocab_obj, seq_len=args.seq_len)

    print("Creating Dataloader...")
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print("Load fine-tuned BERT classifier model with feats")
    # cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cpu") #torch.device("cuda:0" if cuda_condition else "cpu")
    finetunedBERTclassifier = torch.load(args.finetuned_bert_classifier_checkpoint, map_location=device)
    if isinstance(finetunedBERTclassifier, torch.nn.DataParallel):
        finetunedBERTclassifier = finetunedBERTclassifier.module
    
    new_log_folder = f"{args.workspace_name}/logs"
    new_output_folder = f"{args.workspace_name}/output"
    if args.finetune_task: # is sent almost all the time
        new_log_folder = f"{args.workspace_name}/logs/{args.finetune_task}"
        new_output_folder = f"{args.workspace_name}/output/{args.finetune_task}"

    if not os.path.exists(new_log_folder):
        os.makedirs(new_log_folder)
    if not os.path.exists(new_output_folder):
        os.makedirs(new_output_folder)

    print("Creating BERT Fine Tuned Test Trainer")
    trainer = BERTFineTuneTrainer(finetunedBERTclassifier, 
                    len(vocab_obj.vocab), test_dataloader=test_data_loader, 
                  lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, 
                  with_cuda=args.with_cuda, cuda_devices = args.cuda_devices, log_freq=args.log_freq, 
                  workspace_name = args.workspace_name, num_labels=args.num_labels, log_folder_path=new_log_folder)

    # trainer = BERTFineTuneCalibratedTrainer(finetunedBERTclassifier, 
    #                 len(vocab_obj.vocab), test_dataloader=test_data_loader, 
    #               lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, 
    #               with_cuda=args.with_cuda, cuda_devices = args.cuda_devices, log_freq=args.log_freq, 
    #               workspace_name = args.workspace_name, num_labels=args.num_labels, log_folder_path=new_log_folder)
    print("Testing fine-tuned model Start....")
    start_time = time.time()
    repoch = range(args.check_epoch, args.epochs) if args.check_epoch else range(args.epochs)
    counter = 0
    # patience = 10
    for epoch in repoch:
            print(f'Test Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
            trainer.test(epoch)
            # pickle.dump(trainer.probability_list, open(f"{args.workspace_name}/output/aaai/change4_mid_prob_{epoch}.pkl","wb"))
            print(f'Test Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')
    end_time = time.time()
    print("Time Taken to fine-tune model = ", end_time - start_time)
    print(f'Pretraining Ends, Time: {time.strftime("%D %T", time.localtime(end_time))}')
    
    
    
if __name__ == "__main__":
    train()