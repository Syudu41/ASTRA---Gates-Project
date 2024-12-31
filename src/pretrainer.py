import torch
import torch.nn as nn
# from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
# import pickle

from .bert import BERT
from .seq_model import BERTSM
from .classifier_model import BERTForClassification, BERTForClassificationWithFeats
from .optim_schedule import ScheduledOptim

import tqdm
import sys
import time

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os

class BERTTrainer:
    """
    BERTTrainer pretrains BERT model on input sequence of strategies.
    BERTTrainer make the pretrained BERT model with one training method objective.
        1. Masked Strategy Modeling :Masked SM
    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, val_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=5000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, log_folder_path: str = None):
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

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print(cuda_condition, " Device used = ", self.device)
        
        available_gpus = list(range(torch.cuda.device_count()))

        # This BERT model will be saved 
        self.bert = bert.to(self.device)
        # Initialize the BERT Sequence Model, with BERT model
        self.model = BERTSM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=available_gpus)

        # Setting the train, validation and test data loader
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)
    
        self.log_freq = log_freq
        self.log_folder_path = log_folder_path
        # self.workspace_name = workspace_name
        self.save_model = False
        # self.code = code
        self.avg_loss = 10000
        for fi in ['train', 'val', 'test']:
            f = open(self.log_folder_path+f"/log_{fi}_pretrained.txt", 'w')
            f.close()
        self.start_time = time.time()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def val(self, epoch):
        if epoch == 0:
            self.avg_loss = 10000
        self.iteration(epoch, self.val_data, phase="val")
        
    def test(self, epoch):
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
        
        # self.log_file = f"{self.workspace_name}/logs/{self.code}/log_{phase}_pretrained.txt"
        # bert_hidden_representations = [] can be used
        # if epoch == 0:
        #     f = open(self.log_file, 'w')
        #     f.close()
            
        # Progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (phase, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_correct = 0
        total_element = 0
        avg_loss = 0.0
        
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()
        with open(self.log_folder_path+f"/log_{phase}_pretrained.txt", 'a') as f:
            sys.stdout = f
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}
                
                # 1. forward masked_sm model
                # mask_sm_output is log-probabilities output
                mask_sm_output, bert_hidden_rep = self.model.forward(data["bert_input"], data["segment_label"])

                # 2. NLLLoss of predicting masked token word
                loss = self.criterion(mask_sm_output.transpose(1, 2), data["bert_label"])
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                    
                # 3. backward and optimization only in train
                if phase == "train":
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()

                # tokens with highest log-probabilities creates a predicted sequence
                pred_tokens = torch.argmax(mask_sm_output, dim=-1) 
                mask_correct = (data["bert_label"] == pred_tokens) & data["masked_pos"]
                
                total_correct += mask_correct.sum().item()
                total_element += data["masked_pos"].sum().item()
                avg_loss +=loss.item()
                
                torch.cuda.empty_cache()
                
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc_mask": (total_correct / total_element * 100) if total_element != 0 else 0,
                    "loss": loss.item()
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
                    
            end_time = time.time()
            final_msg = {
                "epoch": f"EP{epoch}_{phase}",
                "avg_loss": avg_loss / len(data_iter),
                "total_masked_acc": (total_correct / total_element * 100) if total_element != 0 else 0,
                "time_taken_from_start": end_time - self.start_time
            }
            print(final_msg)
            f.close()
        sys.stdout = sys.__stdout__
        
        if phase == "val":
            self.save_model = False
            if self.avg_loss > (avg_loss / len(data_iter)):
                self.save_model = True
                self.avg_loss = (avg_loss / len(data_iter))

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class BERTFineTuneTrainer:
    
    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, workspace_name=None, 
                 num_labels=2, log_folder_path: str = None):
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

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print(cuda_condition, " Device used = ", self.device)
        
        available_gpus = list(range(torch.cuda.device_count()))

        # This BERT model will be saved every epoch
        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # for name, param in self.bert.named_parameters():
        #     if '.attention.linear_layers.0' in name or \
        #        '.attention.linear_layers.1' in name or \
        #        '.attention.linear_layers.2' in name:
        #     # if 'transformer_blocks.' in name:# or \
        #        # 'transformer_blocks.3.' in name:
        #     # if '2.attention.linear_layers.' in name or \
        #        # '3.attention.linear_layers.' in name:
        #         param.requires_grad = True
        # Initialize the BERT Language Model, with BERT model
        # self.model = BERTForClassification(self.bert, vocab_size, num_labels).to(self.device)
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 8).to(self.device)
        self.model = BERTForClassificationWithFeats(self.bert, num_labels, 17).to(self.device)
        
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 1).to(self.device)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=available_gpus)

        # Setting the train, validation and test data loader
        self.train_data = train_dataloader
        # self.val_data = val_dataloader
        self.test_data = test_dataloader
    
        # self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) #, eps=1e-9
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
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
        self.save_model = False
        self.avg_loss = 10000
        self.start_time = time.time()
        # self.probability_list = []
        for fi in ['train', 'test']: #'val', 
            f = open(self.log_folder_path+f"/log_{fi}_finetuned.txt", 'w')
            f.close()
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    # def val(self, epoch):
    #     self.iteration(epoch, self.val_data, phase="val")
        
    def test(self, epoch):
        if epoch == 0:
            self.avg_loss = 10000
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
                data = {key: value.to(self.device) for key, value in data.items()}
                if phase == "train":
                    logits = self.model.forward(data["input"], data["segment_label"], data["feat"])
                else:
                    with torch.no_grad():
                        logits = self.model.forward(data["input"], data["segment_label"], data["feat"])

                loss = self.criterion(logits, data["label"])
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                # 3. backward and optimization only in train
                if phase == "train":
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()

                # prediction accuracy
                probs = nn.Softmax(dim=-1)(logits) # Probabilities
                probabs.extend(probs.detach().cpu().numpy().tolist())
                predicted_labels = torch.argmax(probs, dim=-1) #correct
                # self.probability_list.append(probs)
                # true_labels = torch.argmax(data["label"], dim=-1)
                plabels.extend(predicted_labels.cpu().numpy())
                tlabels.extend(data['label'].cpu().numpy())

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
            final_msg = {
                "epoch": f"EP{epoch}_{phase}",
                "avg_loss": avg_loss / len(data_iter),
                "total_acc": total_correct * 100.0 / total_element,
                "precisions": precisions,
                "recalls": recalls,
                "f1_scores": f1_scores,
                # "confusion_matrix": f"{cmatrix}",
                # "true_labels": f"{tlabels}",
                # "predicted_labels": f"{plabels}",
                "time_taken_from_start": end_time - self.start_time
            }
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
        
        if phase == "test":
            self.save_model = False
            if self.avg_loss > (avg_loss / len(data_iter)):
                self.save_model = True
                self.avg_loss = (avg_loss / len(data_iter))
            
    def iteration_1(self, epoch_idx, data):
        try:
            data = {key: value.to(self.device) for key, value in data.items()}
            logits = self.model(data['input_ids'], data['segment_label'])
            # Ensure logits is a tensor, not a tuple
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, data['labels'])

            # Backpropagation and optimization
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.log_freq > 0 and epoch_idx % self.log_freq == 0:
                print(f"Epoch {epoch_idx}: Loss = {loss.item()}")

            return loss

        except Exception as e:
            print(f"Error during iteration: {e}")
            raise


    def save(self, epoch, file_path="output/bert_fine_tuned_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path        
    
class BERTFineTuneTrainer1:
    
    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, workspace_name=None, 
                 num_labels=2, log_folder_path: str = None):
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

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print(cuda_condition, " Device used = ", self.device)
        
        available_gpus = list(range(torch.cuda.device_count()))

        # This BERT model will be saved every epoch
        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = False
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTForClassification(self.bert, vocab_size, num_labels).to(self.device)
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 8).to(self.device)
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 8*2).to(self.device)
        
        # self.model = BERTForClassificationWithFeats(self.bert, num_labels, 1).to(self.device)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=available_gpus)

        # Setting the train, validation and test data loader
        self.train_data = train_dataloader
        # self.val_data = val_dataloader
        self.test_data = test_dataloader
    
        # self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) #, eps=1e-9
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
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
        self.save_model = False
        self.avg_loss = 10000
        self.start_time = time.time()
        # self.probability_list = []
        for fi in ['train', 'test']: #'val', 
            f = open(self.log_folder_path+f"/log_{fi}_finetuned.txt", 'w')
            f.close()
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    # def val(self, epoch):
    #     self.iteration(epoch, self.val_data, phase="val")
        
    def test(self, epoch):
        if epoch == 0:
            self.avg_loss = 10000
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
                data = {key: value.to(self.device) for key, value in data.items()}
                if phase == "train":
                    logits = self.model.forward(data["input"], data["segment_label"])#, data["feat"])
                else:
                    with torch.no_grad():
                        logits = self.model.forward(data["input"], data["segment_label"])#, data["feat"])

                loss = self.criterion(logits, data["label"])
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                # 3. backward and optimization only in train
                if phase == "train":
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()

                # prediction accuracy
                probs = nn.Softmax(dim=-1)(logits) # Probabilities
                probabs.extend(probs.detach().cpu().numpy().tolist())
                predicted_labels = torch.argmax(probs, dim=-1) #correct
                # self.probability_list.append(probs)
                # true_labels = torch.argmax(data["label"], dim=-1)
                plabels.extend(predicted_labels.cpu().numpy())
                tlabels.extend(data['label'].cpu().numpy())

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
            final_msg = {
                "epoch": f"EP{epoch}_{phase}",
                "avg_loss": avg_loss / len(data_iter),
                "total_acc": total_correct * 100.0 / total_element,
                "precisions": precisions,
                "recalls": recalls,
                "f1_scores": f1_scores,
                # "confusion_matrix": f"{cmatrix}",
                # "true_labels": f"{tlabels}",
                # "predicted_labels": f"{plabels}",
                "time_taken_from_start": end_time - self.start_time
            }
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
        
        if phase == "test":
            self.save_model = False
            if self.avg_loss > (avg_loss / len(data_iter)):
                self.save_model = True
                self.avg_loss = (avg_loss / len(data_iter))
            
    def iteration_1(self, epoch_idx, data):
        try:
            data = {key: value.to(self.device) for key, value in data.items()}
            logits = self.model(data['input_ids'], data['segment_label'])
            # Ensure logits is a tensor, not a tuple
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, data['labels'])

            # Backpropagation and optimization
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.log_freq > 0 and epoch_idx % self.log_freq == 0:
                print(f"Epoch {epoch_idx}: Loss = {loss.item()}")

            return loss

        except Exception as e:
            print(f"Error during iteration: {e}")
            raise


    def save(self, epoch, file_path="output/bert_fine_tuned_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path        
    

class BERTAttention:
    def __init__(self, bert: BERT, vocab_obj, train_dataloader: DataLoader, workspace_name=None, code=None, finetune_task=None, with_cuda=True):
        
        # available_gpus = list(range(torch.cuda.device_count()))

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print(with_cuda, cuda_condition, " Device used = ", self.device)
        self.bert = bert.to(self.device)
        
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.bert = nn.DataParallel(self.bert, device_ids=available_gpus)
            
        self.train_dataloader = train_dataloader
        self.workspace_name = workspace_name
        self.code = code
        self.finetune_task = finetune_task
        self.vocab_obj = vocab_obj

    def getAttention(self):        
        # self.log_file = f"{self.workspace_name}/logs/{self.code}/log_attention.txt"

                
        labels = ['PercentChange', 'NumeratorQuantity2', 'NumeratorQuantity1', 'DenominatorQuantity1',
                  'OptionalTask_1', 'EquationAnswer', 'NumeratorFactor', 'DenominatorFactor',
                  'OptionalTask_2', 'FirstRow1:1', 'FirstRow1:2', 'FirstRow2:1', 'FirstRow2:2', 'SecondRow',
                  'ThirdRow', 'FinalAnswer','FinalAnswerDirection']
        df_all = pd.DataFrame(0.0, index=labels, columns=labels)
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.train_dataloader),
                              desc="attention",
                              total=len(self.train_dataloader),
                              bar_format="{l_bar}{r_bar}")
        count = 0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            a = self.bert.forward(data["bert_input"], data["segment_label"])
            non_zero = np.sum(data["segment_label"].cpu().detach().numpy())

            # Last Transformer Layer
            last_layer = self.bert.attention_values[-1].transpose(1,0,2,3)
            # print(last_layer.shape)
            head, d_model, s, s = last_layer.shape

            for d in range(d_model):
                seq_labels = self.vocab_obj.to_sentence(data["bert_input"].cpu().detach().numpy().tolist()[d])[1:non_zero-1]
                # df_all = pd.DataFrame(0.0, index=seq_labels, columns=seq_labels)
                indices_to_choose = defaultdict(int)

                for k,s in enumerate(seq_labels):
                    if s in labels:
                        indices_to_choose[s] = k
                indices_chosen = list(indices_to_choose.values())
                selected_seq_labels = [s for l,s in enumerate(seq_labels) if l in indices_chosen]
                # print(len(seq_labels), len(selected_seq_labels))
                for h in range(head):
                    # fig, ax = plt.subplots(figsize=(12, 12)) 
                    # seq_labels = self.vocab_obj.to_sentence(data["bert_input"].cpu().detach().numpy().tolist()[d])#[1:non_zero-1]
                    # seq_labels = self.vocab_obj.to_sentence(data["bert_input"].cpu().detach().numpy().tolist()[d])[1:non_zero-1]
#                     indices_to_choose = defaultdict(int)

#                     for k,s in enumerate(seq_labels):
#                         if s in labels:
#                             indices_to_choose[s] = k
#                     indices_chosen = list(indices_to_choose.values())
#                     selected_seq_labels = [s for l,s in enumerate(seq_labels) if l in indices_chosen]
                    # print(f"Chosen index: {seq_labels, indices_to_choose, indices_chosen, selected_seq_labels}")

                    df_cm = pd.DataFrame(last_layer[h][d][indices_chosen,:][:,indices_chosen], index = selected_seq_labels, columns = selected_seq_labels)
                    df_all = df_all.add(df_cm, fill_value=0)
                    count += 1
                    
                    # df_cm = pd.DataFrame(last_layer[h][d][1:non_zero-1,:][:,1:non_zero-1], index=seq_labels, columns=seq_labels)
                    # df_all = df_all.add(df_cm, fill_value=0)
                
                # df_all = df_all.reindex(index=seq_labels, columns=seq_labels)
                # sns.heatmap(df_all, annot=False)
                # plt.title("Attentions") #Probabilities
                # plt.xlabel("Steps")
                # plt.ylabel("Steps")
                # plt.grid(True)
                # plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=90)
                # plt.savefig(f"{self.workspace_name}/plots/{self.code}/{self.finetune_task}_attention_scores_over_[{h}]_head_n_data[{d}].png", bbox_inches='tight')
                # plt.show()
                # plt.close()



        print(f"Count of total : {count, head * self.train_dataloader.dataset.len}")    
        df_all = df_all.div(count) # head * self.train_dataloader.dataset.len
        df_all = df_all.reindex(index=labels, columns=labels)
        sns.heatmap(df_all, annot=False)
        plt.title("Attentions") #Probabilities
        plt.xlabel("Steps")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=90)
        plt.savefig(f"{self.workspace_name}/plots/{self.code}/{self.finetune_task}_attention_scores.png", bbox_inches='tight')
        plt.show()
        plt.close()
            
        
        
        
