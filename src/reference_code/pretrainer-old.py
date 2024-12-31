import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import pickle

from ..bert import BERT
from ..seq_model import BERTSM
from ..classifier_model import BERTForClassification
from ..optim_schedule import ScheduledOptim

import tqdm
import sys
import time

import numpy as np
# import visualization

from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os

class ECE(nn.Module):

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        labels = torch.argmax(labels,1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

class BERTTrainer:
    """
    BERTTrainer pretrains BERT model on input sequence of strategies.
    BERTTrainer make the pretrained BERT model with one training method objective.
        1. Masked Strategy Modelling : 3.3.1 Task #1: Masked SM
    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, val_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=5000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, same_student_prediction = False,
                workspace_name=None, code=None):
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

        # This BERT model will be saved every epoch
        self.bert = bert.to(self.device)
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTSM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=available_gpus)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)
    
        self.log_freq = log_freq
        self.same_student_prediction = same_student_prediction
        self.workspace_name = workspace_name
        self.save_model = False
        self.code = code
        self.avg_loss = 10000
        self.start_time = time.time()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def val(self, epoch):
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
        # str_code = "train" if train else "test"
        # code = "masked_prediction" if self.same_student_prediction else "masked"
        
        self.log_file = f"{self.workspace_name}/logs/{self.code}/log_{phase}_pretrained.txt"
        # bert_hidden_representations = []
        if epoch == 0:
            f = open(self.log_file, 'w')
            f.close()
            if phase == "val":
                self.avg_loss = 10000
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (phase, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss_mask = 0.0
        total_correct_mask = 0
        total_element_mask = 0
        
        avg_loss_pred = 0.0
        total_correct_pred = 0
        total_element_pred = 0
        
        avg_loss = 0.0
        
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()
        with open(self.log_file, 'a') as f:
            sys.stdout = f
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}
                # if i == 0:
                #     print(f"data : {data[0]}")
                # 1. forward the next_sentence_prediction and masked_lm model
                # next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
                if self.same_student_prediction:
                    bert_hidden_rep, mask_lm_output, same_student_output = self.model.forward(data["bert_input"], data["segment_label"], self.same_student_prediction)
                else:
                    bert_hidden_rep, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"], self.same_student_prediction)

                # embeddings = [h for h in bert_hidden_rep.cpu().detach().numpy()]
                # bert_hidden_representations.extend(embeddings)


                # 2-2. NLLLoss of predicting masked token word
                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

                # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                if self.same_student_prediction:
                    # 2-1. NLL(negative log likelihood) loss of is_next classification result
                    same_student_loss = self.criterion(same_student_output, data["is_same_student"])
                    loss = same_student_loss + mask_loss
                else:
                    loss = mask_loss

                # 3. backward and optimization only in train
                if phase == "train":
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()


                # print(f"mask_lm_output : {mask_lm_output}")
                # non_zero_mask = (data["bert_label"] != 0).float()
                # print(f"bert_label : {data['bert_label']}")
                non_zero_mask = (data["bert_label"] != 0).float()
                predictions = torch.argmax(mask_lm_output, dim=-1)
                # print(f"predictions : {predictions}")
                predicted_masked = predictions*non_zero_mask
                # print(f"predicted_masked : {predicted_masked}")
                mask_correct = ((data["bert_label"] == predicted_masked)*non_zero_mask).sum().item()
                # print(f"mask_correct : {mask_correct}")
                # print(f"non_zero_mask.sum().item() : {non_zero_mask.sum().item()}")

                avg_loss_mask += loss.item()
                total_correct_mask += mask_correct
                total_element_mask += non_zero_mask.sum().item()
                # total_element_mask += data["bert_label"].sum().item()

                torch.cuda.empty_cache()
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss_mask / (i + 1),
                    "avg_acc_mask": (total_correct_mask / total_element_mask * 100) if total_element_mask != 0 else 0,
                    "loss": loss.item()
                }

                # next sentence prediction accuracy
                if self.same_student_prediction:
                    correct = same_student_output.argmax(dim=-1).eq(data["is_same_student"]).sum().item()
                    avg_loss_pred += loss.item()
                    total_correct_pred += correct
                    total_element_pred += data["is_same_student"].nelement()
                # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                    post_fix["avg_loss"] = avg_loss_pred / (i + 1)
                    post_fix["avg_acc_pred"] = total_correct_pred / total_element_pred * 100
                    post_fix["loss"] = loss.item()

                avg_loss +=loss.item()

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
                # if not train and epoch > 20 :
                #     pickle.dump(mask_lm_output.cpu().detach().numpy(), open(f"logs/mask/mask_out_e{epoch}_{i}.pkl","wb"))
                #     pickle.dump(data["bert_label"].cpu().detach().numpy(), open(f"logs/mask/label_e{epoch}_{i}.pkl","wb"))
            end_time = time.time()
            final_msg = {
                "epoch": f"EP{epoch}_{phase}",
                "avg_loss": avg_loss / len(data_iter),
                "total_masked_acc": total_correct_mask * 100.0 / total_element_mask if total_element_mask != 0 else 0,
                "time_taken_from_start": end_time - self.start_time
            }

            if self.same_student_prediction:
                final_msg["total_prediction_acc"] = total_correct_pred * 100.0 / total_element_pred

            print(final_msg)
            
            f.close()
        sys.stdout = sys.__stdout__
        
        if phase == "val":
            self.save_model = False
            if self.avg_loss > (avg_loss / len(data_iter)):
                self.save_model = True
                self.avg_loss = (avg_loss / len(data_iter))

        # pickle.dump(bert_hidden_representations, open(f"embeddings/{code}/{str_code}_embeddings_{epoch}.pkl","wb"))

        

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
#         if self.code:
#             fpath = file_path.split("/")
#             # output_path = fpath[0]+ "/"+ fpath[1]+f"/{self.code}/" + fpath[2] + ".ep%d" % epoch
#             output_path = "/",join(fpath[0]+ "/"+ fpath[1]+f"/{self.code}/" + fpath[-1] + ".ep%d" % epoch

#         else:
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
                 num_labels=2, finetune_task=""):
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
        print(with_cuda, cuda_condition, " Device used = ", self.device)

        # This BERT model will be saved every epoch
        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = False
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTForClassification(self.bert, vocab_size, num_labels).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
    
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) #, eps=1e-9
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
        
        if num_labels == 1:
            self.criterion = nn.MSELoss()
        elif num_labels == 2:
            self.criterion = nn.BCEWithLogitsLoss()
            # self.criterion = nn.CrossEntropyLoss()
        elif num_labels > 2:
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = nn.BCEWithLogitsLoss()
        
        # self.ece_criterion = ECE().to(self.device)
        
        self.log_freq = log_freq
        self.workspace_name = workspace_name
        self.finetune_task = finetune_task
        self.save_model = False
        self.avg_loss = 10000
        self.start_time = time.time()
        self.probability_list = []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

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
        
        self.log_file = f"{self.workspace_name}/logs/{self.finetune_task}/log_{str_code}_finetuned.txt"

        if epoch == 0:
            f = open(self.log_file, 'w')
            f.close()
            if not train:
                self.avg_loss = 10000
            
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
        
        eval_accurate_nb = 0
        nb_eval_examples = 0
        logits_list = []
        labels_list = []
        
        if train:
            self.model.train()
        else:
            self.model.eval()
        self.probability_list = []
        with open(self.log_file, 'a') as f:
            sys.stdout = f
            
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}
                if train:
                    h_rep, logits = self.model.forward(data["bert_input"], data["segment_label"])
                else:
                    with torch.no_grad():
                        h_rep, logits = self.model.forward(data["bert_input"], data["segment_label"])
                    # print(logits, logits.shape)
                    logits_list.append(logits.cpu())
                    labels_list.append(data["progress_status"].cpu())
                # print(">>>>>>>>>>>>", progress_output)
                # print(f"{epoch}---nelement--- {data['progress_status'].nelement()}")
                # print(data["progress_status"].shape, logits.shape)
                progress_loss = self.criterion(logits, data["progress_status"])
                loss = progress_loss
                
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                # 3. backward and optimization only in train
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()

                # progress prediction accuracy
                # correct = progress_output.argmax(dim=-1).eq(data["progress_status"]).sum().item()
                probs = nn.LogSoftmax(dim=-1)(logits)
                self.probability_list.append(probs)
                predicted_labels = torch.argmax(probs, dim=-1)
                true_labels = torch.argmax(data["progress_status"], dim=-1)
                plabels.extend(predicted_labels.cpu().numpy())
                tlabels.extend(true_labels.cpu().numpy())

                # Compare predicted labels to true labels and calculate accuracy
                correct = (predicted_labels == true_labels).sum().item()
                avg_loss += loss.item()
                total_correct += correct
                # total_element += true_labels.nelement()
                total_element += data["progress_status"].nelement()
                # print(">>>>>>>>>>>>>>", predicted_labels, true_labels, correct, total_correct, total_element)
                
                # if train: 
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }
#                 else:
#                     logits = logits.detach().cpu().numpy()
#                     label_ids = data["progress_status"].to('cpu').numpy()
#                     tmp_eval_nb = accurate_nb(logits, label_ids)

#                     eval_accurate_nb += tmp_eval_nb
#                     nb_eval_examples += label_ids.shape[0]

#                     # total_element += data["progress_status"].nelement()
#                     # avg_loss += loss.item()

#                     post_fix = {
#                         "epoch": epoch,
#                         "iter": i,
#                         "avg_loss": avg_loss / (i + 1),
#                         "avg_acc": tmp_eval_nb / total_element * 100,
#                         "loss": loss.item()
#                     }


                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
            
            # precisions = precision_score(plabels, tlabels, average="weighted")
            # recalls = recall_score(plabels, tlabels, average="weighted")
            f1_scores = f1_score(plabels, tlabels, average="weighted")
            # if train:
            end_time = time.time()
            final_msg = {
                "epoch": f"EP{epoch}_{str_code}",
                "avg_loss": avg_loss / len(data_iter),
                "total_acc": total_correct * 100.0 / total_element,
                # "precisions": precisions,
                # "recalls": recalls,
                "f1_scores": f1_scores,
                "time_taken_from_start": end_time - self.start_time
            }
#             else:
#                 eval_accuracy = eval_accurate_nb/nb_eval_examples

#                 logits_ece = torch.cat(logits_list)
#                 labels_ece = torch.cat(labels_list)
#                 ece = self.ece_criterion(logits_ece, labels_ece).item()
#                 end_time = time.time()
#                 final_msg = {
#                     "epoch": f"EP{epoch}_{str_code}",
#                     "eval_accuracy": eval_accuracy,
#                     "ece": ece,
#                     "avg_loss": avg_loss / len(data_iter),
#                     "precisions": precisions,
#                     "recalls": recalls,
#                     "f1_scores": f1_scores,
#                     "time_taken_from_start": end_time - self.start_time
#                 }
#                 if self.save_model:
#                     conf_hist = visualization.ConfidenceHistogram()
#                     plt_test = conf_hist.plot(np.array(logits_ece), np.array(labels_ece), title= f"Confidence Histogram {epoch}")
#                     plt_test.savefig(f"{self.workspace_name}/plots/confidence_histogram/{self.finetune_task}/conf_histogram_test_{epoch}.png",bbox_inches='tight')
#                     plt_test.close()

#                     rel_diagram = visualization.ReliabilityDiagram()
#                     plt_test_2 = rel_diagram.plot(np.array(logits_ece), np.array(labels_ece),title=f"Reliability Diagram {epoch}")
#                     plt_test_2.savefig(f"{self.workspace_name}/plots/confidence_histogram/{self.finetune_task}/rel_diagram_test_{epoch}.png",bbox_inches='tight')
#                     plt_test_2.close()
            print(final_msg)

            # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=", total_correct * 100.0 / total_element)
            f.close()
        sys.stdout = sys.__stdout__
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




        
                # plt_test.show()
        # print("EP%d_%s, " % (epoch, str_code))

    def save(self, epoch, file_path="output/bert_fine_tuned_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if self.finetune_task:
            fpath = file_path.split("/")
            output_path = fpath[0]+ "/"+ fpath[1]+f"/{self.finetune_task}/" + fpath[2] + ".ep%d" % epoch
        else:
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

        

        
