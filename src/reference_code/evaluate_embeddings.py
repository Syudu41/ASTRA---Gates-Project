from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy

import pickle
import tqdm

from ..bert import BERT
from ..vocab import Vocab
from ..dataset import TokenizerDataset
import argparse
from itertools import combinations

def generate_subset(s):
    subsets = []
    for r in range(len(s) + 1):
        combinations_result = combinations(s, r)
        if r==1:
            subsets.extend(([item] for sublist in combinations_result for item in sublist))
        else:
            subsets.extend((list(sublist) for sublist in combinations_result))
    subsets_dict = {i:s for i, s in enumerate(subsets)}
    return subsets_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-workspace_name', type=str, default=None)
    parser.add_argument("-seq_len", type=int, default=100, help="maximum sequence length")
    parser.add_argument('-pretrain', type=bool, default=False)
    parser.add_argument('-masked_pred', type=bool, default=False)
    parser.add_argument('-epoch', type=str, default=None)
    # parser.add_argument('-set_label', type=bool, default=False)
    # parser.add_argument('--label_standard', nargs='+', type=str, help='List of optional tasks')

    options = parser.parse_args()
    
    folder_path = options.workspace_name+"/" if options.workspace_name else ""
    
    # if options.set_label:
    #     label_standard = generate_subset({'optional-tasks-1', 'optional-tasks-2'})
    #     pickle.dump(label_standard, open(f"{folder_path}pretraining/pretrain_opt_label.pkl", "wb"))
    # else:
    #     label_standard = pickle.load(open(f"{folder_path}pretraining/pretrain_opt_label.pkl", "rb"))
    # print(f"options.label _standard: {options.label_standard}")
    vocab_path = f"{folder_path}check/pretraining/vocab.txt"
    # vocab_path = f"{folder_path}pretraining/vocab.txt"

    
    print("Loading Vocab", vocab_path)
    vocab_obj = Vocab(vocab_path)
    vocab_obj.load_vocab()
    print("Vocab Size: ", len(vocab_obj.vocab))
    
    # label_standard = list(pickle.load(open(f"dataset/CL4999_1920/{options.workspace_name}/unique_problems_list.pkl", "rb")))
    # label_standard = generate_subset({'optional-tasks-1', 'optional-tasks-2', 'OptionalTask_1', 'OptionalTask_2'})
    # pickle.dump(label_standard, open(f"{folder_path}pretraining/pretrain_opt_label.pkl", "wb"))
    
    if options.masked_pred:
        str_code = "masked_prediction"
        output_name = f"{folder_path}output/bert_trained.seq_model.ep{options.epoch}"
    else:
        str_code = "masked"
        output_name = f"{folder_path}output/bert_trained.seq_encoder.model.ep{options.epoch}"
    
    folder_path = folder_path+"check/"
    # folder_path = folder_path
    if options.pretrain:
        pretrain_file = f"{folder_path}pretraining/pretrain.txt"
        pretrain_label = f"{folder_path}pretraining/pretrain_opt.pkl"
        
        # pretrain_file = f"{folder_path}finetuning/train.txt"
        # pretrain_label = f"{folder_path}finetuning/train_label.txt"
        
        embedding_file_path = f"{folder_path}embeddings/pretrain_embeddings_{str_code}_{options.epoch}.pkl"
        print("Loading Pretrain Dataset ", pretrain_file)
        pretrain_dataset = TokenizerDataset(pretrain_file, pretrain_label, vocab_obj, seq_len=options.seq_len)
        
        print("Creating Dataloader")
        pretrain_data_loader = DataLoader(pretrain_dataset, batch_size=32, num_workers=4)
    else:
        val_file = f"{folder_path}pretraining/test.txt"
        val_label = f"{folder_path}pretraining/test_opt.txt"
            
#         val_file = f"{folder_path}finetuning/test.txt"
#         val_label = f"{folder_path}finetuning/test_label.txt"
        embedding_file_path = f"{folder_path}embeddings/test_embeddings_{str_code}_{options.epoch}.pkl"

        print("Loading Validation Dataset ", val_file)
        val_dataset = TokenizerDataset(val_file, val_label, vocab_obj, seq_len=options.seq_len)

        print("Creating Dataloader")
        val_data_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Load Pre-trained BERT model...")
    print(output_name)
    bert = torch.load(output_name, map_location=device)  
#     learned_parameters = model_ep0.state_dict()
    for param in bert.parameters():
        param.requires_grad = False

    if options.pretrain:
        print("Pretrain-embeddings....")
        data_iter = tqdm.tqdm(enumerate(pretrain_data_loader),
                                  desc="pre-train",
                                  total=len(pretrain_data_loader),
                                  bar_format="{l_bar}{r_bar}")
        pretrain_embeddings = []
        for i, data in data_iter:
            data = {key: value.to(device) for key, value in data.items()}
            hrep = bert(data["bert_input"], data["segment_label"]) 
            # print(hrep[:,0].cpu().detach().numpy())
            embeddings = [h for h in hrep[:,0].cpu().detach().numpy()]
            pretrain_embeddings.extend(embeddings)
        pickle.dump(pretrain_embeddings, open(embedding_file_path,"wb"))
        # pickle.dump(pretrain_embeddings, open("embeddings/finetune_cfa_train_embeddings.pkl","wb"))

    else:
        print("Validation-embeddings....")
        data_iter = tqdm.tqdm(enumerate(val_data_loader),
                                  desc="validation",
                                  total=len(val_data_loader),
                                  bar_format="{l_bar}{r_bar}")
        val_embeddings = []
        for i, data in data_iter:
            data = {key: value.to(device) for key, value in data.items()}
            hrep = bert(data["bert_input"], data["segment_label"]) 
            # print(,hrep[:,0].shape)
            embeddings = [h for h in hrep[:,0].cpu().detach().numpy()]
            val_embeddings.extend(embeddings)
        pickle.dump(val_embeddings, open(embedding_file_path,"wb"))
        # pickle.dump(val_embeddings, open("embeddings/finetune_cfa_test_embeddings.pkl","wb"))

