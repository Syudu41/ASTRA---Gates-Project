import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tqdm
import random
from .vocab import Vocab
import pickle
import copy
# from sklearn.preprocessing import OneHotEncoder

class PretrainerDataset(Dataset):
    """
        Class name: PretrainDataset
        
    """
    def __init__(self, dataset_path, vocab, seq_len=30, max_mask=0.15):
        self.dataset_path = dataset_path
        self.vocab = vocab # Vocab object
        
        # Related to input dataset file
        self.lines = []
        self.index_documents = {}

        seq_len_list = []
        with open(self.dataset_path, "r") as reader:
            i = 0
            index = 0
            self.index_documents[i] = []
            for line in tqdm.tqdm(reader.readlines()):
                if line:
                    line = line.strip()
                    if not line:
                        i+=1
                        self.index_documents[i] = []
                    else:
                        self.index_documents[i].append(index)
                        self.lines.append(line.split("\t"))
                        len_line = len(line.split("\t"))
                        seq_len_list.append(len_line)
                        index+=1
            reader.close()
        print("Sequence Stats: len: %s, min: %s, max: %s, average: %s"% (len(seq_len_list),
              min(seq_len_list), max(seq_len_list), sum(seq_len_list)/len(seq_len_list)))
        print("Unique Sequences: ", len({tuple(ll) for ll in self.lines}))
        self.index_documents = {k:v for k,v in self.index_documents.items() if v}
        print(len(self.index_documents))
        self.seq_len = seq_len
        print("Sequence length set at: ", self.seq_len)
        self.max_mask = max_mask
        print("% of input tokens selected for masking : ",self.max_mask)

    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, item):
        token_a = self.lines[item]
        # sa_masked = None
        # sa_masked_label = None
        # token_b = None
        # is_same_student = None
        # sb_masked = None
        # sb_masked_label = None
        
        # if self.select_next_seq:
        #     is_same_student, token_b = self.get_token_b(item)
        #     is_same_student = 1 if is_same_student else 0
        #     token_a1, token_b1 = self.truncate_to_max_seq(token_a, token_b)
        #     sa_masked, sa_masked_label = self.random_mask_seq(token_a1)
        #     sb_masked, sb_masked_label = self.random_mask_seq(token_b1)
        # else:
        token_a = token_a[:self.seq_len-2]
        sa_masked, sa_masked_label, sa_masked_pos = self.random_mask_seq(token_a)

        s1 = ([self.vocab.vocab['[CLS]']] + sa_masked + [self.vocab.vocab['[SEP]']])
        s1_label = ([self.vocab.vocab['[PAD]']] + sa_masked_label + [self.vocab.vocab['[PAD]']])
        segment_label = [1 for _ in range(len(s1))]
        masked_pos = ([0] + sa_masked_pos + [0])
        
        # if self.select_next_seq:
        #     s1 = s1 + sb_masked + [self.vocab.vocab['[SEP]']]
        #     s1_label = s1_label + sb_masked_label + [self.vocab.vocab['[PAD]']]
        #     segment_label = segment_label + [2 for _ in range(len(sb_masked)+1)]
        
        padding = [self.vocab.vocab['[PAD]'] for _ in range(self.seq_len - len(s1))]
        s1.extend(padding)
        s1_label.extend(padding)
        segment_label.extend(padding)
        masked_pos.extend(padding)
 
        output = {'bert_input': s1,
                 'bert_label': s1_label,
                 'segment_label': segment_label,
                 'masked_pos': masked_pos}
        # print(f"tokenA: {token_a}")
        # print(f"output: {output}")
        
        # if self.select_next_seq:
        #     output['is_same_student'] = is_same_student
        
        # print(item, len(s1), len(s1_label), len(segment_label))
        # print(f"{item}.")
        return {key: torch.tensor(value) for key, value in output.items()}
    
    def random_mask_seq(self, tokens):
        """
        Input: original token seq
        Output: masked token seq, output label
        """
        
        masked_pos = []
        output_labels = []
        output_tokens = copy.deepcopy(tokens)
        opt_step = False
        for i, token in enumerate(tokens):
            if token in ['OptionalTask_1', 'EquationAnswer', 'NumeratorFactor', 'DenominatorFactor', 'OptionalTask_2', 'FirstRow1:1', 'FirstRow1:2', 'FirstRow2:1', 'FirstRow2:2', 'SecondRow', 'ThirdRow']:
                opt_step = True
            # if opt_step:
            #     prob = random.random()
            #     if prob < self.max_mask:
            #         output_tokens[i] = random.choice([3,7,8,9,11,12,13,14,15,16,22,23,24,25,26,27,30,31,32])
            #         masked_pos.append(1)
            #     else:
            #         output_tokens[i] = self.vocab.vocab.get(token, self.vocab.vocab['[UNK]'])
            #         masked_pos.append(0)
            #     output_labels.append(self.vocab.vocab.get(token, self.vocab.vocab['[UNK]']))
            #     opt_step = False
            # else:    
            prob = random.random()
            if prob < self.max_mask:
             # chooses 15% of token positions at random
                # prob /= 0.15
                prob = random.random()
                if prob < 0.8: #[MASK] token 80% of the time
                    output_tokens[i] = self.vocab.vocab['[MASK]']
                    masked_pos.append(1)
                elif prob < 0.9: # a random token 10% of the time 
                    # print(".......0.8-0.9......")
                    if opt_step:
                        output_tokens[i] = random.choice([7,8,9,11,12,13,14,15,16,22,23,24,25,26,27,30,31,32])
                        opt_step = False
                    else:
                        output_tokens[i] = random.randint(1, len(self.vocab.vocab)-1)
                    masked_pos.append(1)
                else: # the unchanged i-th token 10% of the time
                    # print(".......unchanged......")
                    output_tokens[i] = self.vocab.vocab.get(token, self.vocab.vocab['[UNK]'])
                    masked_pos.append(0)
                # True Label
                output_labels.append(self.vocab.vocab.get(token, self.vocab.vocab['[UNK]']))
                # masked_pos_label[i] = self.vocab.vocab.get(token, self.vocab.vocab['[UNK]'])
            else:
                # i-th token with original value
                output_tokens[i] = self.vocab.vocab.get(token, self.vocab.vocab['[UNK]'])
                # Padded label
                output_labels.append(self.vocab.vocab['[PAD]'])
                masked_pos.append(0)
        # label_position = []
        # label_tokens = []
        # for k, v in masked_pos_label.items():
        #     label_position.append(k)
        #     label_tokens.append(v)
        return  output_tokens, output_labels, masked_pos
    
#     def get_token_b(self, item):
#         document_id = [k for k,v in self.index_documents.items() if item in v][0]
#         random_document_id = document_id
        
#         if random.random() < 0.5:
#             document_ids = [k for k in self.index_documents.keys() if k != document_id]
#             random_document_id = random.choice(document_ids) 

#         same_student = (random_document_id == document_id)
        
#         nex_seq_list = self.index_documents.get(random_document_id)

#         if same_student:
#             if len(nex_seq_list) != 1:
#                 nex_seq_list = [v for v in nex_seq_list if v !=item]

#         next_seq = random.choice(nex_seq_list)
#         tokens = self.lines[next_seq]
#         # print(f"item = {item}, tokens: {tokens}")
#         # print(f"item={item}, next={next_seq}, same_student = {same_student}, {document_id} == {random_document_id}, b. {tokens}")
#         return same_student, tokens

#     def truncate_to_max_seq(self, s1, s2):
#         sa = copy.deepcopy(s1)
#         sb = copy.deepcopy(s1)
#         total_allowed_seq = self.seq_len - 3
        
#         while((len(sa)+len(sb)) > total_allowed_seq):
#             if random.random() < 0.5:
#                 sa.pop()
#             else:
#                 sb.pop()
#         return sa, sb
            
                
class TokenizerDataset(Dataset):
    """
        Class name: TokenizerDataset
        Tokenize the data in the dataset
        
    """
    def __init__(self, dataset_path, label_path, vocab, seq_len=30):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.vocab = vocab # Vocab object
        # self.encoder = OneHotEncoder(sparse=False)
        
        # Related to input dataset file
        self.lines = []
        self.labels = []
        self.feats = []
        if self.label_path:
            self.label_file = open(self.label_path, "r")
            for line in self.label_file:
                if line:
                    line = line.strip()
                    if not line:
                        continue
                    self.labels.append(int(line))
            self.label_file.close()
            
            # Comment this section if you are not using feat attribute
            try:
                j = 0
                dataset_info_file = open(self.label_path.replace("label", "info"), "r")
                for line in dataset_info_file:
                    if line:
                        line = line.strip()
                        if not line:
                            continue
                      
                        # # highGRschool_w_prior
                        # feat_vec = [float(i) for i in line.split(",")[-3].split("\t")]
                        
                        # highGRschool_w_prior_w_diffskill_wo_fa
                        feat_vec = [float(i) for i in line.split(",")[-3].split("\t")]
                        feat2 = [float(i) for i in line.split(",")[-2].split("\t")]
                        feat_vec.extend(feat2[1:])
                        
                        # # highGRschool_w_prior_w_p_diffskill_wo_fa
                        # feat_vec = [float(i) for i in line.split(",")[-3].split("\t")]
                        # feat2 = [-float(i) for i in line.split(",")[-2].split("\t")]
                        # feat_vec.extend(feat2[1:])
                        
#                         # highGRschool_w_prior_w_diffskill_0fa_skill
#                         feat_vec = [float(i) for i in line.split(",")[-3].split("\t")]
#                         feat2 = [float(i) for i in line.split(",")[-2].split("\t")]
#                         fa_feat_vec = [float(i) for i in line.split(",")[-1].split("\t")]
                        
#                         diff_skill = [f2 if f1==0 else 0 for f2, f1 in zip(feat2, fa_feat_vec)]
#                         feat_vec.extend(diff_skill)
                         
                        if j == 0:
                            print(len(feat_vec))
                            j+=1
                        
                        # feat_vec.extend(feat2[1:])
                        # feat_vec.extend(feat2)
                        # feat_vec = [float(i) for i in line.split(",")[-2].split("\t")]
                        # feat_vec = feat_vec[1:]
                        # feat_vec = [float(line.split(",")[-1])]
                        # feat_vec = [float(i) for i in line.split(",")[-1].split("\t")]
                        # feat_vec = [ft-f1 for ft, f1 in zip(feat_vec, fa_feat_vec)]

                        self.feats.append(feat_vec)
                dataset_info_file.close()
            except Exception as e:
                print(e)
            # labeler = np.array([0, 1]) #np.unique(self.labels)
            # print(f"Labeler {labeler}")
            # self.encoder.fit(labeler.reshape(-1,1))
            # self.labels = self.encoder.transform(np.array(self.labels).reshape(-1,1))

        self.file = open(self.dataset_path, "r")
        for line in self.file:
            if line:
                line = line.strip()
                if line:
                    self.lines.append(line)
        self.file.close()             
        
        self.len = len(self.lines)
        self.seq_len = seq_len
        print("Sequence length set at ", self.seq_len, len(self.lines), len(self.labels) if self.label_path else 0)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, item):
        org_line = self.lines[item].split("\t")
        dup_line = []
        opt = False
        for l in org_line:
            if l in ["OptionalTask_1", "EquationAnswer", "NumeratorFactor", "DenominatorFactor", "OptionalTask_2", "FirstRow1:1", "FirstRow1:2", "FirstRow2:1", "FirstRow2:2", "SecondRow", "ThirdRow"]:
                opt = True
            if opt and 'FinalAnswer-' in l: 
                dup_line.append('[UNK]')
            else:
                dup_line.append(l)
        dup_line = "\t".join(dup_line)
        # print(dup_line)
        s1 = self.vocab.to_seq(dup_line, self.seq_len) # This is like tokenizer and adds [CLS] and [SEP].
        s1_label = self.labels[item] if self.label_path else 0
        segment_label = [1 for _ in range(len(s1))]
        s1_feat = self.feats[item] if len(self.feats)>0 else 0
        padding = [self.vocab.vocab['[PAD]'] for _ in range(self.seq_len - len(s1))]
        s1.extend(padding), segment_label.extend(padding)
        
        output = {'input': s1,
                 'label': s1_label,
                  'feat': s1_feat,
                 'segment_label': segment_label}
        return {key: torch.tensor(value) for key, value in output.items()}
        
        
class TokenizerDatasetForCalibration(Dataset):
    """
        Class name: TokenizerDataset
        Tokenize the data in the dataset
        
    """
    def __init__(self, dataset_path, label_path, vocab, seq_len=30):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.vocab = vocab # Vocab object
        # self.encoder = OneHotEncoder(sparse=False)
        
        # Related to input dataset file
        self.lines = []
        self.labels = []
        self.feats = []
        if self.label_path:
            self.label_file = open(self.label_path, "r")
            for line in self.label_file:
                if line:
                    line = line.strip()
                    if not line:
                        continue
                    self.labels.append(int(line))
            self.label_file.close()
            
            # Comment this section if you are not using feat attribute
            try:
                j = 0
                dataset_info_file = open(self.label_path.replace("label", "info"), "r")
                for line in dataset_info_file:
                    if line:
                        line = line.strip()
                        if not line:
                            continue
                      
                        # # highGRschool_w_prior
                        # feat_vec = [float(i) for i in line.split(",")[-3].split("\t")]
                        
                        # highGRschool_w_prior_w_diffskill_wo_fa
                        feat_vec = [float(i) for i in line.split(",")[-3].split("\t")]
                        feat2 = [float(i) for i in line.split(",")[-2].split("\t")]
                        feat_vec.extend(feat2[1:])
                        
#                         # highGRschool_w_prior_w_diffskill_0fa_skill
#                         feat_vec = [float(i) for i in line.split(",")[-3].split("\t")]
#                         feat2 = [float(i) for i in line.split(",")[-2].split("\t")]
#                         fa_feat_vec = [float(i) for i in line.split(",")[-1].split("\t")]
                        
#                         diff_skill = [f2 if f1==0 else 0 for f2, f1 in zip(feat2, fa_feat_vec)]
#                         feat_vec.extend(diff_skill)
                         
                        if j == 0:
                            print(len(feat_vec))
                            j+=1
                        
                        # feat_vec.extend(feat2[1:])
                        # feat_vec.extend(feat2)
                        # feat_vec = [float(i) for i in line.split(",")[-2].split("\t")]
                        # feat_vec = feat_vec[1:]
                        # feat_vec = [float(line.split(",")[-1])]
                        # feat_vec = [float(i) for i in line.split(",")[-1].split("\t")]
                        # feat_vec = [ft-f1 for ft, f1 in zip(feat_vec, fa_feat_vec)]

                        self.feats.append(feat_vec)
                dataset_info_file.close()
            except Exception as e:
                print(e)
            # labeler = np.array([0, 1]) #np.unique(self.labels)
            # print(f"Labeler {labeler}")
            # self.encoder.fit(labeler.reshape(-1,1))
            # self.labels = self.encoder.transform(np.array(self.labels).reshape(-1,1))

        self.file = open(self.dataset_path, "r")
        for line in self.file:
            if line:
                line = line.strip()
                if line:
                    self.lines.append(line)
        self.file.close()             
        
        self.len = len(self.lines)
        self.seq_len = seq_len
        print("Sequence length set at ", self.seq_len, len(self.lines), len(self.labels) if self.label_path else 0)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, item):
        org_line = self.lines[item].split("\t")
        dup_line = []
        opt = False
        for l in org_line:
            if l in ["OptionalTask_1", "EquationAnswer", "NumeratorFactor", "DenominatorFactor", "OptionalTask_2", "FirstRow1:1", "FirstRow1:2", "FirstRow2:1", "FirstRow2:2", "SecondRow", "ThirdRow"]:
                opt = True
            if opt and 'FinalAnswer-' in l: 
                dup_line.append('[UNK]')
            else:
                dup_line.append(l)
        dup_line = "\t".join(dup_line)
        # print(dup_line)
        s1 = self.vocab.to_seq(dup_line, self.seq_len) # This is like tokenizer and adds [CLS] and [SEP].
        s1_label = self.labels[item] if self.label_path else 0
        segment_label = [1 for _ in range(len(s1))]
        s1_feat = self.feats[item] if len(self.feats)>0 else 0
        padding = [self.vocab.vocab['[PAD]'] for _ in range(self.seq_len - len(s1))]
        s1.extend(padding), segment_label.extend(padding)
        
        output = {'input': s1,
                 'label': s1_label,
                  'feat': s1_feat,
                 'segment_label': segment_label}
        return ({key: torch.tensor(value) for key, value in output.items()}, s1_label)
        
        
        
        # if __name__ == "__main__":
#     # import pickle
#     # k = pickle.load(open("dataset/CL4999_1920/unique_steps_list.pkl","rb"))
#     # print(k)
#     vocab_obj = Vocab("pretraining/vocab.txt")
#     vocab_obj.load_vocab()
#     datasetTrain = PretrainerDataset("pretraining/pretrain.txt", vocab_obj)
    
#     print(datasetTrain, len(datasetTrain))#, datasetTrain.documents_index)
#     print(datasetTrain[len(datasetTrain)-1])
#     for i, d in enumerate(datasetTrain):
#         print(d.items())
#         break
        
#     fine_tune = TokenizerDataset("finetuning/finetune.txt", "finetuning/finetune_label.txt", vocab_obj)
#     print(fine_tune)
#     print(fine_tune[len(fine_tune)-1])
#     print(fine_tune[random.randint(0, len(fine_tune))])
#     for i, d in enumerate(fine_tune):
#         print(d.items())
#         break
        
    