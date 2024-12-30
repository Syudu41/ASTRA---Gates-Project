import argparse

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from src.bert import BERT
from src.pretrainer import BERTTrainer, BERTFineTuneTrainer, BERTAttention
from src.dataset import PretrainerDataset, TokenizerDataset
from src.vocab import Vocab

import time
import os
import tqdm
import pickle

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
    parser.add_argument("-pretrained_bert_checkpoint", type=str, default=None, help="checkpoint of saved pretrained bert model")  #."output_feb09/bert_trained.model.ep40"
    parser.add_argument('-check_epoch', type=int, default=None)

    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model") #64
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers") #4
    parser.add_argument("-a", "--attn_heads", type=int, default=4, help="number of attention heads") #8
    parser.add_argument("-s", "--seq_len", type=int, default=50, help="maximum sequence length")

    parser.add_argument("-b", "--batch_size", type=int, default=500, help="number of batch_size") #64
    parser.add_argument("-e", "--epochs", type=int, default=50)#1501, help="number of epochs") #501
    # Use 50 for pretrain, and 10 for fine tune
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

    # Later run with cuda
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
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
    
    if args.attention:
        print(f"Attention aggregate...... code: {args.code}, dataset: {args.finetune_task}")
        if args.code:
            new_folder = f"{args.workspace_name}/plots/{args.code}/"
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
        
        train_dataset = TokenizerDataset(args.train_dataset_path, None, vocab_obj, seq_len=args.seq_len)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        print("Load Pre-trained BERT model")
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        device = torch.device("cuda:0" if cuda_condition else "cpu")
        bert = torch.load(args.pretrained_bert_checkpoint, map_location=device)
        trainer = BERTAttention(bert, vocab_obj, train_dataloader = train_data_loader, workspace_name = args.workspace_name, code=args.code, finetune_task = args.finetune_task)
        trainer.getAttention()
        
    elif args.embeddings:
        print("Get embeddings... and cluster... ")
        train_dataset = TokenizerDataset(args.test_dataset_path, None, vocab_obj, seq_len=args.seq_len)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        print("Load Pre-trained BERT model")
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        device = torch.device("cuda:0" if cuda_condition else "cpu")
        bert = torch.load(args.pretrained_bert_checkpoint).to(device)
        available_gpus = list(range(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            bert = nn.DataParallel(bert, device_ids=available_gpus)

        data_iter = tqdm.tqdm(enumerate(train_data_loader), 
                              desc="Model: %s" % (args.pretrained_bert_checkpoint.split("/")[-1]), 
                              total=len(train_data_loader), bar_format="{l_bar}{r_bar}")
        all_embeddings = []
        for i, data in data_iter:
            data = {key: value.to(device) for key, value in data.items()}
            embedding = bert(data["input"], data["segment_label"])
            # print(embedding.shape, embedding[:, 0].shape)
            embeddings = [h for h in embedding[:,0].cpu().detach().numpy()]
            all_embeddings.extend(embeddings)
            
        new_emb_folder = f"{args.workspace_name}/embeddings"
        if not os.path.exists(new_emb_folder):
            os.makedirs(new_emb_folder)
        pickle.dump(all_embeddings, open(f"{new_emb_folder}/{args.embeddings_file_name}.pkl", "wb"))
    else:
        if args.pretrain:
            print("Pre-training......")
            print("Loading Pretraining Train Dataset", args.train_dataset_path)
            print(f"Workspace: {args.workspace_name}")
            pretrain_dataset = PretrainerDataset(args.train_dataset_path, vocab_obj, seq_len=args.seq_len, max_mask = args.max_mask)

            print("Loading Pretraining Validation Dataset", args.val_dataset_path)
            pretrain_valid_dataset = PretrainerDataset(args.val_dataset_path, vocab_obj, seq_len=args.seq_len, max_mask = args.max_mask) \
                if args.val_dataset_path is not None else None
            
            print("Loading Pretraining Test Dataset", args.test_dataset_path)
            pretrain_test_dataset = PretrainerDataset(args.test_dataset_path, vocab_obj, seq_len=args.seq_len, max_mask = args.max_mask) \
                if args.test_dataset_path is not None else None

            print("Creating Dataloader")
            pretrain_data_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            pretrain_val_data_loader = DataLoader(pretrain_valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)\
                if pretrain_valid_dataset is not None else None
            pretrain_test_data_loader = DataLoader(pretrain_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)\
                if pretrain_test_dataset is not None else None
            
            print("Building BERT model")
            bert = BERT(len(vocab_obj.vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)

            if args.pretrained_bert_checkpoint:
                print(f"BERT model : {args.pretrained_bert_checkpoint}")
                bert = torch.load(args.pretrained_bert_checkpoint)
            
            new_log_folder = f"{args.workspace_name}/logs"
            new_output_folder = f"{args.workspace_name}/output"
            if args.code: # is sent almost all the time
                new_log_folder = f"{args.workspace_name}/logs/{args.code}"
                new_output_folder = f"{args.workspace_name}/output/{args.code}"
                            
            if not os.path.exists(new_log_folder):
                os.makedirs(new_log_folder)
            if not os.path.exists(new_output_folder):
                os.makedirs(new_output_folder)
                
            print(f"Creating BERT Trainer .... masking: True, max_mask: {args.max_mask}")
            trainer = BERTTrainer(bert, len(vocab_obj.vocab), train_dataloader=pretrain_data_loader, 
                                  val_dataloader=pretrain_val_data_loader, test_dataloader=pretrain_test_data_loader,
                                  lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, 
                                  with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, 
                                  log_folder_path=new_log_folder)

            start_time = time.time()
            print(f'Pretraining Starts, Time: {time.strftime("%D %T", time.localtime(start_time))}')
            # if need to pretrain from a check-point, need :check_epoch
            repoch = range(args.check_epoch, args.epochs) if args.check_epoch else range(args.epochs)
            counter = 0
            patience = 20
            for epoch in repoch:
                print(f'Training Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
                trainer.train(epoch)
                print(f'Training Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')

                if pretrain_val_data_loader is not None:
                    print(f'Validation Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
                    trainer.val(epoch)
                    print(f'Validation Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')

                if trainer.save_model: #  or epoch%10 == 0 and epoch > 4  
                    trainer.save(epoch, args.output_path)
                    counter = 0
                    if pretrain_test_data_loader is not None:
                        print(f'Test Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
                        trainer.test(epoch)
                        print(f'Test Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')
                else:
                    counter +=1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            end_time = time.time()
            print("Time Taken to pretrain model = ", end_time - start_time)
            print(f'Pretraining Ends, Time: {time.strftime("%D %T", time.localtime(end_time))}')
        else:
            print("Fine Tuning......")
            print("Loading Train Dataset", args.train_dataset_path)            
            train_dataset = TokenizerDataset(args.train_dataset_path, args.train_label_path, vocab_obj, seq_len=args.seq_len)

#             print("Loading Validation Dataset", args.val_dataset_path)            
#             val_dataset = TokenizerDataset(args.val_dataset_path, args.val_label_path, vocab_obj, seq_len=args.seq_len) \
#                 if args.val_dataset_path is not None else None
            
            print("Loading Test Dataset", args.test_dataset_path)
            test_dataset = TokenizerDataset(args.test_dataset_path, args.test_label_path, vocab_obj, seq_len=args.seq_len) \
                if args.test_dataset_path is not None else None

            print("Creating Dataloader...")
            train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            # val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
            #     if val_dataset is not None else None
            test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
                if test_dataset is not None else None

            print("Load Pre-trained BERT model")
            # bert = BERT(len(vocab_obj.vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
            cuda_condition = torch.cuda.is_available() and args.with_cuda
            device = torch.device("cuda:0" if cuda_condition else "cpu")
            bert = torch.load(args.pretrained_bert_checkpoint, map_location=device)

    #         if args.finetune_task == "SL":
    #             if args.workspace_name == "ratio_proportion_change4":
    #                 num_labels = 9
    #             elif args.workspace_name == "ratio_proportion_change3":
    #                 num_labels = 9
    #             elif args.workspace_name == "scale_drawings_3":
    #                 num_labels = 9
    #             elif args.workspace_name == "sales_tax_discounts_two_rates":
    #                 num_labels = 3
    #         else:
            # num_labels = 2
    #         # num_labels = 1
            # print(f"Number of Labels : {args.num_labels}")
            new_log_folder = f"{args.workspace_name}/logs"
            new_output_folder = f"{args.workspace_name}/output"
            if args.finetune_task: # is sent almost all the time
                new_log_folder = f"{args.workspace_name}/logs/{args.finetune_task}"
                new_output_folder = f"{args.workspace_name}/output/{args.finetune_task}"
                            
            if not os.path.exists(new_log_folder):
                os.makedirs(new_log_folder)
            if not os.path.exists(new_output_folder):
                os.makedirs(new_output_folder)
                
            print("Creating BERT Fine Tune Trainer")
            trainer = BERTFineTuneTrainer(bert, len(vocab_obj.vocab), 
                          train_dataloader=train_data_loader, test_dataloader=test_data_loader, 
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, 
                          with_cuda=args.with_cuda, cuda_devices = args.cuda_devices, log_freq=args.log_freq, 
                          workspace_name = args.workspace_name, num_labels=args.num_labels, log_folder_path=new_log_folder)

            print("Fine-tune training Start....")
            start_time = time.time()
            repoch = range(args.check_epoch, args.epochs) if args.check_epoch else range(args.epochs)
            counter = 0
            patience = 10
            for epoch in repoch:
                print(f'Training Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
                trainer.train(epoch)
                print(f'Training Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')

                if test_data_loader is not None:
                    print(f'Test Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
                    trainer.test(epoch)
                    # pickle.dump(trainer.probability_list, open(f"{args.workspace_name}/output/aaai/change4_mid_prob_{epoch}.pkl","wb"))
                    print(f'Test Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')

                # if val_data_loader is not None:
                #     print(f'Validation Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
                #     trainer.val(epoch)
                #     print(f'Validation Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')
                
                if trainer.save_model: #  or epoch%10 == 0
                    trainer.save(epoch, args.output_path)
                    counter = 0
                else:
                    counter +=1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                    
            end_time = time.time()
            print("Time Taken to fine-tune model = ", end_time - start_time)
            print(f'Pretraining Ends, Time: {time.strftime("%D %T", time.localtime(end_time))}')
            

if __name__ == "__main__":
    train()