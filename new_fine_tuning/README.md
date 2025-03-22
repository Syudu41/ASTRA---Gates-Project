## Pre-training Data

### ratio_proportion_change3 : Calculating Percent Change and Final Amounts
> clear;python3 prepare_pretraining_input_vocab_file.py -analyze_dataset_by_section True -workspace_name ratio_proportion_change3 -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -pretrain True -train_file_path pretraining/pretrain1000.txt -train_info_path pretraining/pretrain1000_info.txt -test_file_path pretraining/test1000.txt -test_info_path pretraining/test1000_info.txt

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change3 -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -pretrain True -train_file_path pretraining/pretrain2000.txt -train_info_path pretraining/pretrain2000_info.txt -test_file_path pretraining/test2000.txt -test_info_path pretraining/test2000_info.txt

#### Test simple
> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change3 -code full -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path full.txt -train_info_path full_info.txt

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change3 -code gt -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path er.txt -train_info_path er_info.txt -test_file_path me.txt -test_info_path me_info.txt

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change3 -code correct -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path correct.txt -train_info_path correct_info.txt -test_file_path incorrect.txt -test_info_path incorrect_info.txt -final_step FinalAnswer

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change3 -code progress -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path graduated.txt -train_info_path graduated_info.txt -test_file_path promoted.txt -test_info_path promoted_info.txt 

### ratio_proportion_change4 : Using Percents and Percent Change
> clear;python3 prepare_pretraining_input_vocab_file.py -analyze_dataset_by_section True -workspace_name ratio_proportion_change4 -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor NumeratorLabel1 DenominatorLabel1 -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -pretrain True -train_file_path pretraining/pretrain1000.txt -train_info_path pretraining/pretrain1000_info.txt -test_file_path pretraining/test1000.txt -test_info_path pretraining/test1000_info.txt

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change4 -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor NumeratorLabel1 DenominatorLabel1 -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -pretrain True -train_file_path pretraining/pretrain2000.txt -train_info_path pretraining/pretrain2000_info.txt -test_file_path pretraining/test2000.txt -test_info_path pretraining/test2000_info.txt

#### Test simple
> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change4 -code full -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path full.txt -train_info_path full_info.txt

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change4 -code gt -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path er.txt -train_info_path er_info.txt -test_file_path me.txt -test_info_path me_info.txt

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change4 -code correct -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path correct.txt -train_info_path correct_info.txt -test_file_path incorrect.txt -test_info_path incorrect_info.txt -final_step FinalAnswer

> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change4 -code progress -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -train_file_path graduated.txt -train_info_path graduated_info.txt -test_file_path promoted.txt -test_info_path promoted_info.txt 

## Pretraining

### ratio_proportion_change3 : Calculating Percent Change and Final Amounts
> clear;python3 src/main.py -workspace_name ratio_proportion_change3_1920 -code pretrain1000 --pretrain_dataset pretraining/pretrain1000.txt --pretrain_val_dataset pretraining/test1000.txt
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000 --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt

#### Test simple models
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 1 --attn_heads 1

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 1 --attn_heads 2

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 2 --attn_heads 2

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 2 --attn_heads 4

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 4 --attn_heads 4

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 4 --attn_heads 8



### ratio_proportion_change4 : Using Percents and Percent Change
> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain1000 --pretrain_dataset pretraining/pretrain1000.txt --pretrain_val_dataset pretraining/test1000.txt
> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain2000 --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt

#### Test simple models
> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain2000_1l1h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 1 --attn_heads 1

> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain2000_1l2h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 1 --attn_heads 2

> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain2000_2l2h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 2 --attn_heads 2

> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain2000_2l4h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 2 --attn_heads 4

> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain2000_4l4h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 4 --attn_heads 4

> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -code pretrain2000_4l8h-5lr --pretrain_dataset pretraining/pretrain2000.txt --pretrain_val_dataset pretraining/test2000.txt --layers 4 --attn_heads 8


## Preparing Fine Tuning Data

### ratio_proportion_change3 : Calculating Percent Change and Final Amounts
> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change3 -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -final_step FinalAnswer

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -finetune_task check2 --train_dataset finetuning/check2/train.txt --test_dataset finetuning/check2/test.txt --train_label finetuning/check2/train_label.txt --test_label finetuning/check2/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/bert_trained.seq_encoder.model.ep279 --epochs 51

#### Attention Head Check
<!-- > PercentChange	NumeratorQuantity2	NumeratorQuantity1	DenominatorQuantity1	OptionalTask_1	EquationAnswer	NumeratorFactor	EquationAnswer	NumeratorFactor	EquationAnswer	NumeratorFactor	DenominatorFactor	NumeratorFactor	DenominatorFactor	NumeratorFactor	DenominatorFactor	FirstRow1:2	FirstRow1:1	FirstRow2:1	FirstRow2:2	FirstRow2:1	SecondRow	ThirdRow	FinalAnswerDirection	ThirdRow	FinalAnswer -->


> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset full/full.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep598 --attention True -finetune_task full;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset gt/er.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep598 --attention True -finetune_task er ;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset gt/me.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep598 --attention True -finetune_task me;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset correct/correct.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep598 --attention True -finetune_task correct ;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset correct/incorrect.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep598 --attention True -finetune_task incorrect;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset progress/graduated.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep598 --attention True -finetune_task graduated;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset progress/promoted.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep598 --attention True -finetune_task promoted 

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset full/full.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep823 --attention True -finetune_task full;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset gt/er.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep823 --attention True -finetune_task er;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset gt/me.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep823 --attention True -finetune_task me;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset correct/correct.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep823 --attention True -finetune_task correct;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset correct/incorrect.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep823 --attention True -finetune_task incorrect;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset progress/graduated.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep823 --attention True -finetune_task graduated;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset progress/promoted.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep823 --attention True -finetune_task promoted 

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --train_dataset full/full.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l2h-5lr/bert_trained.seq_encoder.model.ep1045 --attention True -finetune_task full;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --train_dataset gt/er.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l2h-5lr/bert_trained.seq_encoder.model.ep1045 --attention True -finetune_task er;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --train_dataset gt/me.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l2h-5lr/bert_trained.seq_encoder.model.ep1045 --attention True -finetune_task me;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --train_dataset correct/correct.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l2h-5lr/bert_trained.seq_encoder.model.ep1045 --attention True -finetune_task correct;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --train_dataset correct/incorrect.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l2h-5lr/bert_trained.seq_encoder.model.ep1045 --attention True -finetune_task incorrect;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --train_dataset progress/graduated.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l2h-5lr/bert_trained.seq_encoder.model.ep1045 --attention True -finetune_task graduated;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l2h-5lr --train_dataset progress/promoted.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l2h-5lr/bert_trained.seq_encoder.model.ep1045 --attention True -finetune_task promoted

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --train_dataset full/full.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l4h-5lr/bert_trained.seq_encoder.model.ep1336 --attention True -finetune_task full;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --train_dataset gt/er.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l4h-5lr/bert_trained.seq_encoder.model.ep1336 --attention True -finetune_task er;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --train_dataset gt/me.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l4h-5lr/bert_trained.seq_encoder.model.ep1336 --attention True -finetune_task me;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --train_dataset correct/correct.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l4h-5lr/bert_trained.seq_encoder.model.ep1336 --attention True -finetune_task correct;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --train_dataset correct/incorrect.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l4h-5lr/bert_trained.seq_encoder.model.ep1336 --attention True -finetune_task incorrect;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --train_dataset progress/graduated.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l4h-5lr/bert_trained.seq_encoder.model.ep1336 --attention True -finetune_task graduated;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_2l4h-5lr --train_dataset progress/promoted.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_2l4h-5lr/bert_trained.seq_encoder.model.ep1336 --attention True -finetune_task promoted

<!-- > clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset full/full.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep923 --attention True -->

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset full/full.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep871 --attention True -finetune_task full;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset gt/er.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep871 --attention True -finetune_task er;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset gt/me.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep871 --attention True -finetune_task me;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset correct/correct.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep871 --attention True -finetune_task correct;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset correct/incorrect.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep871 --attention True -finetune_task incorrect;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset progress/graduated.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep871 --attention True -finetune_task graduated;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l4h-5lr --train_dataset progress/promoted.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l4h-5lr/bert_trained.seq_encoder.model.ep871 --attention True -finetune_task promoted

clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset full/full_attn.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task full


> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset full/full.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task full;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset gt/er.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task er;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset gt/me.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task me;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset correct/correct.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task correct;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset correct/incorrect.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task incorrect;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset progress/graduated.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task graduated;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_4l8h-5lr --train_dataset progress/promoted.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_4l8h-5lr/bert_trained.seq_encoder.model.ep1349 --attention True -finetune_task promoted


<!-- PercentChange	NumeratorQuantity2	NumeratorQuantity1	DenominatorQuantity1	OptionalTask_2	FirstRow2:1	FirstRow2:2	FirstRow1:1	SecondRow	ThirdRow	FinalAnswer	FinalAnswerDirection --> me

<!-- PercentChange	NumeratorQuantity2	NumeratorQuantity1	DenominatorQuantity1	OptionalTask_1	DenominatorFactor	NumeratorFactor	OptionalTask_2	EquationAnswer	FirstRow1:1	FirstRow1:2	FirstRow2:2	FirstRow2:1	FirstRow1:2	SecondRow	ThirdRow	FinalAnswer --> er

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l1h-5lr --train_dataset pretraining/attention_train.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l1h-5lr/bert_trained.seq_encoder.model.ep273 --attention True

<!-- PercentChange	NumeratorQuantity2	NumeratorQuantity1	DenominatorQuantity1	OptionalTask_1	DenominatorFactor	NumeratorFactor	OptionalTask_2	EquationAnswer	FirstRow1:1	FirstRow1:2	FirstRow2:2	FirstRow2:1	FirstRow1:2	SecondRow	ThirdRow	FinalAnswer -->

> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -code pretrain2000_1l2h-5lr --train_dataset pretraining/attention_train.txt --pretrained_bert_checkpoint  ratio_proportion_change3/output/pretrain2000_1l2h-5lr/bert_trained.seq_encoder.model.ep1021 --attention True



### ratio_proportion_change4 : Using Percents and Percent Change
> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name ratio_proportion_change4 -opt_step1 OptionalTask_1 EquationAnswer NumeratorFactor DenominatorFactor NumeratorLabel1 DenominatorLabel1 -opt_step2 OptionalTask_2 FirstRow1:1 FirstRow1:2 FirstRow2:1 FirstRow2:2 SecondRow ThirdRow -final_step FinalAnswer

### scale_drawings_3 : Calculating Measurements Using a Scale
> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name scale_drawings_3 -opt_step1 opt1-check opt1-ratio-L-n opt1-ratio-L-d opt1-ratio-R-n opt1-ratio-R-d opt1-me2-top-3 opt1-me2-top-4 opt1-me2-top-2 opt1-me2-top-1 opt1-me2-middle-1 opt1-me2-bottom-1 -opt_step2 opt2-check opt2-ratio-L-n opt2-ratio-L-d opt2-ratio-R-n opt2-ratio-R-d opt2-me2-top-3 opt2-me2-top-4 opt2-me2-top-1 opt2-me2-top-2 opt2-me2-middle-1 opt2-me2-bottom-1 -final_step unk-value1 unk-value2

### sales_tax_discounts_two_rates : Solving Problems with Both Sales Tax and Discounts
> clear;python3 prepare_pretraining_input_vocab_file.py -workspace_name sales_tax_discounts_two_rates -opt_step1 optionalTaskGn salestaxFactor2 discountFactor2 multiplyOrderStatementGn -final_step totalCost1


# Fine Tuning Pre-trained model

## ratio_proportion_change3 : Calculating Percent Change and Final Amounts
> Selected Pretrained model: **ratio_proportion_change3/output/bert_trained.seq_encoder.model.ep279**
> New **bert/ratio_proportion_change3/output/pretrain2000/bert_trained.seq_encoder.model.ep731**

### 10per
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -finetune_task 10per --train_dataset finetuning/10per/train.txt --test_dataset finetuning/10per/test.txt --train_label finetuning/10per/train_label.txt --test_label finetuning/10per/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000/bert_trained.seq_encoder.model.ep731 --epochs 51

### IS
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -finetune_task IS --train_dataset finetuning/IS/train.txt --test_dataset finetuning/FS/train.txt --train_label finetuning/IS/train_label.txt --test_label finetuning/FS/train_label.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000/bert_trained.seq_encoder.model.ep731 --epochs 51

### FS
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -finetune_task FS --train_dataset finetuning/FS/train.txt --test_dataset finetuning/IS/train.txt --train_label finetuning/FS/train_label.txt --test_label finetuning/IS/train_label.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/pretrain2000/bert_trained.seq_encoder.model.ep731 --epochs 51
 
### correctness
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -finetune_task correctness --train_dataset finetuning/correctness/train.txt --test_dataset finetuning/correctness/test.txt --train_label finetuning/correctness/train_label.txt --test_label finetuning/correctness/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/bert_trained.seq_encoder.model.ep279 --epochs 51

### SL
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -finetune_task SL --train_dataset finetuning/SL/train.txt --test_dataset finetuning/SL/test.txt --train_label finetuning/SL/train_label.txt --test_label finetuning/SL/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/bert_trained.seq_encoder.model.ep279 --epochs 51

### effectiveness
> clear;python3 src/main.py -workspace_name ratio_proportion_change3 -finetune_task effectiveness --train_dataset finetuning/effectiveness/train.txt --test_dataset finetuning/effectiveness/test.txt --train_label finetuning/effectiveness/train_label.txt --test_label finetuning/effectiveness/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change3/output/bert_trained.seq_encoder.model.ep279 --epochs 51


## ratio_proportion_change4 : Using Percents and Percent Change
> Selected Pretrained model: **ratio_proportion_change4/output/bert_trained.seq_encoder.model.ep287**
### 10per
> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -finetune_task 10per --train_dataset finetuning/10per/train.txt --test_dataset finetuning/10per/test.txt --train_label finetuning/10per/train_label.txt --test_label finetuning/10per/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change4/output/bert_trained.seq_encoder.model.ep287 --epochs 51

### IS

### FS

### correctness
> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -finetune_task correctness --train_dataset finetuning/correctness/train.txt --test_dataset finetuning/correctness/test.txt --train_label finetuning/correctness/train_label.txt --test_label finetuning/correctness/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change4/output/bert_trained.seq_encoder.model.ep287 --epochs 51

### SL
> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -finetune_task SL --train_dataset finetuning/SL/train.txt --test_dataset finetuning/SL/test.txt --train_label finetuning/SL/train_label.txt --test_label finetuning/SL/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change4/output/bert_trained.seq_encoder.model.ep287 --epochs 51

### effectiveness
> clear;python3 src/main.py -workspace_name ratio_proportion_change4 -finetune_task effectiveness --train_dataset finetuning/effectiveness/train.txt --test_dataset finetuning/effectiveness/test.txt --train_label finetuning/effectiveness/train_label.txt --test_label finetuning/effectiveness/test_label.txt --pretrained_bert_checkpoint ratio_proportion_change4/output/bert_trained.seq_encoder.model.ep287 --epochs 51


## scale_drawings_3 : Calculating Measurements Using a Scale
> Selected Pretrained model: **scale_drawings_3/output/bert_trained.seq_encoder.model.ep252**
### 10per
> clear;python3 src/main.py -workspace_name scale_drawings_3 -finetune_task 10per --train_dataset finetuning/10per/train.txt --test_dataset finetuning/10per/test.txt --train_label finetuning/10per/train_label.txt --test_label finetuning/10per/test_label.txt --pretrained_bert_checkpoint scale_drawings_3/output/bert_trained.seq_encoder.model.ep252 --epochs 51

### IS

### FS

### correctness
> clear;python3 src/main.py -workspace_name scale_drawings_3 -finetune_task correctness --train_dataset finetuning/correctness/train.txt --test_dataset finetuning/correctness/test.txt --train_label finetuning/correctness/train_label.txt --test_label finetuning/correctness/test_label.txt --pretrained_bert_checkpoint scale_drawings_3/output/bert_trained.seq_encoder.model.ep252 --epochs 51

### SL
> clear;python3 src/main.py -workspace_name scale_drawings_3 -finetune_task SL --train_dataset finetuning/SL/train.txt --test_dataset finetuning/SL/test.txt --train_label finetuning/SL/train_label.txt --test_label finetuning/SL/test_label.txt --pretrained_bert_checkpoint scale_drawings_3/output/bert_trained.seq_encoder.model.ep252 --epochs 51

### effectiveness

## sales_tax_discounts_two_rates : Solving Problems with Both Sales Tax and Discounts
> Selected Pretrained model: **sales_tax_discounts_two_rates/output/bert_trained.seq_encoder.model.ep255**

### 10per
> clear;python3 src/main.py -workspace_name sales_tax_discounts_two_rates -finetune_task 10per --train_dataset finetuning/10per/train.txt --test_dataset finetuning/10per/test.txt --train_label finetuning/10per/train_label.txt --test_label finetuning/10per/test_label.txt --pretrained_bert_checkpoint sales_tax_discounts_two_rates/output/bert_trained.seq_encoder.model.ep255 --epochs 51

### IS

### FS

### correctness
> clear;python3 src/main.py -workspace_name sales_tax_discounts_two_rates -finetune_task correctness --train_dataset finetuning/correctness/train.txt --test_dataset finetuning/correctness/test.txt --train_label finetuning/correctness/train_label.txt --test_label finetuning/correctness/test_label.txt --pretrained_bert_checkpoint sales_tax_discounts_two_rates/output/bert_trained.seq_encoder.model.ep255 --epochs 51

### SL

### effectiveness