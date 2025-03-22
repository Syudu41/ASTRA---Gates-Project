import subprocess
subprocess.run([
    "python", "new_test_saved_finetuned_model.py",
    "-workspace_name", "ratio_proportion_change3_2223/sch_largest_100-coded",
    "-finetune_task", "highGRschool10",
    "-finetuned_bert_classifier_checkpoint", 
    "ratio_proportion_change3_2223/sch_largest_100-coded/output/highGRschool10/bert_fine_tuned.model.ep42"
])