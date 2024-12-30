import argparse
import pickle
import random
import copy
import pandas as pd
import numpy as np
from collections import Counter
import os
from data_preprocessor import DataPreprocessor

def prepare_pretraining_files(data_processor, options):
    
    # steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")

    val_file = open(options.val_file_path, "w")
    val_info = open(options.val_info_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")


    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            # if options.workspace_name == section:
            if "ratio_proportion_change3" == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    # step_names_token = [step for step in prob_groups['Step Name'] if str(step) != 'nan']
                    # print(step_names_token)
                    
                    # writtenTrain = False
                    # writtenTest = False
                    
                    student_groups.sort_values(by="Time", inplace=True)
                    # prob_list = list(pd.unique(student_groups["Problem Name"]))
                    # print(len(prob_list), prob_list)
                    
                    # first_prob_list = prob_list[:3]
                    # last_prob_list = prob_list[-3:]
                    # print(len(first_prob_list), first_prob_list)
                    # print(len(last_prob_list), last_prob_list)
                    
                    # final_prob_list = first_prob_list + last_prob_list
                    # print(len(prob_list), len(final_prob_list), final_prob_list)
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # For first 3 and last 3 only
                        # if not prob in final_prob_list:
                        #     continue
                        # print(prob)
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups["Step Name"]))
                        unique_steps_len = len(set([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not(s in options.opt_step2)]))
                        if unique_steps_len < 4:
                            continue
                                                    
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 1800:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        # progress = ""
                        
                        step_names_token = []
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'CF (Etalon)', 'Outcome', 'Help Level','CF (Workspace Progress Status)']].iterrows():
                            
                            step = row["Step Name"]
                            progress = row["CF (Workspace Progress Status)"]
                            etalon = row["CF (Etalon)"]
                            
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                        
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                              
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                     
                        unique_steps_len = len(set([s for s in step_names_token if not (s in options.opt_step1) and not(s in options.opt_step2)]))

                        # 4 and more in sequence
                        if step_names_token and unique_steps_len > 4: 
                            # and len(step_names_token) > 3
                            # For information
                            # indices = [str(i) for i in prob_groups.index]
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            proba = random.random()
                            
                            # if prob in first_prob_list:
                            if proba <= 0.8:
                                # writtenTrain = True
                                train_file.write("\t".join(step_names_token))
                                train_file.write("\n")
                                # train_info.write(",".join([str(progress),str(prob), str(student), str(len(step_names_token)),
                                #                            "\t".join(map(str, outcome)), "\t".join(map(str, help_level))]))
                                # progress, problem name, student id, auto_complete, total steps length, er or me, outcome seq, help_level seq, encoding in steps length
                                train_info.write(",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)), 
                                                           f"{1 if means_and_extremes else 0}", "\t".join(map(str, outcome)), 
                                                           "\t".join(map(str, help_level)), "\t".join(map(str, where_opt))]))
                                train_info.write("\n")

                            elif proba > 0.9:
                            # elif prob in last_prob_list:
                            
                                # writtenTest = True

                                test_file.write("\t".join(step_names_token))
                                test_file.write("\n")
                                # test_info.write(",".join([str(progress),str(prob), str(student),  str(auto_complete), str(len(step_names_token)),
                                #                        "\t".join(map(str, outcome)), "\t".join(map(str, help_level))]))
                                # progress, problem name, student id, total steps length, er or me, outcome seq, help_level seq, encoding in steps length
                                test_info.write(",".join([str(progress),str(prob), str(student),  str(auto_complete), str(len(step_names_token)),
                                                f"{1 if means_and_extremes else 0}", "\t".join(map(str, outcome)), 
                                                  "\t".join(map(str, help_level)), "\t".join(map(str, where_opt))]))
                                test_info.write("\n")
                            else:
                                val_file.write("\t".join(step_names_token))
                                val_file.write("\n")
                                # test_info.write(",".join([str(progress),str(prob), str(student),  str(auto_complete), str(len(step_names_token)),
                                #                        "\t".join(map(str, outcome)), "\t".join(map(str, help_level))]))
                                # progress, problem name, student id, total steps length, er or me, outcome seq, help_level seq, encoding in steps length
                                val_info.write(",".join([str(progress),str(prob), str(student),  str(auto_complete), str(len(step_names_token)),
                                                f"{1 if means_and_extremes else 0}", "\t".join(map(str, outcome)), 
                                                  "\t".join(map(str, help_level)), "\t".join(map(str, where_opt))]))
                                val_info.write("\n")
                    # Indicates actions of next student
                    # Indicates next problem
                    # if writtenTrain:
                    #     train_file.write("\n")
                    #     train_info.write("\n")
                    # if writtenTest:
                    #     test_file.write("\n")
                    #     test_info.write("\n")
            # if not writtenTrain and not writtenTest:
            #     print(f"Student {student} is not involved in workspace : {options.workspace_name}.")


    train_file.close()
    train_info.close()
    
    val_file.close()
    val_info.close()
    
    test_file.close()
    test_info.close()

def prepare_school_pretraining_files(data_processor, options):
    
    # steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")

    val_file = open(options.val_file_path, "w")
    val_info = open(options.val_info_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")


    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                for class_id, class_group in school_group.groupby('CF (Anon Class Id)'):
                    for student, student_groups in class_group.groupby("Anon Student Id"):
                        student_groups.sort_values(by="Time")
                        # prob_list = list(pd.unique(student_groups["Problem Name"]))
                        for prob, prob_groups in student_groups.groupby("Problem Name"):
                            # For first 3 and last 3 only
                            # if not prob in final_prob_list:
                            #     continue
                            # print(prob)
                            step_names_token = []
                            means_and_extremes = False
                            for index, row in prob_groups[['Time', 'Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)', 'CF (Workspace Progress Status)', 'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                                progress = row["CF (Workspace Progress Status)"]
                                action = row["Action"]
                                attempt = row["Attempt At Step"]
                                autofilled = row["CF (Is Autofilled)"]
                                step = row["Step Name"]
                                scenario = row['CF (Problem Scenario Tags)']
                                
                                if not pd.isna(step):
                                    if step in options.opt_step1 and not means_and_extremes:
                                        etalon = row["CF (Etalon)"]
                                        if not pd.isna(etalon):
                                            etalon = etalon.strip('{}')
                                            key, value = etalon.split('=')
                                            etalon = value
                                            try:
                                                etalon = int(etalon)
                                            except Exception as e:
                                                try:
                                                    etalon = float(etalon)
                                                    means_and_extremes = True
                                                except Exception as e:
                                                    pass
                                
                                if not autofilled:
                                    new_step = f"{step}:{action}:{attempt}"
                                    step_names_token.append(new_step)
                            
                            if step_names_token:
                                where_opt = []
                                step1 = False
                                step2 = False
                                strategy_data = False
                                for step_oh in step_names_token:
                                    step = step_oh.split(":")
                                    if len(step) == 3:
                                        step = step[0]
                                    else:
                                        step = ":".join(step[:2])
                                        
                                    # print(f"changed {step_oh} = ? {step}")
                                    if step == options.opt_step1[0]:
                                        where_opt.append("_1")
                                        step1 = True
                                    elif step == options.opt_step2[0]:
                                        where_opt.append("_2")
                                        step2 = True
                                    elif step in options.opt_step1[1:]:
                                        where_opt.append("1")
                                        if step1:
                                            strategy_data = True
                                    elif step in options.opt_step2[1:]:
                                        where_opt.append("2")
                                        if step2:
                                            strategy_data = True
                                    else:
                                        where_opt.append("0")
                                        
                                if strategy_data and step_names_token[-1].split(":")[-2] != "Done":
                                    strategy_data = False
                                        
                                if strategy_data:
                                    proba = random.random()
                                    step_names_tokens = [":".join(s.split(":")[:-2]) for s in step_names_token]
                                    step_names_token = []
                                    for s in step_names_tokens:
                                        if s != "nan":
                                            if not step_names_token or s != step_names_token[-1]:
                                                step_names_token.append(s)
                                    # if prob in first_prob_list:
                                    if proba <= 0.8:
                                        train_file.write("\t".join(step_names_token))
                                        train_file.write("\n")
                                        # school, class, student id, progress, problem name, scenario, prefered ER or ME, total steps length, encoding in steps length
                                        train_info.write(",".join([str(school), str(class_id), str(student), str(progress), str(prob), str(scenario), f"{1 if means_and_extremes else 0}", str(len(step_names_token)), "\t".join(map(str, where_opt))]))
                                        train_info.write("\n")

                                    elif proba > 0.9:
                                    # elif prob in last_prob_list:
                                        test_file.write("\t".join(step_names_token))
                                        test_file.write("\n")
                                        # school, class, student id, progress, problem name, scenario, prefered ER or ME, total steps length, encoding in steps length
                                        test_info.write(",".join([str(school), str(class_id), str(student), str(progress), str(prob), str(scenario), f"{1 if means_and_extremes else 0}", str(len(step_names_token)), "\t".join(map(str, where_opt))]))
                                        test_info.write("\n")

                                    else:
                                        val_file.write("\t".join(step_names_token))
                                        val_file.write("\n")
                                        # school, class, student id, progress, problem name, scenario, prefered ER or ME, total steps length, encoding in steps length
                                        val_info.write(",".join([str(school), str(class_id), str(student), str(progress), str(prob), str(scenario), f"{1 if means_and_extremes else 0}", str(len(step_names_token)), "\t".join(map(str, where_opt))]))
                                        val_info.write("\n")
                            # break
                        # break
                    # break
                # break
        # break



    train_file.close()
    train_info.close()
    
    val_file.close()
    val_info.close()
    
    test_file.close()
    test_info.close()
    
def prepare_school_coded_pretraining_files(data_processor, options):
    
    # steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")

    val_file = open(options.val_file_path, "w")
    val_info = open(options.val_info_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")


    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                print(f"{school} : {school_group.shape}")
                school_group = school_group[(school_group['CF (Is StepByStep)'] == False) &
                                            (school_group['CF (Encounter)'] == 0) &
                                            (school_group['CF (Is Review Mode)'] == -1) ]
                print(f"{school} : {school_group.shape}")
                # for class_id, class_group in school_groups.groupby('CF (Anon Class Id)'):
                for student, student_groups in school_group.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)
                    # At least 3 last problems are selected
                    prob_list= list(student_groups[student_groups["CF (Workspace Progress Status)"]=="GRADUATED"]["Problem Name"])
                    prob_list = prob_list[-int(len(prob_list)/2):]
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        if not prob in prob_list:
                            continue
                        progress = list(pd.unique(prob_groups["CF (Workspace Progress Status)"]))[0]
                        if progress != "GRADUATED":
                            continue
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups[prob_groups["CF (Is Autofilled)"] == False]["Step Name"]))
                        unique_steps_len = len([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not (s in options.opt_step2)])
                        if unique_steps_len < 4:
                            continue
                        class_id = list(pd.unique(prob_groups["CF (Anon Class Id)"]))
                        step_names_token = []
                        original_steps_actions_attempts_help_levels_outcomes = []
                        original_steps = []
                        means_and_extremes = False
                        for index, row in prob_groups[['Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)',
                                                       'Outcome', 'Help Level', 'CF (Workspace Progress Status)', 
                                                       'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                            step = row["Step Name"]
                            action = row["Action"]            # ['Attempt', 'Hint Request', 'Hint Level Change', 'Done']
                            attempt = row["Attempt At Step"]  # number
                            outcome = row["Outcome"]          # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            help_level = row["Help Level"]    # number
                            # progress = row["CF (Workspace Progress Status)"]
                            scenario = row['CF (Problem Scenario Tags)']
                            
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    etalon = row["CF (Etalon)"]
                                    if not pd.isna(etalon):
                                        etalon = etalon.strip('{}')
                                        key, value = etalon.split('=')
                                        etalon = value
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                            except Exception as e:
                                                pass
                                if row['CF (Is Autofilled)'] == True:
                                    continue
                                prev = step_names_token[-1] if step_names_token else ""
                                prev_step = step_names_token[-1].split("-")[0] if step_names_token else ""

                                if not step_names_token or step != prev_step:
                                    if step in options.opt_step1 or step in options.opt_step2:
                                        new_step = step
                                    else:
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                    step_names_token.append(new_step)
                                else:
                                    if not (step in options.opt_step1 or step in options.opt_step2):
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                        if prev < new_step:
                                            step_names_token[-1] = new_step
                                original_steps_actions_attempts_help_levels_outcomes.append(f"{step}-{action}-{attempt}-{help_level}-{outcome}")
                                original_steps.append(step)
                                
                        unique_steps_len = len([s for s in original_steps if not (s in options.opt_step1) and not(s in options.opt_step2)])
                        if step_names_token and unique_steps_len > 4:
                            proba = random.random()
                            # if prob in first_prob_list:
                            if proba <= 0.8:
                                train_file.write("\t".join(step_names_token))
                                train_file.write("\n")
                                # school, class, student id, progress, problem name, scenario, 
                                # prefered ER or ME, total steps length, 
                                # original seq-action-attempt-help_level-outcome
                                train_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                               f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                               "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
                                train_info.write("\n")

                            elif proba > 0.9:
                            # elif prob in last_prob_list:
                                test_file.write("\t".join(step_names_token))
                                test_file.write("\n")
                                # school, class, student id, progress, problem name, scenario, 
                                # prefered ER or ME, total steps length, 
                                # original seq-action-attempt-help_level-outcome
                                test_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                               f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                               "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
                                test_info.write("\n")

                            else:
                                val_file.write("\t".join(step_names_token))
                                val_file.write("\n")
                                # school, class, student id, progress, problem name, scenario, 
                                # prefered ER or ME, total steps length, 
                                # original seq-action-attempt-help_level-outcome
                                val_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                               f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                               "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
                                val_info.write("\n")
                        # break
                    # break
                    # break
                # break
        # break



    train_file.close()
    train_info.close()
    
    val_file.close()
    val_info.close()
    
    test_file.close()
    test_info.close()


def prepare_school_attention_files(data_processor, options):
    
    # steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")

    val_file = open(options.val_file_path, "w")
    val_info = open(options.val_info_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")


    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                for class_id, class_group in school_group.groupby('CF (Anon Class Id)'):
                    for student, student_groups in class_group.groupby("Anon Student Id"):
                        student_groups.sort_values(by="Time")
#                         prob_list = list(pd.unique(student_groups["Problem Name"]))
#                         if len(prob_list) > 0 :
#                             first_fews = int(len(prob_list)/2)
#                             last_fews = len(prob_list) - first_fews
#                             first_prob_list = prob_list[:first_fews]
#                             last_prob_list = prob_list[-last_fews:]
                    
                    # final_prob_list = first_prob_list + last_prob_list
                        for prob, prob_groups in student_groups.groupby("Problem Name"):
                            step_names_token = []
                            means_and_extremes = False
                            for index, row in prob_groups[['Time', 'Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)', 'CF (Workspace Progress Status)', 'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                                progress = row["CF (Workspace Progress Status)"]
                                action = row["Action"]
                                attempt = row["Attempt At Step"]
                                autofilled = row["CF (Is Autofilled)"]
                                step = row["Step Name"]
                                scenario = row['CF (Problem Scenario Tags)']
                                
                                if not pd.isna(step):
                                    if step in options.opt_step1 and not means_and_extremes:
                                        etalon = row["CF (Etalon)"]
                                        if not pd.isna(etalon):
                                            etalon = etalon.strip('{}')
                                            key, value = etalon.split('=')
                                            etalon = value
                                            try:
                                                etalon = int(etalon)
                                            except Exception as e:
                                                try:
                                                    etalon = float(etalon)
                                                    means_and_extremes = True
                                                except Exception as e:
                                                    pass
                                
                                if not autofilled:
                                    new_step = f"{step}:{action}:{attempt}"
                                    step_names_token.append(new_step)
                            
                            if step_names_token:
                                where_opt = []
                                step1 = False
                                step2 = False
                                strategy_data = False
                                for step_oh in step_names_token:
                                    step = step_oh.split(":")
                                    if len(step) == 3:
                                        step = step[0]
                                    else:
                                        step = ":".join(step[:2])
                                        
                                    # print(f"changed {step_oh} = ? {step}")
                                    if step == options.opt_step1[0]:
                                        where_opt.append("_1")
                                        step1 = True
                                    elif step == options.opt_step2[0]:
                                        where_opt.append("_2")
                                        step2 = True
                                    elif step in options.opt_step1[1:]:
                                        where_opt.append("1")
                                        if step1:
                                            strategy_data = True
                                    elif step in options.opt_step2[1:]:
                                        where_opt.append("2")
                                        if step2:
                                            strategy_data = True
                                    else:
                                        where_opt.append("0")
                                        
                                if strategy_data and step_names_token[-1].split(":")[-2] != "Done":
                                    strategy_data = False
                                        
                                if strategy_data:
                                    # proba = random.random()
                                    step_names_tokens = [":".join(s.split(":")[:-2]) for s in step_names_token]
                                    step_names_token = []
                                    for s in step_names_tokens:
                                        if s != "nan":
                                            if not step_names_token or s != step_names_token[-1]:
                                                step_names_token.append(s)
                                    # if prob in first_prob_list:
                                    if progress == "GRADUATED":# and means_and_extremes:# and prob in first_prob_list:
                                        train_file.write("\t".join(step_names_token))
                                        train_file.write("\n")
                                        # school, class, student id, progress, problem name, scenario, prefered ER or ME, total steps length, encoding in steps length
                                        train_info.write(",".join([str(school), str(class_id), str(student), str(progress), str(prob), str(scenario), f"{1 if means_and_extremes else 0}", str(len(step_names_token)), "\t".join(map(str, where_opt))]))
                                        train_info.write("\n")

                                    elif progress == "PROMOTED":# and means_and_extremes:# and prob in last_prob_list:
                                    # elif prob in last_prob_list:
                                        test_file.write("\t".join(step_names_token))
                                        test_file.write("\n")
                                        # school, class, student id, progress, problem name, scenario, prefered ER or ME, total steps length, encoding in steps length
                                        test_info.write(",".join([str(school), str(class_id), str(student), str(progress), str(prob), str(scenario), f"{1 if means_and_extremes else 0}", str(len(step_names_token)), "\t".join(map(str, where_opt))]))
                                        test_info.write("\n")

                                    # else:
                                    #     val_file.write("\t".join(step_names_token))
                                    #     val_file.write("\n")
                                    #     # school, class, student id, progress, problem name, scenario, prefered ER or ME, total steps length, encoding in steps length
                                    #     val_info.write(",".join([str(school), str(class_id), str(student), str(progress), str(prob), str(scenario), f"{1 if means_and_extremes else 0}", str(len(step_names_token)), "\t".join(map(str, where_opt))]))
                                    #     val_info.write("\n")
                            # break
                        # break
                    # break
                # break
        # break



    train_file.close()
    train_info.close()
    
    val_file.close()
    val_info.close()
    
    test_file.close()
    test_info.close()
    
def prepare_finetuning_10per_files(data_processor, options):
    '''
        Used for L@S paper.
        Only two strategies were defined as:
        0: non-opt strategy
        1: opt used strategy
    '''
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
#     val_file = open(options.val_file_path, "w")
#     val_info = open(options.val_info_path, "w")
#     val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if "ratio_proportion_change3" == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups["Step Name"]))
                        unique_steps_len = len(set([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not(s in options.opt_step2)]))
                        if unique_steps_len < 4:
                            continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 1800:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                        
                        unique_steps_len = len(set([s for s in step_names_token if not (s in options.opt_step1) and not(s in options.opt_step2)]))
                        # 4 and more in sequence
                        if step_names_token and unique_steps_len > 4: 
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])
                                if any_opt1:
                                    label_opt = "1"
   
                            if options.opt_step2:
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "1"
                          
                            # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),
                                             "\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), 
                                             f"{1 if means_and_extremes else 0}"])
                            overall_data.append(["\t".join(step_names_token), info])
                            overall_labels.append(label_opt)
                            
                    # overall_data.append('')
                    # overall_labels.append('')
    
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    
    train_len = int(len(overall_labels) * 0.10)
    sample_size = int(train_len/2)
    print(f"sample_size: {sample_size}")
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))
    
    balanced_test = min(len(indices_of_zeros), len(indices_of_ones))
    test_sampled_instances = random.sample(indices_of_zeros, balanced_test)
    test_sampled_instances.extend(random.sample(indices_of_ones, balanced_test))

    # writtenTrain = False
    # writtenTest = False
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):

        steps_seq = all_data[0]
        info = all_data[1]

        if index in sampled_instances:
            train_file.write(steps_seq)
            train_file.write("\n")
            
            train_info.write(info)
            train_info.write("\n")
            
            train_label.write(label)
            train_label.write("\n")
        elif index in test_sampled_instances:
            # proba = random.random()
            # if proba <0.5:
            test_file.write(steps_seq)
            test_file.write("\n")

            test_info.write(info)
            test_info.write("\n")

            test_label.write(label)
            test_label.write("\n")
#             else:
#                 val_file.write(steps_seq)
#                 val_file.write("\n")

#                 val_info.write(info)
#                 val_info.write("\n")

#                 val_label.write(label)
#                 val_label.write("\n")

    train_file.close()
    train_info.close()
    train_label.close()
    
    # val_file.close()
    # val_info.close()
    # val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()

def prepare_finetuning_IS_FS_files(data_processor, options):
    '''
        Used for L@S paper. This function gathers first three problems of each student.
        Only two strategies were defined as:
        0: non-opt strategy
        1: opt used strategy
        train: IS
        test: FS
    '''
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
#     val_file = open(options.val_file_path, "w")
#     val_info = open(options.val_info_path, "w")
#     val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")

    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if "ratio_proportion_change3" == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)    
                    
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    if len(prob_list) < 3:
                        continue
                    selected = 3 #1. int(len(prob_list)/2)
                                #2. 3 & <6
                                #3. 3 & <3
                    first_prob_list = prob_list[:selected]
                    last_prob_list = prob_list[-selected:]
                            
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups["Step Name"]))
                        unique_steps_len = len(set([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not(s in options.opt_step2)]))
                        if unique_steps_len < 4:
                            continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 1800:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                        
                        unique_steps_len = len(set([s for s in step_names_token if not (s in options.opt_step1) and not(s in options.opt_step2)]))
                        # 4 and more in sequence
                        if step_names_token and unique_steps_len > 4: 
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])
                                if any_opt1:
                                    label_opt = "1"
   
                            if options.opt_step2:
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "1"
                          
                            # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),
                                             "\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), 
                                             f"{1 if means_and_extremes else 0}"])
                            if prob in first_prob_list:
                                train_file.write("\t".join(step_names_token))
                                train_file.write("\n")

                                train_info.write(info)
                                train_info.write("\n")

                                train_label.write(label_opt)
                                train_label.write("\n")
                            elif prob in last_prob_list:
                                test_file.write("\t".join(step_names_token))
                                test_file.write("\n")

                                test_info.write(info)
                                test_info.write("\n")

                                test_label.write(label_opt)
                                test_label.write("\n")

    train_file.close()
    train_info.close()
    train_label.close()
    
    # val_file.close()
    # val_info.close()
    # val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    
def prepare_finetuning_IS_files_old(data_processor, opts):
    '''
        Used for L@S paper. This function gathers first three problems of each student.
        Only two strategies were defined as:
        0: non-opt strategy
        1: opt used strategy
    '''
    
    options = copy.deepcopy(opts)
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test"):
            if v:
                f_path = v.split("/")
                f_path = f_path[0]+"/"+f_path[1]+"/IS/"+f_path[2]
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    trainr_label = open(options.trainr_label_path, "w")
    train_gt_label = open(options.train_gt_label_path, "w")
    
    # test_file = open(options.test_file_path, "w")
    # test_info = open(options.test_info_path, "w")
    # test_label = open(options.test_label_path, "w")
    # testr_label = open(options.testr_label_path, "w")
    # test_gt_label = open(options.test_gt_label_path, "w")
    
    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if options.workspace_name == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    
                    student_groups.sort_values(by="Time")
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    
                    if len(prob_list) < 3:
                        continue

                    first_prob_list = prob_list[:3]
#                     last_prob_list = prob_list[-3:]
#                     # print(len(first_prob_list), len(last_prob_list))
                    
#                     final_prob_list = first_prob_list + last_prob_list
                    # print(len(prob_list), len(final_prob_list)) #, final_prob_list)
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # For first 3 and last 3 only
                        if not prob in first_prob_list:
                            continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        progress = ""
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        finals = len(options.final_step)
                        totals = 0
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                    if finals == 0:
                                        totals += 1
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                
                        error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                        errors = 0
                        for step, out in zip(step_names_token, outcome):
                            if (finals and step in options.final_step) or totals > 0:
                                out = out.split(":")
                                if any(any(ind in o for o in out) for ind in error_ind):
                                    errors +=1
                                    
                        if finals:
                            totals = finals
                        # 4 and more in sequence
                        if step_names_token: # and len(step_names_token) > 3
                            
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])
                                if any_opt1:
                                    label_opt = "1"

                                
                            if options.opt_step2:
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "1"
                            
                            correctness = 1 - errors/totals
                            strat_correct = "0"
                            if correctness > 0.75:
                                strat_correct = "1" 
                                
                             # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),"\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), str(correctness)])
                            
                            overall_data.append(["\t".join(step_names_token), strat_correct, info, f"{1 if means_and_extremes else 0}"])
                            overall_labels.append(label_opt)
                            
                    overall_data.append('')
                    overall_labels.append('')   
                            
#     overall_labels = np.array(overall_labels)
#     indices_of_zeros = list(np.where(overall_labels == '0')[0])
#     indices_of_ones = list(np.where(overall_labels == '1')[0])
    
#     zeros_instances_size = int(1 * len(indices_of_zeros))
#     ones_instances_size = int(1 * len(indices_of_ones))
#     sample_size = min(zeros_instances_size, ones_instances_size)
#     sampled_instances = random.sample(indices_of_zeros, sample_size)
#     sampled_instances.extend(random.sample(indices_of_ones, sample_size))

    writtenTrain = False
    # writtenTest = False
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):        
        if all_data:
            steps_seq = all_data[0]
            strat_correct = all_data[1]
            info = all_data[2]
            me_opt = all_data[3]
            
            # if index in sampled_instances:
            writtenTrain = True
            train_file.write(steps_seq)
            train_file.write("\n")
            train_label.write(label)
            train_label.write("\n")
            trainr_label.write(strat_correct)
            trainr_label.write("\n")
            train_info.write(info)
            train_info.write("\n")
            train_gt_label.write(me_opt)
            train_gt_label.write("\n")
            # else:
            #     writtenTest = True
            #     test_file.write(steps_seq)
            #     test_file.write("\n")
            #     # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
            #     test_label.write(label)
            #     test_label.write("\n")
            #     # testr_label.write(str(correctness))
            #     testr_label.write(strat_correct)
            #     testr_label.write("\n")
            #     test_info.write(info)
            #     test_info.write("\n")
            #     test_gt_label.write(me_opt)
            #     test_gt_label.write("\n")
        else:
            # Indicates actions of next student
            # Indicates next problem
            if writtenTrain:
                writtenTrain = False
                train_file.write("\n")
                train_info.write("\n")
                train_label.write("\n")
                trainr_label.write("\n")
                train_gt_label.write("\n")
            # if writtenTest:
            #     writtenTest = False
            #     test_file.write("\n")
            #     test_info.write("\n")
            #     test_label.write("\n")
            #     testr_label.write("\n")
            #     test_gt_label.write("\n")                        

    train_file.close()
    train_info.close()
    train_label.close()
    trainr_label.close()
    train_gt_label.close()

    # test_file.close()
    # test_info.close()
    # test_label.close()
    # testr_label.close()
    # test_gt_label.close()
    
def prepare_finetuning_FS_files_old(data_processor, opts):
    '''
        Used for L@S paper. This function gathers last three problems of each student.
        Only two strategies were defined as:
        0: non-opt strategy
        1: opt used strategy
    '''
    
    options = copy.deepcopy(opts)
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test"):
            if v:
                f_path = v.split("/")
                f_path = f_path[0]+"/"+f_path[1]+"/FS/"+f_path[2]
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    trainr_label = open(options.trainr_label_path, "w")
    train_gt_label = open(options.train_gt_label_path, "w")
    
    # test_file = open(options.test_file_path, "w")
    # test_info = open(options.test_info_path, "w")
    # test_label = open(options.test_label_path, "w")
    # testr_label = open(options.testr_label_path, "w")
    # test_gt_label = open(options.test_gt_label_path, "w")

    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if options.workspace_name == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                  
                    student_groups.sort_values(by="Time")
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    
                    if len(prob_list) < 3:
                        continue

                    # first_prob_list = prob_list[:3]
                    last_prob_list = prob_list[-3:]
#                     # print(len(first_prob_list), len(last_prob_list))
                    
#                     final_prob_list = first_prob_list + last_prob_list
                    # print(len(prob_list), len(final_prob_list)) #, final_prob_list)
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # For first 3 and last 3 only
                        if not prob in last_prob_list:
                            continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        progress = ""
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        finals = len(options.final_step)
                        totals = 0
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                    if finals == 0:
                                        totals += 1
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                
                        error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                        errors = 0
                        for step, out in zip(step_names_token, outcome):
                            if (finals and step in options.final_step) or totals > 0:
                                out = out.split(":")
                                if any(any(ind in o for o in out) for ind in error_ind):
                                    errors +=1
                                    
                        if finals:
                            totals = finals
                        # 4 and more in sequence
                        if step_names_token: # and len(step_names_token) > 3
                            
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])
                                if any_opt1:
                                    label_opt = "1"

                                
                            if options.opt_step2:
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "1"
                            
                            correctness = 1 - errors/totals
                            strat_correct = "0"
                            if correctness > 0.75:
                                strat_correct = "1" 
                                
                             # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),"\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), str(correctness)])
                            
                            overall_data.append(["\t".join(step_names_token), strat_correct, info, f"{1 if means_and_extremes else 0}"])
                            overall_labels.append(label_opt)
                            
                    overall_data.append('')
                    overall_labels.append('')   
                            
#     overall_labels = np.array(overall_labels)
#     indices_of_zeros = list(np.where(overall_labels == '0')[0])
#     indices_of_ones = list(np.where(overall_labels == '1')[0])
    
#     zeros_instances_size = int(0.10 * len(indices_of_zeros))
#     ones_instances_size = int(0.10 * len(indices_of_ones))
#     sample_size = min(zeros_instances_size, ones_instances_size)
#     sampled_instances = random.sample(indices_of_zeros, sample_size)
#     sampled_instances.extend(random.sample(indices_of_ones, sample_size))

    writtenTrain = False
    # writtenTest = False
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):        
        if all_data:
            steps_seq = all_data[0]
            strat_correct = all_data[1]
            info = all_data[2]
            me_opt = all_data[3]
            
            # if index in sampled_instances:
            writtenTrain = True
            train_file.write(steps_seq)
            train_file.write("\n")
            train_label.write(label)
            train_label.write("\n")
            trainr_label.write(strat_correct)
            trainr_label.write("\n")
            train_info.write(info)
            train_info.write("\n")
            train_gt_label.write(me_opt)
            train_gt_label.write("\n")
            # else:
            #     writtenTest = True
            #     test_file.write(steps_seq)
            #     test_file.write("\n")
            #     # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
            #     test_label.write(label)
            #     test_label.write("\n")
            #     # testr_label.write(str(correctness))
            #     testr_label.write(strat_correct)
            #     testr_label.write("\n")
            #     test_info.write(info)
            #     test_info.write("\n")
            #     test_gt_label.write(me_opt)
            #     test_gt_label.write("\n")
        else:
            # Indicates actions of next student
            # Indicates next problem
            if writtenTrain:
                writtenTrain = False
                train_file.write("\n")
                train_info.write("\n")
                train_label.write("\n")
                trainr_label.write("\n")
                train_gt_label.write("\n")
            # if writtenTest:
            #     writtenTest = False
            #     test_file.write("\n")
            #     test_info.write("\n")
            #     test_label.write("\n")
            #     testr_label.write("\n")
            #     test_gt_label.write("\n")                        

    train_file.close()
    train_info.close()
    train_label.close()
    trainr_label.close()
    train_gt_label.close()

    # test_file.close()
    # test_info.close()
    # test_label.close()
    # testr_label.close()
    # test_gt_label.close()


def prepare_finetuning_correctness_files(data_processor, options):
    '''
        Ongoing research. Student strategy learning/predicting.
        FinalAnswer step
        Correct: 1 , correctness of final strategy > 0.75
        Incorrect: 0 , else < 0.75
    '''
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
#     val_file = open(options.val_file_path, "w")
#     val_info = open(options.val_info_path, "w")
#     val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if "ratio_proportion_change3" == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups["Step Name"]))
                        unique_steps_len = len(set([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not(s in options.opt_step2)]))
                        if unique_steps_len < 4:
                            continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 1800:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        final_correct = 0
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                if step == "FinalAnswer":
                                    final_correct += 1
                        unique_steps_len = len(set([s for s in step_names_token if not (s in options.opt_step1) and not(s in options.opt_step2)]))
                        # 4 and more in sequence
                        if step_names_token and unique_steps_len > 4: 
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if final_correct == 1:
                                label_opt = "1"
                          
                            # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),
                                             "\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), 
                                             f"{1 if means_and_extremes else 0}"])
                            overall_data.append(["\t".join(step_names_token), info])
                            overall_labels.append(label_opt)
                            
                    # overall_data.append('')
                    # overall_labels.append('')
    
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    
    train_len = int(len(overall_labels) * 0.10)
    sample_size = int(train_len/2)
    print(f"sample_size: {sample_size}")
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))

    # writtenTrain = False
    # writtenTest = False
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):

        steps_seq = all_data[0]
        info = all_data[1]

        if index in sampled_instances:
            train_file.write(steps_seq)
            train_file.write("\n")
            
            train_info.write(info)
            train_info.write("\n")
            
            train_label.write(label)
            train_label.write("\n")
        else:
            # proba = random.random()
            # if proba <0.5:
            test_file.write(steps_seq)
            test_file.write("\n")

            test_info.write(info)
            test_info.write("\n")

            test_label.write(label)
            test_label.write("\n")
#             else:
#                 val_file.write(steps_seq)
#                 val_file.write("\n")

#                 val_info.write(info)
#                 val_info.write("\n")

#                 val_label.write(label)
#                 val_label.write("\n")

    train_file.close()
    train_info.close()
    train_label.close()
    
    # val_file.close()
    # val_info.close()
    # val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()

def prepare_finetuning_correctness_files_old(data_processor, opts):
    '''
        Ongoing research. Student strategy learning/predicting.
        Correct, 1: correctness of final strategy > 0.75
        Incorrect, 0: else < 0.75
    '''
    options = copy.deepcopy(opts)
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test"):
            if v:
                f_path = v.split("/")
                f_path = f_path[0]+"/"+f_path[1]+"/fa_correctness/"+f_path[2]
                # f_path = f_path[0]+"/"+f_path[1]+"/check2/"+f_path[2]
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    # trainr_label = open(options.trainr_label_path, "w")
    # train_gt_label = open(options.train_gt_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    # testr_label = open(options.testr_label_path, "w")
    # test_gt_label = open(options.test_gt_label_path, "w")
    ws = "_".join(options.workspace_name.split("_")[:-1])
    print("Workspace: ", ws)
    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if ws == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    
                    student_groups.sort_values(by="Time")
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    
                    # if len(prob_list) < 3:
                    #     continue

#                     first_prob_list = prob_list[:3]
                    # last_prob_list = prob_list[-3:]
#                     # print(len(first_prob_list), len(last_prob_list))
                    
#                     final_prob_list = first_prob_list + last_prob_list
                    # print(len(prob_list), len(final_prob_list)) #, final_prob_list)
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # For first 3 and last 3 only
#                         if not prob in last_prob_list:
#                             continue
                        # print(options.final_step in list(prob_groups["Step Name"]))
                        # if not (options.final_step in list(prob_groups["Step Name"])):
                        #     continue
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        progress = ""
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        # finals = len(options.final_step)
                        
                       
                        totals = 0
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                    # if finals == 0:
                                    #     totals += 1
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                
                        error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                        errors = 0
                        for step, out in zip(step_names_token, outcome):
                            if (step in options.final_step):# or totals > 0:
                                out = out.split(":")
                                totals = len(out)
                                # print(totals)
                                for ind in error_ind:
                                    if ind in out:
                                        errors +=1
                                    
                        # if finals:
                        #     totals = finals
                        # 4 and more in sequence
                        if step_names_token and totals>0: # and len(step_names_token) > 3
                            
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                all_opt1 = all(any(opt in step for step in step_names_token) for opt in options.opt_step1)
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])

                                if any_opt1:
                                    label_opt = "2"
                                if all_opt1:
                                    label_opt = "1"

                                
                            if options.opt_step2:
                                all_opt2 = all(any(opt in step for step in step_names_token) for opt in options.opt_step2)
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "4"
                                if all_opt2:
                                    label_opt = "3"
                                if any_opt1 and any_opt2:
                                    label_opt = "5"
                                if any_opt1 and all_opt2:
                                    label_opt = "6"
                                if all_opt1 and any_opt2:
                                    label_opt = "7"
                                if all_opt1 and all_opt2:
                                    label_opt = "8"
                            
                            
                            correctness = 1 - errors/totals
                            strat_correct = "0"
                            if correctness > 0.75:
                                strat_correct = "1"
                            
                            # if not means_and_extremes and label_opt == "2":
                                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),"\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), str(correctness), f"{1 if means_and_extremes else 0}"])

                            overall_data.append(["\t".join(step_names_token), label_opt, info])
                            overall_labels.append(strat_correct)
                            
                    overall_data.append('')
                    overall_labels.append('')    
                            
    overall_labels = np.array(overall_labels, dtype=str)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    
    per = 0.20
    zeros_instances_size = int(per * len(indices_of_zeros))
    ones_instances_size = int(per * len(indices_of_ones))
    
    sample_size = min(zeros_instances_size, ones_instances_size)
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))

    writtenTrain = False
    writtenTest = False
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):        
        if all_data:
            steps_seq = all_data[0]
            label_opt = all_data[1]
            info = all_data[2]
            # me_opt = all_data[3]
            
            if index in sampled_instances:
                writtenTrain = True
                train_file.write(steps_seq)
                train_file.write("\n")
                train_label.write(label)
                train_label.write("\n")
                # trainr_label.write(label_opt)
                # trainr_label.write("\n")
                train_info.write(info)
                train_info.write("\n")
                # train_gt_label.write(me_opt)
                # train_gt_label.write("\n")
            else:
                writtenTest = True
                test_file.write(steps_seq)
                test_file.write("\n")
                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                test_label.write(label)
                test_label.write("\n")
                # testr_label.write(str(correctness))
                # testr_label.write(label_opt)
                # testr_label.write("\n")
                test_info.write(info)
                test_info.write("\n")
                # test_gt_label.write(me_opt)
                # test_gt_label.write("\n")
        else:
            # Indicates actions of next student
            # Indicates next problem
            if writtenTrain:
                writtenTrain = False
                train_file.write("\n")
                train_info.write("\n")
                train_label.write("\n")
                # trainr_label.write("\n")
                # train_gt_label.write("\n")
            if writtenTest:
                writtenTest = False
                test_file.write("\n")
                test_info.write("\n")
                test_label.write("\n")
                # testr_label.write("\n")
                # test_gt_label.write("\n")
        

    train_file.close()
    train_info.close()
    train_label.close()
    # trainr_label.close()
    # train_gt_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    # testr_label.close()
    # test_gt_label.close()

def prepare_finetuning_correctness_aaai_files(data_processor, opts):
    '''
        Ongoing research. Student strategy learning/predicting.
        Correct, 1: correctness of final strategy > 0.75
        Incorrect, 0: else < 0.75
    '''
    options = copy.deepcopy(opts)
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test") or k.startswith("val"):
            if v:
                f_path = v.split("/")
                # f_path = f_path[0]+"/"+f_path[1]+"/correctness/"+f_path[2]
                f_path = f_path[0]+"/"+f_path[1]+"/aaai/"+f_path[2]
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    # steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
    val_file = open(options.val_file_path, "w")
    val_info = open(options.val_info_path, "w")
    val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    high_performer = pickle.load(open(f"{options.workspace_name}/aaai/change3_high_performers.pkl", "rb"))
    mid_performer = pickle.load(open(f"{options.workspace_name}/aaai/change3_mid_performers.pkl", "rb"))
    low_performer = pickle.load(open(f"{options.workspace_name}/aaai/change3_low_performers.pkl", "rb"))
    prob_sel_list = pickle.load(open(f"{options.workspace_name}/aaai/change3_problem_list.pkl", "rb"))

    ws = "_".join(options.workspace_name.split("_")[:-1])

    print(ws, len(high_performer), len(mid_performer), len(low_performer), len(prob_sel_list))
    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            # if options.workspace_name == section:
            if ws == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    if student in high_performer or student in mid_performer or student in low_performer:
                        student_groups.sort_values(by="Time")
                        prob_list = list(pd.unique(student_groups["Problem Name"]))

                        for prob, prob_groups in student_groups.groupby("Problem Name"):
                            # For first 3 and last 3 only
                            if not prob in prob_sel_list:
                                continue

                            step_names_token = []

                            time_stamps = list(prob_groups["Time"])
                            time_stamps_list = set()
                            for i in range(len(time_stamps)-1):
                                if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                    time_stamps_list.add(time_stamps[i+1])

                            progress = ""
                            outcome = []
                            help_level = []
                            auto_complete = False
                            means_and_extremes = False
                            totals = 0

                            for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                                step = row["Step Name"]
                                etalon = row["CF (Etalon)"]
                                progress = row["CF (Workspace Progress Status)"]
                                if not pd.isna(step):
                                    if step in options.opt_step1:
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                                # break
                                            except Exception as e:
                                                pass
                                    if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                    # if row["Time"] in time_stamps_list:
                                        auto_complete = True
                                        # print(row)
                                        continue
                                    # if not step_names_token or step != step_names_token[-1]:
                                    #     step_names_token.append(step)

                                    if not step_names_token or step != step_names_token[-1]:
                                        step_names_token.append(step)
                                        # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                        outcome.append(row['Outcome'])
                                        help_level.append(str(row["Help Level"]))

                                    else:
                                        outcome[-1] = outcome[-1]+":"+row['Outcome']
                                        help_level[-1] = help_level[-1]+":"+str(row['Help Level'])

                            error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            errors = 0
                            for step, out in zip(step_names_token, outcome):
                                if (step in options.final_step):
                                    out = out.split(":")
                                    totals = len(out)
                                    # print(totals)
                                    for ind in error_ind:
                                        if ind in out:
                                            errors +=1

                            # 4 and more in sequence
                            if step_names_token and totals>0: # and len(step_names_token) > 3

                                where_opt = []
                                for stp in step_names_token:
                                    if stp in options.opt_step1:
                                        where_opt.append("1")
                                    elif stp in options.opt_step2:
                                        where_opt.append("2")
                                    else:
                                        where_opt.append("0")

                                

                                correctness = 1 - errors/totals
                                strat_correct = "0"
                                if correctness > 0.75:
                                    strat_correct = "1"

                                # if not means_and_extremes and label_opt == "2":
                                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                                info = ",".join([str(progress), str(correctness), f"{1 if means_and_extremes else 0}",str(prob), str(student), str(auto_complete), str(len(step_names_token)),"\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt))])

                                overall_data.append(["\t".join(step_names_token), info])
                                overall_labels.append(strat_correct)

                        # overall_data.append('')
                        # overall_labels.append('')    
                            
    overall_labels = np.array(overall_labels)
    
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):        
        if all_data:
            steps_seq = all_data[0]
            info = all_data[1]
            student = info.split(",")[4]
            
            if student in high_performer:
                train_file.write(steps_seq)
                train_file.write("\n")
                train_label.write(label)
                train_label.write("\n")
                train_info.write(info)
                train_info.write("\n")
            elif student in mid_performer:
                val_file.write(steps_seq)
                val_file.write("\n")
                val_label.write(label)
                val_label.write("\n")
                val_info.write(info)
                val_info.write("\n")
            elif student in low_performer:
                test_file.write(steps_seq)
                test_file.write("\n")
                test_label.write(label)
                test_label.write("\n")
                test_info.write(info)
                test_info.write("\n")

        

    train_file.close()
    train_info.close()
    train_label.close()
    
    val_file.close()
    val_info.close()
    val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()

def prepare_finetuning_SL_files(data_processor, opts):
    '''
        Ongoing research. Student strategy learning/predicting.
        We have defined 9 strategy as:
        Notation; Label
        UU; 0
        CU; 1
        PU; 2
        UC; 3
        UP; 4
        PP; 5
        PC; 6
        CP; 7
        CC; 8
    '''
    options = copy.deepcopy(opts)
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test"):
            if v:
                f_path = v.split("/")
                f_path = f_path[0]+"/"+f_path[1]+"/SL/"+f_path[2]
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    trainr_label = open(options.trainr_label_path, "w")
    train_gt_label = open(options.train_gt_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    testr_label = open(options.testr_label_path, "w")
    test_gt_label = open(options.test_gt_label_path, "w")

    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if options.workspace_name == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    
                    student_groups.sort_values(by="Time")
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    
                    # if len(prob_list) < 3:
                    #     continue

#                     first_prob_list = prob_list[:3]
                    # last_prob_list = prob_list[-3:]
#                     # print(len(first_prob_list), len(last_prob_list))
                    
#                     final_prob_list = first_prob_list + last_prob_list
                    # print(len(prob_list), len(final_prob_list)) #, final_prob_list)
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # For first 3 and last 3 only
                        # if not prob in last_prob_list:
                        #     continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        progress = ""
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        finals = len(options.final_step)
                        totals = 0
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                    if finals == 0:
                                        totals += 1
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                
                        error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                        errors = 0
                        for step, out in zip(step_names_token, outcome):
                            if (finals and step in options.final_step) or totals > 0:
                                out = out.split(":")
                                if any(any(ind in o for o in out) for ind in error_ind):
                                    errors +=1
                                    
                        if finals:
                            totals = finals
                        # 4 and more in sequence
                        if step_names_token: # and len(step_names_token) > 3
                            
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                all_opt1 = all(any(opt in step for step in step_names_token) for opt in options.opt_step1)
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])

                                if any_opt1:
                                    label_opt = "2"
                                if all_opt1:
                                    label_opt = "1"

                                
                            if options.opt_step2:
                                all_opt2 = all(any(opt in step for step in step_names_token) for opt in options.opt_step2)
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "4"
                                if all_opt2:
                                    label_opt = "3"
                                if any_opt1 and any_opt2:
                                    label_opt = "5"
                                if any_opt1 and all_opt2:
                                    label_opt = "6"
                                if all_opt1 and any_opt2:
                                    label_opt = "7"
                                if all_opt1 and all_opt2:
                                    label_opt = "8"
                            
                            
                            correctness = 1 - errors/totals
                            strat_correct = "0"
                            if correctness > 0.75:
                                strat_correct = "1"
                                
                            # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),"\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), str(correctness)])
                            
                            overall_data.append(["\t".join(step_names_token), strat_correct, info, f"{1 if means_and_extremes else 0}"])
                            overall_labels.append(label_opt)
                            
                    overall_data.append('')
                    overall_labels.append('')    
                            
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    indices_of_twos = list(np.where(overall_labels == '2')[0])
    indices_of_threes = list(np.where(overall_labels == '3')[0])
    indices_of_fours = list(np.where(overall_labels == '4')[0])
    indices_of_fives = list(np.where(overall_labels == '5')[0])
    indices_of_sixes = list(np.where(overall_labels == '6')[0])
    indices_of_sevens = list(np.where(overall_labels == '7')[0])
    indices_of_eights = list(np.where(overall_labels == '8')[0])
    
    per = 0.20
    zeros_instances_size = int(per * len(indices_of_zeros))
    ones_instances_size = int(per * len(indices_of_ones))
    twos_instances_size = int(per * len(indices_of_twos))
    threes_instances_size = int(per * len(indices_of_threes))
    fours_instances_size = int(per * len(indices_of_fours))
    fives_instances_size = int(per * len(indices_of_fives))
    sixes_instances_size = int(per * len(indices_of_sixes))
    sevens_instances_size = int(per * len(indices_of_sevens))
    eights_instances_size = int(per * len(indices_of_eights))

    sample_size = min(zeros_instances_size, ones_instances_size, twos_instances_size, threes_instances_size, fours_instances_size, fives_instances_size, sixes_instances_size, sevens_instances_size, eights_instances_size)
    print(f"Sample size.... {sample_size}")
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))
    sampled_instances.extend(random.sample(indices_of_twos, sample_size))
    sampled_instances.extend(random.sample(indices_of_threes, sample_size))
    sampled_instances.extend(random.sample(indices_of_fours, sample_size))
    sampled_instances.extend(random.sample(indices_of_fives, sample_size))
    sampled_instances.extend(random.sample(indices_of_sixes, sample_size))
    sampled_instances.extend(random.sample(indices_of_sevens, sample_size))
    sampled_instances.extend(random.sample(indices_of_eights, sample_size))

    writtenTrain = False
    writtenTest = False
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):        
        if all_data:
            steps_seq = all_data[0]
            strat_correct = all_data[1]
            info = all_data[2]
            me_opt = all_data[3]
            
            if index in sampled_instances:
                writtenTrain = True
                train_file.write(steps_seq)
                train_file.write("\n")
                train_label.write(label)
                train_label.write("\n")
                trainr_label.write(strat_correct)
                trainr_label.write("\n")
                train_info.write(info)
                train_info.write("\n")
                train_gt_label.write(me_opt)
                train_gt_label.write("\n")
            else:
                writtenTest = True
                test_file.write(steps_seq)
                test_file.write("\n")
                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                test_label.write(label)
                test_label.write("\n")
                # testr_label.write(str(correctness))
                testr_label.write(strat_correct)
                testr_label.write("\n")
                test_info.write(info)
                test_info.write("\n")
                test_gt_label.write(me_opt)
                test_gt_label.write("\n")
        else:
            # Indicates actions of next student
            # Indicates next problem
            if writtenTrain:
                writtenTrain = False
                train_file.write("\n")
                train_info.write("\n")
                train_label.write("\n")
                trainr_label.write("\n")
                train_gt_label.write("\n")
            if writtenTest:
                writtenTest = False
                test_file.write("\n")
                test_info.write("\n")
                test_label.write("\n")
                testr_label.write("\n")
                test_gt_label.write("\n")
        

    train_file.close()
    train_info.close()
    train_label.close()
    trainr_label.close()
    train_gt_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    testr_label.close()
    test_gt_label.close()

def prepare_finetuning_effectiveness_files(data_processor, opts):
    '''
        Ongoing research. Student strategy learning/predicting.
        We have defined 9 strategy as:
        Notation; Label
        UU; 0
        CU; 1
        PU; 2
        UC; 3
        UP; 4
        PP; 5
        PC; 6
        CP; 7
        CC; 8
        
        if UU and CU and PU and gt = ER and correct, a positive instance
        if UU and UC and UP and gt = ME and correct, a positive instance
        else a strategy PP, PC, CP, CC and gt = ER/ME or incorrect, a negative instance
    '''
    options = copy.deepcopy(opts)
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test"):
            if v:
                f_path = v.split("/")
                f_path = f_path[0]+"/"+f_path[1]+"/effectiveness/"+f_path[2]
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    trainr_label = open(options.trainr_label_path, "w")
    train_gt_label = open(options.train_gt_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    testr_label = open(options.testr_label_path, "w")
    test_gt_label = open(options.test_gt_label_path, "w")

    overall_data = []
    overall_labels = []
    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if options.workspace_name == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):    
                    student_groups.sort_values(by="Time")
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    
                    # if len(prob_list) < 3:
                    #     continue

#                     first_prob_list = prob_list[:3]
                    # last_prob_list = prob_list[-3:]
#                     # print(len(first_prob_list), len(last_prob_list))
                    
#                     final_prob_list = first_prob_list + last_prob_list
                    # print(len(prob_list), len(final_prob_list)) #, final_prob_list)
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # For first 3 and last 3 only
                        # if not prob in last_prob_list:
                        #     continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        progress = ""
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        finals = len(options.final_step)
                        totals = 0
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                    if finals == 0:
                                        totals += 1
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                
                        error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                        errors = 0
                        for step, out in zip(step_names_token, outcome):
                            if (finals and step in options.final_step) or totals > 0:
                                out = out.split(":")
                                if any(any(ind in o for o in out) for ind in error_ind):
                                    errors +=1
                                    
                        if finals:
                            totals = finals
                        # 4 and more in sequence
                        if step_names_token: # and len(step_names_token) > 3
                            
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                all_opt1 = all(any(opt in step for step in step_names_token) for opt in options.opt_step1)
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])

                                if any_opt1:
                                    label_opt = "2"
                                if all_opt1:
                                    label_opt = "1"

                                
                            if options.opt_step2:
                                all_opt2 = all(any(opt in step for step in step_names_token) for opt in options.opt_step2)
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "4"
                                if all_opt2:
                                    label_opt = "3"
                                if any_opt1 and any_opt2:
                                    label_opt = "5"
                                if any_opt1 and all_opt2:
                                    label_opt = "6"
                                if all_opt1 and any_opt2:
                                    label_opt = "7"
                                if all_opt1 and all_opt2:
                                    label_opt = "8"
                            
                            
                            correctness = 1 - errors/totals
                            strat_correct = "0"
                            if correctness > 0.75:
                                strat_correct = "1"
                            
                            label_effectiveness = "0"
                            if label_opt in ["0", "1", "2"] and not means_and_extremes and strat_correct == "1":
                                label_effectiveness = "1"
                            elif label_opt in ["0", "3", "4"] and means_and_extremes and strat_correct == "1":
                                label_effectiveness = "1"
                            # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                            info = ",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),"\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), str(correctness), label_opt, f"{1 if means_and_extremes else 0}"])
                            
                            overall_data.append(["\t".join(step_names_token), strat_correct, info, f"{1 if means_and_extremes else 0}"])
                            overall_labels.append(label_effectiveness)
                            
                    overall_data.append('')
                    overall_labels.append('')    
                            
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    
    per = 0.20
    zeros_instances_size = int(per * len(indices_of_zeros))
    ones_instances_size = int(per * len(indices_of_ones))

    sample_size = min(zeros_instances_size, ones_instances_size)
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))

    writtenTrain = False
    writtenTest = False
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):        
        if all_data:
            steps_seq = all_data[0]
            strat_correct = all_data[1]
            info = all_data[2]
            me_opt = all_data[3]
            
            if index in sampled_instances:
                writtenTrain = True
                train_file.write(steps_seq)
                train_file.write("\n")
                train_label.write(label)
                train_label.write("\n")
                trainr_label.write(strat_correct)
                trainr_label.write("\n")
                train_info.write(info)
                train_info.write("\n")
                train_gt_label.write(me_opt)
                train_gt_label.write("\n")
            else:
                writtenTest = True
                test_file.write(steps_seq)
                test_file.write("\n")
                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                test_label.write(label)
                test_label.write("\n")
                # testr_label.write(str(correctness))
                testr_label.write(strat_correct)
                testr_label.write("\n")
                test_info.write(info)
                test_info.write("\n")
                test_gt_label.write(me_opt)
                test_gt_label.write("\n")
        else:
            # Indicates actions of next student
            # Indicates next problem
            if writtenTrain:
                writtenTrain = False
                train_file.write("\n")
                train_info.write("\n")
                train_label.write("\n")
                trainr_label.write("\n")
                train_gt_label.write("\n")
            if writtenTest:
                writtenTest = False
                test_file.write("\n")
                test_info.write("\n")
                test_label.write("\n")
                testr_label.write("\n")
                test_gt_label.write("\n")
        

    train_file.close()
    train_info.close()
    train_label.close()
    trainr_label.close()
    train_gt_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    testr_label.close()
    test_gt_label.close()

def prepare_attn_test_files(data_processor, opts):
    options = copy.deepcopy(opts)
    
    if options.code:
        new_folder = f"{options.workspace_name}/{options.code}"
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
                
                    
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test"):
            if v:
                f_path = (f"/{options.code}/").join(v.split("/"))
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    
    if options.code != "full":
        test_file = open(options.test_file_path, "w")
        test_info = open(options.test_info_path, "w")

    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if options.workspace_name == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    
                    student_groups.sort_values(by="Time")
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        progress = ""
                        outcome = []
                        help_level = []
                        auto_complete = False
                        means_and_extremes = False
                        finals = len(options.final_step)
                        totals = 0
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                    if finals == 0:
                                        totals += 1
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                
                        error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                        errors = 0
                        for step, out in zip(step_names_token, outcome):
                            if (finals and step in options.final_step) or totals > 0:
                                out = out.split(":")
                                if any(any(ind in o for o in out) for ind in error_ind):
                                    errors +=1
                                    
                        if finals:
                            totals = finals
                        # 4 and more in sequence
                        if step_names_token: # and len(step_names_token) > 3
                            
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                all_opt1 = all(any(opt in step for step in step_names_token) for opt in options.opt_step1)
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])

                                if any_opt1:
                                    label_opt = "2"
                                if all_opt1:
                                    label_opt = "1"

                                
                            if options.opt_step2:
                                all_opt2 = all(any(opt in step for step in step_names_token) for opt in options.opt_step2)
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "4"
                                if all_opt2:
                                    label_opt = "3"
                                if any_opt1 and any_opt2:
                                    label_opt = "5"
                                if any_opt1 and all_opt2:
                                    label_opt = "6"
                                if all_opt1 and any_opt2:
                                    label_opt = "7"
                                if all_opt1 and all_opt2:
                                    label_opt = "8"
                            
                            
                            correctness = 1 - errors/totals
                            opt_correct = "0"
                            if correctness > 0.75:
                                opt_correct = "1"
                            
                            proba = random.random()
                            
                            # if proba <= 0.1:
                            # if not means_and_extremes:
                            # if prob in first_prob_list:
                            if options.code == "full" or (options.code == "gt" and not means_and_extremes) or (options.code == "correct" and opt_correct == "1") or (options.code == "progress" and progress == "GRADUATED"):
                                if label_opt == "0":
                                    continue
                                train_file.write("\t".join(step_names_token))
                                train_file.write("\n")
                                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                                train_info.write(",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),
                                                "\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), 
                                                           str(correctness), f"{1 if means_and_extremes else 0}", label_opt]))
                                train_info.write("\n")
                            # if means_and_extremes:
                            # if prob in last_prob_list:
                            else:
                                test_file.write("\t".join(step_names_token))
                                test_file.write("\n")
                                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                                test_info.write(",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),
                                                "\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt)), 
                                                          str(correctness), f"{1 if means_and_extremes else 0}", label_opt]))
                                test_info.write("\n")

    train_file.close()
    train_info.close()
    
    if options.code != "full":
        test_file.close()
        test_info.close()

def prepare_finetuning_future_files(data_processor, opts):
    options = copy.deepcopy(opts)
    for k,v in vars(opts).items():
        if k.startswith("train") or k.startswith("test"):
            if v:
                f_path = ("/effectiveness/").join(v.split("/"))
                setattr(options, f"{k}", f_path)
                print(f"options.{k} : {getattr(options, f'{k}')}")
                
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    chunk_iterator = data_processor.load_file_iterator()

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    trainr_label = open(options.trainr_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    testr_label = open(options.testr_label_path, "w")

    for chunk_data in chunk_iterator:
        for section, section_groups in chunk_data.groupby("Level (Workspace Id)"):
            if options.workspace_name == section:
                for student, student_groups in section_groups.groupby("Anon Student Id"):
                    writtenTrain = False
                    writtenTest = False
                    
                    student_groups.sort_values(by="Time")
                    prob_list = list(pd.unique(student_groups["Problem Name"]))
                    
                    # if len(prob_list) < 6:
                    #     continue

#                     first_prob_list = prob_list[:3]
#                     last_prob_list = prob_list[-3:]
#                     # print(len(first_prob_list), len(last_prob_list))
                    
#                     final_prob_list = first_prob_list + last_prob_list
                    # print(len(prob_list), len(final_prob_list)) #, final_prob_list)
                    
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # For first 3 and last 3 only
                        # if not prob in final_prob_list:
                        #     continue
                            
                        step_names_token = []
                        
                        time_stamps = list(prob_groups["Time"])
                        time_stamps_list = set()
                        for i in range(len(time_stamps)-1):
                            if (time_stamps[i+1] - time_stamps[i]) < 2000:
                                time_stamps_list.add(time_stamps[i+1])
                        
                        progress = ""
                        outcome = []
                        help_level = []
                        auto_complete = False
                        errors = 0
                        totals = 0
                        means_and_extremes = False
                        
                        for index, row in prob_groups[['Time', 'Step Name', 'Outcome', 'Help Level','CF (Workspace Progress Status)', 'CF (Etalon)']].iterrows():
                            step = row["Step Name"]
                            etalon = row["CF (Etalon)"]
                            progress = row["CF (Workspace Progress Status)"]
                            if not pd.isna(step):
                                if step in options.opt_step1:
                                    try:
                                        etalon = int(etalon)
                                    except Exception as e:
                                        try:
                                            etalon = float(etalon)
                                            means_and_extremes = True
                                            # break
                                        except Exception as e:
                                            pass
                                if (step in options.opt_step1 or step in options.opt_step2) and row["Time"] in time_stamps_list:
                                # if row["Time"] in time_stamps_list:
                                    auto_complete = True
                                    # print(row)
                                    continue
                                # if not step_names_token or step != step_names_token[-1]:
                                #     step_names_token.append(step)
                                
                                if not step_names_token or step != step_names_token[-1]:
                                    step_names_token.append(step)
                                    # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                                    outcome.append(row['Outcome'])
                                    help_level.append(str(row["Help Level"]))
                                    totals += 1
                                else:
                                    outcome[-1] = outcome[-1]+":"+row['Outcome']
                                    help_level[-1] = help_level[-1]+":"+str(row['Help Level'])
                                
                        error_ind = ['BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                        for out in outcome:
                            out = out.split(":")
                            if any(any(ind in o for o in out) for ind in error_ind):
                                errors +=1
                        # 4 and more in sequence
                        if step_names_token: # and len(step_names_token) > 3
                            
                            where_opt = []
                            for stp in step_names_token:
                                if stp in options.opt_step1:
                                    where_opt.append("1")
                                elif stp in options.opt_step2:
                                    where_opt.append("2")
                                else:
                                    where_opt.append("0")
                            
                            label_opt = "0"
                            if options.opt_step1:
                                all_opt1 = all(any(opt in step for step in step_names_token) for opt in options.opt_step1)
                                any_opt1 = any(any(opt in step for step in step_names_token) for opt in options.opt_step1[1:])

                                if any_opt1:
                                    label_opt = "2"
                                if all_opt1:
                                    label_opt = "1"

                                
                            if options.opt_step2:
                                all_opt2 = all(any(opt in step for step in step_names_token) for opt in options.opt_step2)
                                any_opt2 = any(any(opt in step for step in step_names_token) for opt in options.opt_step2[1:])
                                if any_opt2:
                                    label_opt = "4"
                                if all_opt2:
                                    label_opt = "3"
                                if any_opt1 and any_opt2:
                                    label_opt = "5"
                                if any_opt1 and all_opt2:
                                    label_opt = "6"
                                if all_opt1 and any_opt2:
                                    label_opt = "7"
                                if all_opt1 and all_opt2:
                                    label_opt = "8"
                            
                            
                            correctness = 1 - errors/totals
                            opt_correct = "0"
                            if correctness < 0.25:
                                opt_correct = "0"
                            elif correctness < 0.5:
                                opt_correct = "1"
                            elif correctness < 0.75:
                                opt_correct = "2"
                            else:
                                opt_correct = "3"
                                
                                
                            
                            proba = random.random()
                            
                            # if proba <= 0.1:
                            if not means_and_extremes:
                            # if prob in first_prob_list:
                                writtenTrain = True
                                train_file.write("\t".join(step_names_token))
                                train_file.write("\n")
                                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                                train_label.write(label_opt)
                                train_label.write("\n")
                                # trainr_label.write(str(correctness))
                                trainr_label.write(opt_correct)
                                trainr_label.write("\n")
                                train_info.write(",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),
                                                "\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt))]))
                                train_info.write("\n")

                            if means_and_extremes:
                            # if prob in last_prob_list:
                            # else:
                                writtenTest = True
                                test_file.write("\t".join(step_names_token))
                                test_file.write("\n")
                                # progress, problem name, student id, total steps length, outcome seq, help_level seq, encoding in steps length
                                test_label.write(label_opt)
                                test_label.write("\n")
                                # testr_label.write(str(correctness))
                                testr_label.write(opt_correct)
                                testr_label.write("\n")
                                test_info.write(",".join([str(progress),str(prob), str(student), str(auto_complete), str(len(step_names_token)),
                                                "\t".join(map(str, outcome)), "\t".join(map(str, help_level)), "\t".join(map(str, where_opt))]))
                                test_info.write("\n")
                    # Indicates actions of next student
                    # Indicates next problem
                    if writtenTrain:
                        train_file.write("\n")
                        train_info.write("\n")
                        train_label.write("\n")
                        trainr_label.write("\n")
                    if writtenTest:
                        test_file.write("\n")
                        test_info.write("\n")
                        test_label.write("\n")
                        testr_label.write("\n")


    train_file.close()
    train_info.close()
    train_label.close()
    trainr_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    testr_label.close()
    
def prepare_school_coded_finetuning_partial_seq_files(data_processor, options):
    '''
        Ongoing research.
        FinalAnswer step correctness
        Correct: 0 if attempt at step>1
                1 if attempt at step==1
    '''
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
#     val_file = open(options.val_file_path, "w")
#     val_info = open(options.val_info_path, "w")
#     val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    overall_data = []
    overall_labels = []
    # kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    # kcs = [kc if not pd.isna(kc) for kc in kcs]
    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                print(f"{school} : {school_group.shape}")
                school_group = school_group[(school_group['CF (Is StepByStep)'] == False) &
                                            (school_group['CF (Encounter)'] == 0) &
                                            (school_group['CF (Is Review Mode)'] == -1) ]
                print(f"{school} : {school_group.shape}")
                # for class_id, class_group in school_groups.groupby('CF (Anon Class Id)'):
                for student, student_groups in school_group.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)
                    # prob_list = list(pd.unique(student_groups["Problem Name"]))
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups[prob_groups["CF (Is Autofilled)"] == False]["Step Name"]))
                        unique_steps_len = len([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not (s in options.opt_step2)])
                        if unique_steps_len < 4:
                            continue
                        unique_opt_steps_len = len([s for s in unique_steps if not pd.isna(s) and (s in options.opt_step1[1:] or s in options.opt_step2[1:])])
                        if unique_opt_steps_len < 2:
                            continue
                            
                        class_id = list(pd.unique(prob_groups["CF (Anon Class Id)"]))
                        step_names_token = []
                        original_steps_actions_attempts_help_levels_outcomes = []
                        original_steps = []
                        means_and_extremes = False
                        correctness = "0"
                        opt_used = False
                        for index, row in prob_groups[['Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)',
                                                       'Outcome', 'Help Level', 'CF (Workspace Progress Status)', 
                                                       'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                            step = row["Step Name"]
                            action = row["Action"]            # ['Attempt', 'Hint Request', 'Hint Level Change', 'Done']
                            attempt = row["Attempt At Step"]  # number
                            outcome = row["Outcome"]          # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            help_level = row["Help Level"]    # number
                            progress = row["CF (Workspace Progress Status)"]
                            scenario = row['CF (Problem Scenario Tags)']
                            
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    etalon = row["CF (Etalon)"]
                                    if not pd.isna(etalon):
                                        etalon = etalon.strip('{}')
                                        key, value = etalon.split('=')
                                        etalon = value
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                            except Exception as e:
                                                pass
                                if row['CF (Is Autofilled)'] == True:
                                    continue
                                prev = step_names_token[-1] if step_names_token else ""
                                prev_step = step_names_token[-1].split("-")[0] if step_names_token else ""

                                if not step_names_token or step != prev_step:
                                    if step in options.opt_step1 or step in options.opt_step2:
                                        new_step = step
                                        opt_used = True
                                    else:
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                    if step != "FinalAnswer":
                                        step_names_token.append(new_step)
                                    else:
                                        step_names_token.append("FinalAnswer")
                                else:
                                    if not (step in options.opt_step1 or step in options.opt_step2 or step == "FinalAnswer"):
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                        if prev < new_step:
                                            step_names_token[-1] = new_step
                                
                                if step == "FinalAnswer" and opt_used:
                                    if attempt == 1 and outcome == "OK":
                                        correctness = "1"
                                    else:
                                        correctness = "0"
                                original_steps_actions_attempts_help_levels_outcomes.append(f"{step}-{action}-{attempt}-{help_level}-{outcome}")
                                original_steps.append(step)
                                
                        unique_steps_len = len([s for s in original_steps if not (s in options.opt_step1) and not(s in options.opt_step2)])
                        if step_names_token and unique_steps_len > 4:
                            info = ",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                           f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                           "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))])
                            overall_data.append(["\t".join(step_names_token), info])
                            overall_labels.append(correctness)
#                             proba = random.random()
#                             # if prob in first_prob_list:
#                             if proba <= 0.8:
#                                 train_file.write("\t".join(step_names_token))
#                                 train_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 train_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 train_info.write("\n")

#                             elif proba > 0.9:
#                             # elif prob in last_prob_list:
#                                 test_file.write("\t".join(step_names_token))
#                                 test_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 test_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 test_info.write("\n")

#                             else:
#                                 val_file.write("\t".join(step_names_token))
#                                 val_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 val_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 val_info.write("\n")
                        # break
                    # break
                    # break
                # break
        # break
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    
    train_len = int(len(overall_labels) * 0.10)
    sample_size = int(train_len/2)
    print(f"sample_size: {sample_size}")
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))

    indices_of_zeros = [i for i in indices_of_zeros if not i in sampled_instances ]
    indices_of_ones = [i for i in indices_of_ones if not i in sampled_instances ]

    balanced_test = min(len(indices_of_zeros), len(indices_of_ones))
    print(f"balanced_test: {balanced_test}")
    test_sampled_instances = random.sample(indices_of_zeros, balanced_test)
    test_sampled_instances.extend(random.sample(indices_of_ones, balanced_test))
    
    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):

        steps_seq = all_data[0]
        info = all_data[1]

        if index in sampled_instances:
            train_file.write(steps_seq)
            train_file.write("\n")
            
            train_info.write(info)
            train_info.write("\n")
            
            train_label.write(label)
            train_label.write("\n")
        elif index in test_sampled_instances:
            # proba = random.random()
            # if proba <0.5:
            test_file.write(steps_seq)
            test_file.write("\n")

            test_info.write(info)
            test_info.write("\n")

            test_label.write(label)
            test_label.write("\n")
#             else:
#                 val_file.write(steps_seq)
#                 val_file.write("\n")

#                 val_info.write(info)
#                 val_info.write("\n")

#                 val_label.write(label)
#                 val_label.write("\n")


    train_file.close()
    train_info.close()
    train_label.close()
    
    # val_file.close()
    # val_info.close()
    # val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    
def prepare_school_coded_finetuning_opts_files(data_processor, options):
    '''
        Ongoing research.
        Labels:
            0 - Opt 1
            1 - Opt 2
            2 - Both Opt
    '''
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
#     val_file = open(options.val_file_path, "w")
#     val_info = open(options.val_info_path, "w")
#     val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    overall_data = []
    overall_labels = []
    # kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    # kcs = [kc if not pd.isna(kc) for kc in kcs]
    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                print(f"{school} : {school_group.shape}")
                school_group = school_group[(school_group['CF (Is StepByStep)'] == False) &
                                            (school_group['CF (Encounter)'] == 0) &
                                            (school_group['CF (Is Review Mode)'] == -1) ]
                print(f"{school} : {school_group.shape}")
                # for class_id, class_group in school_groups.groupby('CF (Anon Class Id)'):
                for student, student_groups in school_group.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)
                    # prob_list = list(pd.unique(student_groups["Problem Name"]))
                    # prob_list= list(student_groups[student_groups["CF (Workspace Progress Status)"]=="GRADUATED"]["Problem Name"])
                    # prob_list = prob_list[-int(len(prob_list)/2):]
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # if not prob in prob_list:
                        #     continue
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups[prob_groups["CF (Is Autofilled)"] == False]["Step Name"]))
                        unique_steps_len = len([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not (s in options.opt_step2)])
                        if unique_steps_len < 4:
                            continue
                        unique_opt_steps_len = len([s for s in unique_steps if not pd.isna(s) and (s in options.opt_step1[1:] or s in options.opt_step2[1:])])
                        if unique_opt_steps_len < 2:
                            continue
                        print(unique_steps, unique_opt_steps_len)
                        class_id = list(pd.unique(prob_groups["CF (Anon Class Id)"]))
                        step_names_token = []
                        original_steps_actions_attempts_help_levels_outcomes = []
                        original_steps = []
                        means_and_extremes = False
                        opt1_used = False
                        opt2_used = False
                        for index, row in prob_groups[['Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)',
                                                       'Outcome', 'Help Level', 'CF (Workspace Progress Status)', 
                                                       'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                            step = row["Step Name"]
                            action = row["Action"]            # ['Attempt', 'Hint Request', 'Hint Level Change', 'Done']
                            attempt = row["Attempt At Step"]  # number
                            outcome = row["Outcome"]          # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            help_level = row["Help Level"]    # number
                            progress = row["CF (Workspace Progress Status)"]
                            scenario = row['CF (Problem Scenario Tags)']
                            
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    etalon = row["CF (Etalon)"]
                                    if not pd.isna(etalon):
                                        etalon = etalon.strip('{}')
                                        key, value = etalon.split('=')
                                        etalon = value
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                            except Exception as e:
                                                pass
                                if row['CF (Is Autofilled)'] == True:
                                    continue
                                prev = step_names_token[-1] if step_names_token else ""
                                prev_step = step_names_token[-1].split("-")[0] if step_names_token else ""

                                if not step_names_token or step != prev_step:
                                    if step in options.opt_step1 or step in options.opt_step2:
                                        new_step = step
                                        if step in options.opt_step1[1:]:
                                            opt1_used = True
                                        elif step in options.opt_step2[2:]:
                                            opt2_used = True
                                    else:
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                    step_names_token.append(new_step)
                                    
                                else:
                                    if not (step in options.opt_step1 or step in options.opt_step2 or step == "FinalAnswer"):
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                        if prev < new_step:
                                            step_names_token[-1] = new_step
                                
                                original_steps_actions_attempts_help_levels_outcomes.append(f"{step}-{action}-{attempt}-{help_level}-{outcome}")
                                original_steps.append(step)
                        if (not opt1_used) and (not opt2_used):
                            continue
                        unique_steps_len = len([s for s in original_steps if not (s in options.opt_step1) and not(s in options.opt_step2)])
                        if step_names_token and unique_steps_len > 4:
                            info = ",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                           f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                           "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))])
                            overall_data.append(["\t".join(step_names_token), info])
                            label = None
                            if opt1_used and opt2_used:
                                label = "2"
                            if (not opt1_used) and opt2_used:
                                label = "1"
                            if opt1_used and (not opt2_used):
                                label = "0"
                            print(f"opt1_used: {opt1_used}, opt2_used: {opt2_used} label : {label}")
                            overall_labels.append(label)
#                             proba = random.random()
#                             # if prob in first_prob_list:
#                             if proba <= 0.8:
#                                 train_file.write("\t".join(step_names_token))
#                                 train_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 train_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 train_info.write("\n")

#                             elif proba > 0.9:
#                             # elif prob in last_prob_list:
#                                 test_file.write("\t".join(step_names_token))
#                                 test_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 test_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 test_info.write("\n")

#                             else:
#                                 val_file.write("\t".join(step_names_token))
#                                 val_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 val_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 val_info.write("\n")
                        # break
                    # break
                    # break
                # break
        # break
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    indices_of_twos = list(np.where(overall_labels == '2')[0])
    
    train_len = int(len(overall_labels) * 0.10)
    sample_size = int(train_len/3)
    print(f"sample_size: {sample_size}")
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))
    sampled_instances.extend(random.sample(indices_of_twos, sample_size))

    indices_of_zeros = [i for i in indices_of_zeros if not i in sampled_instances ]
    indices_of_ones = [i for i in indices_of_ones if not i in sampled_instances ]
    indices_of_twos = [i for i in indices_of_twos if not i in sampled_instances ]

    balanced_test = min(len(indices_of_zeros), len(indices_of_ones), len(indices_of_twos))
    print(f"balanced_test: {balanced_test}")
    test_sampled_instances = random.sample(indices_of_zeros, balanced_test)
    test_sampled_instances.extend(random.sample(indices_of_ones, balanced_test))
    test_sampled_instances.extend(random.sample(indices_of_twos, balanced_test))

    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):

        steps_seq = all_data[0]
        info = all_data[1]

        if index in sampled_instances:
            train_file.write(steps_seq)
            train_file.write("\n")
            
            train_info.write(info)
            train_info.write("\n")
            
            train_label.write(label)
            train_label.write("\n")
        elif index in test_sampled_instances:
            # proba = random.random()
            # if proba <0.5:
            test_file.write(steps_seq)
            test_file.write("\n")

            test_info.write(info)
            test_info.write("\n")

            test_label.write(label)
            test_label.write("\n")
#             else:
#                 val_file.write(steps_seq)
#                 val_file.write("\n")

#                 val_info.write(info)
#                 val_info.write("\n")

#                 val_label.write(label)
#                 val_label.write("\n")


    train_file.close()
    train_info.close()
    train_label.close()
    
    # val_file.close()
    # val_info.close()
    # val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    
def prepare_school_coded_finetuning_opts_intentional_files(data_processor, options):
    '''
        Ongoing research.
        Labels:
            0 - Opt 1
            1 - Opt 2
            2 - Both Opt
    '''
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
    val_file = open(options.val_file_path, "w")
    val_info = open(options.val_info_path, "w")
    val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    # overall_data = []
    # overall_labels = []
    # kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    # kcs = [kc if not pd.isna(kc) for kc in kcs]
    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                print(f"{school} : {school_group.shape}")
                school_group = school_group[(school_group['CF (Is StepByStep)'] == False) &
                                            (school_group['CF (Encounter)'] == 0) &
                                            (school_group['CF (Is Review Mode)'] == -1) ]
                print(f"{school} : {school_group.shape}")
                # for class_id, class_group in school_groups.groupby('CF (Anon Class Id)'):
                for student, student_groups in school_group.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)
                    # prob_list = list(pd.unique(student_groups["Problem Name"]))
                    prob_list= list(pd.unique(student_groups[student_groups["CF (Workspace Progress Status)"]=="GRADUATED"]["Problem Name"]))
                    # prob_list = prob_list[-int(len(prob_list)/2):]
                    if len(prob_list) == 0:
                        continue
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # if not prob in prob_list:
                        #     continue
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups[prob_groups["CF (Is Autofilled)"] == False]["Step Name"]))
                        unique_steps_len = len([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not (s in options.opt_step2)])
                        if unique_steps_len < 4:
                            continue
                        unique_opt_steps_len = len([s for s in unique_steps if not pd.isna(s) and (s in options.opt_step1[1:] or s in options.opt_step2[1:])])
                        if unique_opt_steps_len < 2:
                            continue
                        # print(unique_steps, unique_opt_steps_len)
                        class_id = list(pd.unique(prob_groups["CF (Anon Class Id)"]))
                        step_names_token = []
                        original_steps_actions_attempts_help_levels_outcomes = []
                        original_steps = []
                        means_and_extremes = False
                        opt1_used = False
                        opt2_used = False
                        for index, row in prob_groups[['Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)',
                                                       'Outcome', 'Help Level', 'CF (Workspace Progress Status)', 
                                                       'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                            step = row["Step Name"]
                            action = row["Action"]            # ['Attempt', 'Hint Request', 'Hint Level Change', 'Done']
                            attempt = row["Attempt At Step"]  # number
                            outcome = row["Outcome"]          # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            help_level = row["Help Level"]    # number
                            progress = row["CF (Workspace Progress Status)"]
                            scenario = row['CF (Problem Scenario Tags)']
                            
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    etalon = row["CF (Etalon)"]
                                    if not pd.isna(etalon):
                                        etalon = etalon.strip('{}')
                                        key, value = etalon.split('=')
                                        etalon = value
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                            except Exception as e:
                                                pass
                                if row['CF (Is Autofilled)'] == True:
                                    continue
                                prev = step_names_token[-1] if step_names_token else ""
                                prev_step = step_names_token[-1].split("-")[0] if step_names_token else ""

                                if not step_names_token or step != prev_step:
                                    if step in options.opt_step1 or step in options.opt_step2:
                                        new_step = step
                                        if step in options.opt_step1[1:]:
                                            opt1_used = True
                                        elif step in options.opt_step2[2:]:
                                            opt2_used = True
                                    else:
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                    step_names_token.append(new_step)
                                    
                                else:
                                    if not (step in options.opt_step1 or step in options.opt_step2 or step == "FinalAnswer"):
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                        if prev < new_step:
                                            step_names_token[-1] = new_step
                                
                                original_steps_actions_attempts_help_levels_outcomes.append(f"{step}-{action}-{attempt}-{help_level}-{outcome}")
                                original_steps.append(step)
                        # if (not opt1_used) and (not opt2_used):
                        #     continue
                        unique_steps_len = len([s for s in original_steps if not (s in options.opt_step1) and not(s in options.opt_step2)])
                        if step_names_token and unique_steps_len > 4:
                            info = ",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                           f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                           "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))])
                            # overall_data.append(["\t".join(step_names_token), info])
                            # label = None
                            # if opt1_used and opt2_used:
                            #     label = "2"
                            # if (not opt1_used) and opt2_used:
                            #     label = "1"
                            # if opt1_used and (not opt2_used):
                            #     label = "0"
                            # print(f"opt1_used: {opt1_used}, opt2_used: {opt2_used} label : {label}")
                            # overall_labels.append(label)
                            
                            proba = random.random()
                            # if prob in first_prob_list:
                            if proba <= 0.8:
                                train_file.write("\t".join(step_names_token))
                                train_file.write("\n")
                                # school, class, student id, progress, problem name, scenario, 
                                # prefered ER or ME, total steps length, 
                                # original seq-action-attempt-help_level-outcome
                                train_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                               f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                               "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
                                train_info.write("\n")

                            elif proba > 0.9:
                            # elif prob in last_prob_list:
                                test_file.write("\t".join(step_names_token))
                                test_file.write("\n")
                                # school, class, student id, progress, problem name, scenario, 
                                # prefered ER or ME, total steps length, 
                                # original seq-action-attempt-help_level-outcome
                                test_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                               f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                               "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
                                test_info.write("\n")

                            else:
                                val_file.write("\t".join(step_names_token))
                                val_file.write("\n")
                                # school, class, student id, progress, problem name, scenario, 
                                # prefered ER or ME, total steps length, 
                                # original seq-action-attempt-help_level-outcome
                                val_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                               f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                               "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
                                val_info.write("\n")
                        # break
                    # break
                    # break
                # break
        # break
#     overall_labels = np.array(overall_labels)
#     indices_of_zeros = list(np.where(overall_labels == '0')[0])
#     indices_of_ones = list(np.where(overall_labels == '1')[0])
#     indices_of_twos = list(np.where(overall_labels == '2')[0])
    
#     train_len = int(len(overall_labels) * 0.10)
#     sample_size = int(train_len/3)
#     print(f"sample_size: {sample_size}")
#     sampled_instances = random.sample(indices_of_zeros, sample_size)
#     sampled_instances.extend(random.sample(indices_of_ones, sample_size))
#     sampled_instances.extend(random.sample(indices_of_twos, sample_size))

#     indices_of_zeros = [i for i in indices_of_zeros if not i in sampled_instances ]
#     indices_of_ones = [i for i in indices_of_ones if not i in sampled_instances ]
#     indices_of_twos = [i for i in indices_of_twos if not i in sampled_instances ]

#     balanced_test = min(len(indices_of_zeros), len(indices_of_ones), len(indices_of_twos))
#     print(f"balanced_test: {balanced_test}")
#     test_sampled_instances = random.sample(indices_of_zeros, balanced_test)
#     test_sampled_instances.extend(random.sample(indices_of_ones, balanced_test))
#     test_sampled_instances.extend(random.sample(indices_of_twos, balanced_test))

#     for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):

#         steps_seq = all_data[0]
#         info = all_data[1]

#         if index in sampled_instances:
#             train_file.write(steps_seq)
#             train_file.write("\n")
            
#             train_info.write(info)
#             train_info.write("\n")
            
#             train_label.write(label)
#             train_label.write("\n")
#         elif index in test_sampled_instances:
#             # proba = random.random()
#             # if proba <0.5:
#             test_file.write(steps_seq)
#             test_file.write("\n")

#             test_info.write(info)
#             test_info.write("\n")

#             test_label.write(label)
#             test_label.write("\n")
# #             else:
# #                 val_file.write(steps_seq)
# #                 val_file.write("\n")

# #                 val_info.write(info)
# #                 val_info.write("\n")

# #                 val_label.write(label)
# #                 val_label.write("\n")


    train_file.close()
    train_info.close()
    train_label.close()
    
    val_file.close()
    val_info.close()
    val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    
def prepare_school_coded_finetuning_correctness_after_opts_files(data_processor, options):
    '''
        Ongoing research.
        FinalAnswer step correctness
        Correctness after opts:
            0 if attempt at step>1
            1 if attempt at step==1
    '''
    kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    kcs = [kc for kc in kcs if not pd.isna(kc)]
    kcs = np.array(sorted(list(kcs)))
    print(kcs, type(kcs))
    print(f"KCs: {kcs}")
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
#     val_file = open(options.val_file_path, "w")
#     val_info = open(options.val_info_path, "w")
#     val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    overall_data = []
    overall_labels = []
    # kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    # kcs = [kc if not pd.isna(kc) for kc in kcs]
    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                print(f"{school} : {school_group.shape}")
                school_group = school_group[(school_group['CF (Is StepByStep)'] == False) &
                                            (school_group['CF (Encounter)'] == 0) &
                                            (school_group['CF (Is Review Mode)'] == -1) ]
                print(f"{school} : {school_group.shape}")
                # for class_id, class_group in school_groups.groupby('CF (Anon Class Id)'):
                for student, student_groups in school_group.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)
                    # prob_list = list(pd.unique(student_groups["Problem Name"]))
                    # prob_list= list(student_groups[student_groups["CF (Workspace Progress Status)"]=="GRADUATED"]["Problem Name"])
                    # prob_list = prob_list[-int(len(prob_list)/2):]
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # if not prob in prob_list:
                        #     continue
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups[prob_groups["CF (Is Autofilled)"] == False]["Step Name"]))
                        unique_steps_len = len([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not (s in options.opt_step2)])
                        if unique_steps_len < 4:
                            continue
                        unique_opt_steps_len = len([s for s in unique_steps if not pd.isna(s) and (s in options.opt_step1[1:] or s in options.opt_step2[1:])])
                        if unique_opt_steps_len < 2:
                            continue
                        # print(unique_steps, unique_opt_steps_len)
                        class_id = list(pd.unique(prob_groups["CF (Anon Class Id)"]))
                        step_names_token = []
                        original_steps_actions_attempts_help_levels_outcomes = []
                        original_steps = []
                        means_and_extremes = False
                        opt1_used = False
                        opt2_used = False
                        final_after_opts = False
                        correctness = "0"
                        kcs_skills = [0 for i in kcs]
                        diff_skills = [0 for i in kcs]
                        finalanswer_skill = [0 for i in kcs]
                        for index, row in prob_groups[['Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)',
                                                       'Outcome', 'Help Level', 'CF (Workspace Progress Status)', 
                                                       'CF (Skill Previous p-Known)', 'CF (Skill New p-Known)', 'KC Model(MATHia)', 
                                                       'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                            step = row["Step Name"]
                            action = row["Action"]            # ['Attempt', 'Hint Request', 'Hint Level Change', 'Done']
                            attempt = row["Attempt At Step"]  # number
                            outcome = row["Outcome"]          # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            help_level = row["Help Level"]    # number
                            progress = row["CF (Workspace Progress Status)"]
                            scenario = row['CF (Problem Scenario Tags)']
                            kc = row['KC Model(MATHia)']
                            prev_skill = row['CF (Skill Previous p-Known)']
                            curr_skill = row['CF (Skill New p-Known)']
                            # print(kc, prev_skill)
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    etalon = row["CF (Etalon)"]
                                    if not pd.isna(etalon):
                                        etalon = etalon.strip('{}')
                                        key, value = etalon.split('=')
                                        etalon = value
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                            except Exception as e:
                                                pass
                                if row['CF (Is Autofilled)'] == True:
                                    continue
                                prev = step_names_token[-1] if step_names_token else ""
                                prev_step = step_names_token[-1].split("-")[0] if step_names_token else ""

                                if not step_names_token or step != prev_step:
                                    if step in options.opt_step1 or step in options.opt_step2:
                                        new_step = step
                                        if step in options.opt_step1[1:]:
                                            opt1_used = True
                                        elif step in options.opt_step2[2:]:
                                            opt2_used = True                                            
                                    else:
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                        
                                        if step == "FinalAnswer" and (opt1_used or opt2_used) and not final_after_opts:
                                            final_after_opts = True
                                            if outcome == "OK":
                                                correctness = "1"
                                    step_names_token.append(new_step)
                                    
                                else:
                                    if not (step in options.opt_step1 or step in options.opt_step2 or step == "FinalAnswer"):
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                        if prev < new_step:
                                            step_names_token[-1] = new_step
                                if not pd.isna(kc):
                                    index = np.argwhere(kcs==kc).flatten()[0]
                                    # print(index, type(index))
                                    kcs_skills[index] = prev_skill
                                    diff_skills[index] = prev_skill - curr_skill
                                    if step == "FinalAnswer":
                                        finalanswer_skill[index] = prev_skill

                                original_steps_actions_attempts_help_levels_outcomes.append(f"{step}-{action}-{attempt}-{help_level}-{outcome}")
                                original_steps.append(step)
                        if (not opt1_used) and (not opt2_used):
                            continue
                        unique_steps_len = len([s for s in original_steps if not (s in options.opt_step1) and not(s in options.opt_step2)])
                        if step_names_token and unique_steps_len > 4:
                            label = None
                            if opt1_used and opt2_used:
                                label = "2"
                            if (not opt1_used) and opt2_used:
                                label = "1"
                            if opt1_used and (not opt2_used):
                                label = "0"
                            # print(f"opt1_used: {opt1_ßused}, opt2_used: {opt2_used} label : {label}")
                            info = ",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                           f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                           "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes)), label,
                                            "\t".join(map(str, kcs_skills)), "\t".join(map(str, diff_skills)), 
                                             "\t".join(map(str, finalanswer_skill))])#str(finalanswer_skill)])
                            overall_data.append(["\t".join(step_names_token), info])
                            overall_labels.append(correctness)
#                             proba = random.random()
#                             # if prob in first_prob_list:
#                             if proba <= 0.8:
#                                 train_file.write("\t".join(step_names_token))
#                                 train_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 train_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 train_info.write("\n")

#                             elif proba > 0.9:
#                             # elif prob in last_prob_list:
#                                 test_file.write("\t".join(step_names_token))
#                                 test_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 test_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 test_info.write("\n")

#                             else:
#                                 val_file.write("\t".join(step_names_token))
#                                 val_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 val_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 val_info.write("\n")
                        # break
                    # break
                    # break
                # break
        # break
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    # indices_of_twos = list(np.where(overall_labels == '2')[0])
    
    train_len = int(len(overall_labels) * 0.10)
    sample_size = int(train_len/2)
    print(f"sample_size: {sample_size}")
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))
    # sampled_instances.extend(random.sample(indices_of_twos, sample_size))

    indices_of_zeros = [i for i in indices_of_zeros if not i in sampled_instances ]
    indices_of_ones = [i for i in indices_of_ones if not i in sampled_instances ]
    # indices_of_twos = [i for i in indices_of_twos if not i in sampled_instances ]

    balanced_test = min(len(indices_of_zeros), len(indices_of_ones)) #, len(indices_of_twos))
    print(f"balanced_test: {balanced_test}")
    test_sampled_instances = random.sample(indices_of_zeros, balanced_test)
    test_sampled_instances.extend(random.sample(indices_of_ones, balanced_test))
    # test_sampled_instances.extend(random.sample(indices_of_twos, balanced_test))

    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):

        steps_seq = all_data[0]
        info = all_data[1]

        if index in sampled_instances:
            train_file.write(steps_seq)
            train_file.write("\n")
            
            train_info.write(info)
            train_info.write("\n")
            
            train_label.write(label)
            train_label.write("\n")
        elif index in test_sampled_instances:
            # proba = random.random()
            # if proba <0.5:
            test_file.write(steps_seq)
            test_file.write("\n")

            test_info.write(info)
            test_info.write("\n")

            test_label.write(label)
            test_label.write("\n")
#             else:
#                 val_file.write(steps_seq)
#                 val_file.write("\n")

#                 val_info.write(info)
#                 val_info.write("\n")

#                 val_label.write(label)
#                 val_label.write("\n")


    train_file.close()
    train_info.close()
    train_label.close()
    
    # val_file.close()
    # val_info.close()
    # val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()
    
def prepare_school_coded_finetuning_correctness_after_opts_over_prob_files(data_processor, options):
    '''
        Ongoing research.
        FinalAnswer step correctness
        Correctness after opts:
            0 if attempt at step>1
            1 if attempt at step==1
    '''
    kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    kcs = [kc for kc in kcs if not pd.isna(kc)]
    kcs = np.array(sorted(list(kcs)))
    print(kcs, type(kcs))
    print(f"KCs: {kcs}")
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
#     val_file = open(options.val_file_path, "w")
#     val_info = open(options.val_info_path, "w")
#     val_label = open(options.val_label_path, "w")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    train_data = []
    train_labels = []
    
    test_data = []
    test_labels = []
    # kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    # kcs = [kc if not pd.isna(kc) for kc in kcs]
    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                print(f"{school} : {school_group.shape}")
                school_group = school_group[(school_group['CF (Is StepByStep)'] == False) &
                                            (school_group['CF (Encounter)'] == 0) &
                                            (school_group['CF (Is Review Mode)'] == -1) ]
                print(f"{school} : {school_group.shape}")
                # for class_id, class_group in school_groups.groupby('CF (Anon Class Id)'):
                for student, student_groups in school_group.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)
                    train = True
                    proba = random.random()
                    if proba < 0.5:
                        train = False
                    # prob_list = list(pd.unique(student_groups["Problem Name"]))
                    # prob_list= list(student_groups[student_groups["CF (Workspace Progress Status)"]=="GRADUATED"]["Problem Name"])
                    # prob_list = prob_list[-int(len(prob_list)/2):]
                    prev_kcs_skills = [0 for i in kcs]
                    for pi, (prob, prob_groups) in enumerate(student_groups.groupby("Problem Name")):
                        # if not prob in prob_list:
                        #     continue
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups[prob_groups["CF (Is Autofilled)"] == False]["Step Name"]))
                        unique_steps_len = len([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not (s in options.opt_step2)])
                        if unique_steps_len < 4:
                            continue
                        unique_opt_steps_len = len([s for s in unique_steps if not pd.isna(s) and (s in options.opt_step1[1:] or s in options.opt_step2[1:])])
                        if unique_opt_steps_len < 2:
                            continue
                        # print(unique_steps, unique_opt_steps_len)
                        class_id = list(pd.unique(prob_groups["CF (Anon Class Id)"]))
                        step_names_token = []
                        original_steps_actions_attempts_help_levels_outcomes = []
                        original_steps = []
                        means_and_extremes = False
                        opt1_used = False
                        opt2_used = False
                        final_after_opts = False
                        correctness = "0"
                        kcs_skills = [0 for i in kcs]
                        diff_skills = [0 for i in kcs]
                        finalanswer_skill = [0 for i in kcs]
                        for index, row in prob_groups[['Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)',
                                                       'Outcome', 'Help Level', 'CF (Workspace Progress Status)', 
                                                       'CF (Skill Previous p-Known)', 'CF (Skill New p-Known)', 'KC Model(MATHia)', 
                                                       'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                            step = row["Step Name"]
                            action = row["Action"]            # ['Attempt', 'Hint Request', 'Hint Level Change', 'Done']
                            attempt = row["Attempt At Step"]  # number
                            outcome = row["Outcome"]          # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            help_level = row["Help Level"]    # number
                            progress = row["CF (Workspace Progress Status)"]
                            scenario = row['CF (Problem Scenario Tags)']
                            kc = row['KC Model(MATHia)']
                            prev_skill = row['CF (Skill Previous p-Known)']
                            curr_skill = row['CF (Skill New p-Known)']
                            # print(kc, prev_skill)
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    etalon = row["CF (Etalon)"]
                                    if not pd.isna(etalon):
                                        etalon = etalon.strip('{}')
                                        key, value = etalon.split('=')
                                        etalon = value
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                            except Exception as e:
                                                pass
                                if row['CF (Is Autofilled)'] == True:
                                    continue
                                prev = step_names_token[-1] if step_names_token else ""
                                prev_step = step_names_token[-1].split("-")[0] if step_names_token else ""

                                if not step_names_token or step != prev_step:
                                    if step in options.opt_step1 or step in options.opt_step2:
                                        new_step = step
                                        if step in options.opt_step1[1:]:
                                            opt1_used = True
                                        elif step in options.opt_step2[2:]:
                                            opt2_used = True                                            
                                    else:
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                        
                                        if step == "FinalAnswer" and (opt1_used or opt2_used) and not final_after_opts:
                                            final_after_opts = True
                                            if outcome == "OK":
                                                correctness = "1"
                                    step_names_token.append(new_step)
                                    
                                else:
                                    if not (step in options.opt_step1 or step in options.opt_step2 or step == "FinalAnswer"):
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                        if prev < new_step:
                                            step_names_token[-1] = new_step
                                if not pd.isna(kc):
                                    index = np.argwhere(kcs==kc).flatten()[0]
                                    # print(index, type(index))
                                    kcs_skills[index] = prev_skill
                                    if pi != 0:
                                        diff_skills[index] = prev_skill - prev_kcs_skills[index]
                                    prev_kcs_skills[index] = prev_skill
                                    if step == "FinalAnswer":
                                        finalanswer_skill[index] = prev_skill

                                original_steps_actions_attempts_help_levels_outcomes.append(f"{step}-{action}-{attempt}-{help_level}-{outcome}")
                                original_steps.append(step)
                        if (not opt1_used) and (not opt2_used):
                            continue
                        unique_steps_len = len([s for s in original_steps if not (s in options.opt_step1) and not(s in options.opt_step2)])
                        if step_names_token and unique_steps_len > 4:
                            label = None
                            if opt1_used and opt2_used:
                                label = "2"
                            if (not opt1_used) and opt2_used:
                                label = "1"
                            if opt1_used and (not opt2_used):
                                label = "0"
                            # print(f"opt1_used: {opt1_ßused}, opt2_used: {opt2_used} label : {label}")
                            info = ",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                           f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                           "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes)), label,
                                            "\t".join(map(str, kcs_skills)), "\t".join(map(str, diff_skills)), 
                                             "\t".join(map(str, finalanswer_skill))])#str(finalanswer_skill)])
                            if train:
                                train_data.append(["\t".join(step_names_token), info])
                                train_labels.append(correctness)
                            else:
                                test_data.append(["\t".join(step_names_token), info])
                                test_labels.append(correctness)
#                             proba = random.random()
#                             # if prob in first_prob_list:
#                             if proba <= 0.8:
#                                 train_file.write("\t".join(step_names_token))
#                                 train_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 train_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 train_info.write("\n")

#                             elif proba > 0.9:
#                             # elif prob in last_prob_list:
#                                 test_file.write("\t".join(step_names_token))
#                                 test_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 test_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 test_info.write("\n")

#                             else:
#                                 val_file.write("\t".join(step_names_token))
#                                 val_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 val_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 val_info.write("\n")
                        # break
                    # break
                    # break
                # break
        # break
#     overall_labels = np.array(overall_labels)
#     indices_of_zeros = list(np.where(overall_labels == '0')[0])
#     indices_of_ones = list(np.where(overall_labels == '1')[0])
#     # indices_of_twos = list(np.where(overall_labels == '2')[0])
    
#     train_len = int(len(overall_labels) * 0.10)
#     sample_size = int(train_len/2)
#     print(f"sample_size: {sample_size}")
#     sampled_instances = random.sample(indices_of_zeros, sample_size)
#     sampled_instances.extend(random.sample(indices_of_ones, sample_size))
#     # sampled_instances.extend(random.sample(indices_of_twos, sample_size))

#     indices_of_zeros = [i for i in indices_of_zeros if not i in sampled_instances ]
#     indices_of_ones = [i for i in indices_of_ones if not i in sampled_instances ]
#     # indices_of_twos = [i for i in indices_of_twos if not i in sampled_instances ]

#     balanced_test = min(len(indices_of_zeros), len(indices_of_ones)) #, len(indices_of_twos))
#     print(f"balanced_test: {balanced_test}")
#     test_sampled_instances = random.sample(indices_of_zeros, balanced_test)
#     test_sampled_instances.extend(random.sample(indices_of_ones, balanced_test))
#     # test_sampled_instances.extend(random.sample(indices_of_twos, balanced_test))

    for index, (all_data, label) in enumerate(zip(train_data, train_labels)):
        steps_seq = all_data[0]
        info = all_data[1]

        train_file.write(steps_seq)
        train_file.write("\n")

        train_info.write(info)
        train_info.write("\n")

        train_label.write(label)
        train_label.write("\n")
    train_file.close()
    train_info.close()
    train_label.close()
    
    for index, (all_data, label) in enumerate(zip(test_data, test_labels)):
        steps_seq = all_data[0]
        info = all_data[1]

        test_file.write(steps_seq)
        test_file.write("\n")

        test_info.write(info)
        test_info.write("\n")

        test_label.write(label)
        test_label.write("\n")
    test_file.close()
    test_info.close()
    test_label.close()
    
def prepare_school_coded_finetuning_correctness_after_opts_per_files(data_processor, options):
    '''
        Ongoing research.
        FinalAnswer step correctness
        Correctness after opts:
            0 if attempt at step>1
            1 if attempt at step==1
    '''
    kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    kcs = [kc for kc in kcs if not pd.isna(kc)]
    kcs = np.array(sorted(list(kcs)))
    print(kcs, type(kcs))
    print(f"KCs: {kcs}")
    chunk_iterator = data_processor.load_file_iterator(sep=",")

    train_file = open(options.train_file_path, "w")
    train_info = open(options.train_info_path, "w")
    train_label = open(options.train_label_path, "w")
    
    val_file = open(options.val_file_path, "a")
    val_info = open(options.val_info_path, "a")
    val_label = open(options.val_label_path, "a")
    
    test_file = open(options.test_file_path, "w")
    test_info = open(options.test_info_path, "w")
    test_label = open(options.test_label_path, "w")
    
    overall_data = []
    overall_labels = []
    # kcs = pickle.load(open("dataset/CL_2223/ratio_proportion_change3_2223/unique_kcs_list.pkl", "rb"))
    # kcs = [kc if not pd.isna(kc) for kc in kcs]
    for chunk_data in chunk_iterator:
        for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
            if not options.school or school in options.school:
                print(f"{school} : {school_group.shape}")
                school_group = school_group[(school_group['CF (Is StepByStep)'] == False) &
                                            (school_group['CF (Encounter)'] == 0) &
                                            (school_group['CF (Is Review Mode)'] == -1) ]
                print(f"{school} : {school_group.shape}")
                # for class_id, class_group in school_groups.groupby('CF (Anon Class Id)'):
                for student, student_groups in school_group.groupby("Anon Student Id"):
                    student_groups.sort_values(by="Time", inplace=True)
                    # prob_list = list(pd.unique(student_groups["Problem Name"]))
                    # prob_list= list(student_groups[student_groups["CF (Workspace Progress Status)"]=="GRADUATED"]["Problem Name"])
                    # prob_list = prob_list[-int(len(prob_list)/2):]
                    for prob, prob_groups in student_groups.groupby("Problem Name"):
                        # if not prob in prob_list:
                        #     continue
                        actions = list(prob_groups["Action"])
                        # A problem should be completed by a student clicking Done button.
                        if not "Done" in actions: 
                            continue
                        unique_steps = list(pd.unique(prob_groups[prob_groups["CF (Is Autofilled)"] == False]["Step Name"]))
                        unique_steps_len = len([s for s in unique_steps if not pd.isna(s) and not (s in options.opt_step1) and not (s in options.opt_step2)])
                        if unique_steps_len < 4:
                            continue
                        unique_opt_steps_len = len([s for s in unique_steps if not pd.isna(s) and (s in options.opt_step1[1:] or s in options.opt_step2[1:])])
                        if unique_opt_steps_len < 2:
                            continue
                        # print(unique_steps, unique_opt_steps_len)
                        class_id = list(pd.unique(prob_groups["CF (Anon Class Id)"]))
                        step_names_token = []
                        original_steps_actions_attempts_help_levels_outcomes = []
                        original_steps = []
                        means_and_extremes = False
                        opt1_used = False
                        opt2_used = False
                        final_after_opts = False
                        correctness = "0"
                        kcs_skills = [0 for i in kcs]
                        diff_skills = [0 for i in kcs]
                        finalanswer_skill = [0 for i in kcs]
                        for index, row in prob_groups[['Step Name', 'Action', 'Attempt At Step', 'CF (Is Autofilled)',
                                                       'Outcome', 'Help Level', 'CF (Workspace Progress Status)', 
                                                       'CF (Skill Previous p-Known)', 'CF (Skill New p-Known)', 'KC Model(MATHia)', 
                                                       'CF (Etalon)', 'CF (Problem Scenario Tags)']].iterrows():
                            step = row["Step Name"]
                            action = row["Action"]            # ['Attempt', 'Hint Request', 'Hint Level Change', 'Done']
                            attempt = row["Attempt At Step"]  # number
                            outcome = row["Outcome"]          # ['OK', 'BUG', 'ERROR', 'INITIAL_HINT', 'HINT_LEVEL_CHANGE']
                            help_level = row["Help Level"]    # number
                            progress = row["CF (Workspace Progress Status)"]
                            scenario = row['CF (Problem Scenario Tags)']
                            kc = row['KC Model(MATHia)']
                            prev_skill = row['CF (Skill Previous p-Known)']
                            curr_skill = row['CF (Skill New p-Known)']
                            # print(kc, prev_skill)
                            if not pd.isna(step):
                                if step in options.opt_step1 and not means_and_extremes:
                                    etalon = row["CF (Etalon)"]
                                    if not pd.isna(etalon):
                                        etalon = etalon.strip('{}')
                                        key, value = etalon.split('=')
                                        etalon = value
                                        try:
                                            etalon = int(etalon)
                                        except Exception as e:
                                            try:
                                                etalon = float(etalon)
                                                means_and_extremes = True
                                            except Exception as e:
                                                pass
                                if row['CF (Is Autofilled)'] == True:
                                    continue
                                prev = step_names_token[-1] if step_names_token else ""
                                prev_step = step_names_token[-1].split("-")[0] if step_names_token else ""

                                if not step_names_token or step != prev_step:
                                    if step in options.opt_step1 or step in options.opt_step2:
                                        new_step = step
                                        if step in options.opt_step1[1:]:
                                            opt1_used = True
                                        elif step in options.opt_step2[2:]:
                                            opt2_used = True                                            
                                    else:
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                        
                                        if step == "FinalAnswer" and (opt1_used or opt2_used) and not final_after_opts:
                                            final_after_opts = True
                                            if outcome == "OK":
                                                correctness = "1"
                                    step_names_token.append(new_step)
                                    
                                else:
                                    if not (step in options.opt_step1 or step in options.opt_step2 or step == "FinalAnswer"):
                                        if action == "Attempt" and outcome != "OK":
                                            new_step = step+"-2"
                                        elif "Hint" in action:
                                            new_step = step+"-1"
                                        else:
                                            new_step = step+"-0"
                                            
                                        if prev < new_step:
                                            step_names_token[-1] = new_step
                                if not pd.isna(kc):
                                    index = np.argwhere(kcs==kc).flatten()[0]
                                    # print(index, type(index))
                                    kcs_skills[index] = prev_skill
                                    diff_skills[index] = prev_skill - curr_skill
                                    if step == "FinalAnswer":
                                        finalanswer_skill[index] = prev_skill

                                original_steps_actions_attempts_help_levels_outcomes.append(f"{step}-{action}-{attempt}-{help_level}-{outcome}")
                                original_steps.append(step)
                        if (not opt1_used) and (not opt2_used):
                            continue
                        unique_steps_len = len([s for s in original_steps if not (s in options.opt_step1) and not(s in options.opt_step2)])
                        if step_names_token and unique_steps_len > 4:
                            label = None
                            if opt1_used and opt2_used:
                                label = "2"
                            if (not opt1_used) and opt2_used:
                                label = "1"
                            if opt1_used and (not opt2_used):
                                label = "0"
                            # print(f"opt1_used: {opt1_ßused}, opt2_used: {opt2_used} label : {label}")
                            info = ",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
                                           f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
                                           "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes)), label,
                                            "\t".join(map(str, kcs_skills)), "\t".join(map(str, diff_skills)), 
                                             "\t".join(map(str, finalanswer_skill))])#str(finalanswer_skill)])
                            overall_data.append(["\t".join(step_names_token), info])
                            overall_labels.append(correctness)
#                             proba = random.random()
#                             # if prob in first_prob_list:
#                             if proba <= 0.8:
#                                 train_file.write("\t".join(step_names_token))
#                                 train_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 train_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 train_info.write("\n")

#                             elif proba > 0.9:
#                             # elif prob in last_prob_list:
#                                 test_file.write("\t".join(step_names_token))
#                                 test_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 test_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 test_info.write("\n")

#                             else:
#                                 val_file.write("\t".join(step_names_token))
#                                 val_file.write("\n")
#                                 # school, class, student id, progress, problem name, scenario, 
#                                 # prefered ER or ME, total steps length, 
#                                 # original seq-action-attempt-help_level-outcome
#                                 val_info.write(",".join([str(school), "\t".join(class_id), str(student), str(progress), str(prob), str(scenario), 
#                                                f"{1 if means_and_extremes else 0}", str(len(step_names_token)), 
#                                                "\t".join(map(str, original_steps_actions_attempts_help_levels_outcomes))]))
#                                 val_info.write("\n")
                        # break
                    # break
                    # break
                # break
        # break
    overall_labels = np.array(overall_labels)
    indices_of_zeros = list(np.where(overall_labels == '0')[0])
    indices_of_ones = list(np.where(overall_labels == '1')[0])
    # indices_of_twos = list(np.where(overall_labels == '2')[0])
    
    # train_len = int(len(overall_labels) * 0.10)
    train_len = int(len(overall_labels) * float(options.per))
    
    sample_size = int(train_len/2)
    if float(options.per) == 1:
        sample_size = min(len(indices_of_zeros), len(indices_of_ones))
    elif float(options.per) > 1:
        sample_size = int(options.per)
    print(f"sample_size: {sample_size}")
    sampled_instances = random.sample(indices_of_zeros, sample_size)
    sampled_instances.extend(random.sample(indices_of_ones, sample_size))
    # sampled_instances.extend(random.sample(indices_of_twos, sample_size))

    indices_of_zeros = [i for i in indices_of_zeros if not i in sampled_instances ]
    indices_of_ones = [i for i in indices_of_ones if not i in sampled_instances ]
    # indices_of_twos = [i for i in indices_of_twos if not i in sampled_instances ]

    balanced_test = min(len(indices_of_zeros), len(indices_of_ones)) #, len(indices_of_twos))
    print(f"balanced_test: {balanced_test}")
    test_sampled_instances = random.sample(indices_of_zeros, balanced_test)
    test_sampled_instances.extend(random.sample(indices_of_ones, balanced_test))
    # test_sampled_instances.extend(random.sample(indices_of_twos, balanced_test))

    for index, (all_data, label) in enumerate(zip(overall_data, overall_labels)):

        steps_seq = all_data[0]
        info = all_data[1]

        if index in sampled_instances:
            train_file.write(steps_seq)
            train_file.write("\n")
            
            train_info.write(info)
            train_info.write("\n")
            
            train_label.write(label)
            train_label.write("\n")
            if float(options.per) == 1.0:
                val_file.write(steps_seq)
                val_file.write("\n")

                val_info.write(info)
                val_info.write("\n")

                val_label.write(label)
                val_label.write("\n")
        
        elif index in test_sampled_instances:
            # proba = random.random()
            # if proba <0.5:
            test_file.write(steps_seq)
            test_file.write("\n")

            test_info.write(info)
            test_info.write("\n")

            test_label.write(label)
            test_label.write("\n")
            
            if float(options.per) != 1.0:
                val_file.write(steps_seq)
                val_file.write("\n")

                val_info.write(info)
                val_info.write("\n")

                val_label.write(label)
                val_label.write("\n")


    train_file.close()
    train_info.close()
    train_label.close()
    
    val_file.close()
    val_info.close()
    val_label.close()
    
    test_file.close()
    test_info.close()
    test_label.close()

    
    
def prepare_pretraining_vocab_file(options):
    
    # kc = pickle.load(open("dataset/unique/unique_kcs_list.pkl","rb"))
    # kc_token = {"KC"+str(i):k for i, k in enumerate(kc)}
    # pickle.dump(kc_token, open("pretraining/unique_dict_kc_token.pkl", "wb"))
    
    # steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))
    # step_token = {"step"+str(i):k for i, k in enumerate(steps)}
    # folder_name = options.workspace_name+"/" if options.workspace_name else ""
    # pickle.dump(step_token, open(f"{folder_name}pretraining/unique_dict_step_token.pkl", "wb"))

    # steps = pickle.load(open(f"{options.dataset_folder}unique_new_steps_w_action_attempt_list.pkl","rb"))
    steps = pickle.load(open(f"{options.dataset_folder}unique_steps_list.pkl","rb"))

    # print("No of unique kc", len(kc))
    print("No of unique steps ", len(steps))
    # print("No of unique problem", len(prob))
    # print("Size of vocab ", len(steps))

    ordered_steps = sorted(list(steps))

    with (open(options.vocab_file_path,"w")) as vb_file:
        vb_file.write("[PAD]\n")
        vb_file.write("[UNK]\n")
        vb_file.write("[MASK]\n")
        vb_file.write("[CLS]\n")
        vb_file.write("[SEP]\n")
        # vb_file.write("\n".join(kc_token.keys()))
        # vb_file.write("\n")
        # vb_file.write("\n".join(step_token.keys()))
        # vb_file.write("\n".join(ordered_steps))
        for step in ordered_steps:
            if step in options.opt_step1 or step in options.opt_step2:
                vb_file.write(f"{step}\n")
            else:
                for i in range(3):
                    vb_file.write(f"{step}-{i}\n")
        vb_file.close()
    with open(options.vocab_file_path,"r") as f:
        l = f.readlines()
        print(l, len(l))
        f.close()


def main(opt):
    options = copy.deepcopy(opt)
    if opt.workspace_name:
        options.dataset_folder = opt.dataset_folder+opt.workspace_name+"/"
        
    data_processor = DataPreprocessor(input_file_path=opt.dataset)
    
    if opt.analyze_dataset_by_section:
        print(f"Analyzing dataset by section for workspace: {opt.workspace_name}")
        data_processor.analyze_dataset_by_section(opt.workspace_name)
        
        pickle.dump(data_processor.unique_students, open(f"{options.dataset_folder}unique_students_list.pkl", "wb"))
        pickle.dump(data_processor.unique_problems, open(f"{options.dataset_folder}unique_problems_list.pkl", "wb"))
        pickle.dump(data_processor.unique_prob_hierarchy, open(f"{options.dataset_folder}unique_hierarchy_list.pkl", "wb"))
        pickle.dump(data_processor.unique_kcs, open(f"{options.dataset_folder}unique_kcs_list.pkl", "wb"))
        pickle.dump(data_processor.unique_steps, open(f"{options.dataset_folder}unique_steps_list.pkl", "wb"))
        
    if opt.analyze_dataset_by_school:
        print(f"Analyzing dataset of all school for workspace: {opt.workspace_name}")
        data_processor.analyze_dataset_by_school(opt.workspace_name)
        
        if not os.path.exists(options.dataset_folder):
            os.makedirs(options.dataset_folder)
        pickle.dump(data_processor.unique_schools, open(f"{options.dataset_folder}unique_schools_list.pkl", "wb"))
        pickle.dump(data_processor.unique_class, open(f"{options.dataset_folder}unique_class_list.pkl", "wb"))
        pickle.dump(data_processor.unique_students, open(f"{options.dataset_folder}unique_students_list.pkl", "wb"))
        pickle.dump(data_processor.unique_problems, open(f"{options.dataset_folder}unique_problems_list.pkl", "wb"))
        pickle.dump(data_processor.unique_kcs, open(f"{options.dataset_folder}unique_kcs_list.pkl", "wb"))
        pickle.dump(data_processor.unique_steps, open(f"{options.dataset_folder}unique_steps_list.pkl", "wb"))
        pickle.dump(data_processor.unique_new_steps_w_action_attempt, open(f"{options.dataset_folder}unique_new_steps_w_action_attempt_list.pkl", "wb"))
        pickle.dump(data_processor.unique_new_steps_w_action_attempt_kcs, open(f"{options.dataset_folder}unique_new_steps_w_action_attempt_kcs.pkl", "wb"))
        pickle.dump(data_processor.unique_new_steps_w_kcs, open(f"{options.dataset_folder}unique_new_steps_w_kcs_list.pkl", "wb"))

    if opt.workspace_name:
        for k,v in vars(opt).items():
            if 'path' in k:
                if v:
                    redirect_path = opt.workspace_name+"/"
                    if opt.school and opt.pretrain:
                        sch = f"sch_largest_{len(opt.school)}-coded" #f"sch_largest_655"
                        redirect_path = redirect_path + sch+"/"
                    if opt.school_folder:
                        redirect_path = redirect_path + opt.school_folder+"/"
                    # else:
                    #     sch = "sch_largest_655"                        
                    if k != "vocab_file_path":
                        if opt.pretrain:
                            redirect_path = redirect_path + "pretraining/"
                        else:
                            if opt.code:
                                redirect_path = redirect_path + f"{opt.code}/"
                            elif opt.finetune_task:
                                if opt.diff_val_folder and "val" in v:
                                    redirect_path = redirect_path + f"finetuning/"
                                else:
                                    redirect_path = redirect_path + f"finetuning/{opt.finetune_task}/"
                        if not os.path.exists(redirect_path):
                            os.makedirs(redirect_path)
                    else:
                        if not os.path.exists(redirect_path+"/pretraining/"):
                            os.makedirs(redirect_path+"/pretraining/")
                    setattr(options, f"{k}", redirect_path+v)
                    # setattr(options, f"{k}", opt.workspace_name+"/check/"+v)
                    print(f"options.{k} : {getattr(options, f'{k}')}")


    
    if options.pretrain:
        print("Preparing vocab...")
        prepare_pretraining_vocab_file(options)
        print("Preparing pre-training dataset...")
        # old non-repeated steps
        # prepare_pretraining_files(data_processor, options)
        # coded
        # prepare_school_coded_pretraining_files(data_processor, options)
        prepare_school_coded_finetuning_opts_intentional_files(data_processor, options)
        # prepare_pretraining_files(data_processor, options)
        # prepare_school_pretraining_files(data_processor, options)
    # else:
    #     print("Preparing attention dataset...")
    #     prepare_school_attention_files(data_processor, options)
    else:
        print("Preparing fine-tuning dataset...")
        # _1920
        # prepare_finetuning_10per_files(data_processor, options)
        # prepare_finetuning_IS_FS_files(data_processor, options)
        # prepare_finetuning_correctness_files(data_processor, options)

        # _2223
        # prepare_school_coded_finetuning_partial_seq_files(data_processor, options)
        # prepare_school_coded_finetuning_opts_files(data_processor, options)
        prepare_school_coded_finetuning_correctness_after_opts_per_files(data_processor, options)
        # prepare_school_coded_finetuning_correctness_after_opts_files(data_processor, options)
        # prepare_school_coded_finetuning_correctness_after_opts_over_prob_files(data_processor, options)
        # prepare_finetuning_IS_files(data_processor, options)
    #     # prepare_finetuning_FS_files(data_processor, options)
        # prepare_finetuning_correctness_aaai_files(data_processor, options)
    #     # prepare_finetuning_SL_files(data_processor, options)
    #     # prepare_finetuning_effectiveness_files(data_processor, options)
    #     prepare_attn_test_files(data_processor, options)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_folder', type=str, default="dataset/CL4999_1920/")

    parser.add_argument('-analyze_dataset_by_section', type=bool, default=False)
    parser.add_argument('-analyze_dataset_by_school', type=bool, default=False)
    parser.add_argument('-workspace_name', type=str, default=None)
    parser.add_argument('-school', nargs='+', type=str, default=None)
    parser.add_argument('-school_folder', type=str, default=None)
    
    # parser.add_argument('-highGRschool', nargs='+', type=str, default=None)
    # parser.add_argument('-lowGRschool', nargs='+', type=str, default=None)

    parser.add_argument('-code', type=str, default=None)
    parser.add_argument('-finetune_task', type=str, default=None)

    parser.add_argument('-per', type=float, default=None)
    parser.add_argument("-diff_val_folder", type=bool, default=False, help="use for different val folder")

    parser.add_argument('-opt_step1', nargs='+', type=str, help='List of optional steps 1')
    parser.add_argument('-opt_step2', nargs='+', type=str, help='List of optional steps 2')
    parser.add_argument('-final_step', nargs='+', type=str, help='List of final step')
    
    parser.add_argument('-dataset', type=str, default="dataset/CL4999_1920/course2_1920_4999_students_datashop.txt")
    
    parser.add_argument('-pretrain', type=bool, default=False)
    parser.add_argument('-vocab_file_path', type=str, default="pretraining/vocab.txt") #pretraining/vocab.txt

    # Prepare for pretraining
    parser.add_argument('-train_file_path', type=str, default="train.txt") #pretraining/pretrain.txt
    parser.add_argument('-train_info_path', type=str, default="train_info.txt") #pretraining/pretrain_info.txt
    parser.add_argument('-train_label_path', type=str, default="train_label.txt") #finetuning/train_label.txt

    parser.add_argument('-val_file_path', type=str, default="val.txt") #pretraining/val.txt
    parser.add_argument('-val_info_path', type=str, default="val_info.txt") #pretraining/val_info.txt
    parser.add_argument('-val_label_path', type=str, default="val_label.txt") #finetuning/val_label.txt

    parser.add_argument('-test_file_path', type=str, default="test.txt") #pretraining/test.txt
    parser.add_argument('-test_info_path', type=str, default="test_info.txt") #pretraining/test_info.txt
    parser.add_argument('-test_label_path', type=str, default="test_label.txt") #finetuning/test_label.txt


#     parser.add_argument('-train_gt_label_path', type=str, default="finetuning/train_gt_label.txt")
#     parser.add_argument('-test_gt_label_path', type=str, default="finetuning/test_gt_label.txt")


    options = parser.parse_args()
    if not options.opt_step1:
        setattr(options, "opt_step1", [])
    print("Optional steps 1: ", options.opt_step1)
    
    if not options.opt_step2:
        setattr(options, "opt_step2", [])
    print("Optional steps 2: ", options.opt_step2)
    
    if not options.final_step:
        setattr(options, "final_step", [])
    print("Final steps: ", options.final_step)
    
    main(options)
    
    
    