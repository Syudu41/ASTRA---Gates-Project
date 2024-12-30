import time
import pandas as pd

import sys

class DataPreprocessor:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.unique_students = None
        self.unique_problems = None
        self.unique_prob_hierarchy = None
        self.unique_steps = None
        self.unique_kcs = None

    def analyze_dataset(self):
        file_iterator = self.load_file_iterator()

        start_time = time.time()
        self.unique_students = {"st"}
        self.unique_problems = {"pr"}
        self.unique_prob_hierarchy = {"ph"}
        self.unique_kcs = {"kc"}
        for chunk_data in file_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                self.unique_students.update({student_id})
                prob_hierarchy = std_groups.groupby('Level (Workspace Id)')
                for hierarchy, hierarchy_groups in prob_hierarchy:
                    self.unique_prob_hierarchy.update({hierarchy})
                    prob_name = hierarchy_groups.groupby('Problem Name')
                    for problem_name, prob_name_groups in prob_name:
                        self.unique_problems.update({problem_name})
                        sub_skills = prob_name_groups['KC Model(MATHia)']
                        for a in sub_skills:
                            if str(a) != "nan":
                                temp = a.split("~~")
                                for kc in temp:
                                    self.unique_kcs.update({kc})
        self.unique_students.remove("st")
        self.unique_problems.remove("pr")
        self.unique_prob_hierarchy.remove("ph")
        self.unique_kcs.remove("kc")
        end_time = time.time()
        print("Time Taken to analyze dataset = ", end_time - start_time)
        print("Length of unique students->", len(self.unique_students))
        print("Length of unique problems->", len(self.unique_problems))
        print("Length of unique problem hierarchy->", len(self.unique_prob_hierarchy))
        print("Length of Unique Knowledge components ->", len(self.unique_kcs))

    def analyze_dataset_by_section(self, workspace_name):
        file_iterator = self.load_file_iterator()
        
        start_time = time.time()
        self.unique_students = {"st"}
        self.unique_problems = {"pr"}
        self.unique_prob_hierarchy = {"ph"}
        self.unique_steps = {"s"}
        self.unique_kcs = {"kc"}
        # with open("workspace_info.txt", 'a') as f:
        #     sys.stdout = f
        for chunk_data in file_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                prob_hierarchy = std_groups.groupby('Level (Workspace Id)')
                for hierarchy, hierarchy_groups in prob_hierarchy:
                    if workspace_name == hierarchy:
                        # print("Workspace : ", hierarchy)
                        self.unique_students.update({student_id})   
                        self.unique_prob_hierarchy.update({hierarchy})
                        prob_name = hierarchy_groups.groupby('Problem Name')
                        for problem_name, prob_name_groups in prob_name:
                            self.unique_problems.update({problem_name})
                            step_names = prob_name_groups['Step Name']
                            sub_skills = prob_name_groups['KC Model(MATHia)']
                            for step in step_names:
                                if str(step) != "nan":
                                    self.unique_steps.update({step})
                            for a in sub_skills:
                                if str(a) != "nan":
                                    temp = a.split("~~")
                                    for kc in temp:
                                        self.unique_kcs.update({kc})
        self.unique_problems.remove("pr")
        self.unique_prob_hierarchy.remove("ph")
        self.unique_steps.remove("s")
        self.unique_kcs.remove("kc")
        end_time = time.time()
        print("Time Taken to analyze dataset = ", end_time - start_time)
        print("Workspace-> ",workspace_name)
        print("Length of unique students->", len(self.unique_students))
        print("Length of unique problems->", len(self.unique_problems))
        print("Length of unique problem hierarchy->", len(self.unique_prob_hierarchy))
        print("Length of unique step names ->", len(self.unique_steps))
        print("Length of unique knowledge components ->", len(self.unique_kcs))
        #     f.close()
        # sys.stdout = sys.__stdout__
        
    def analyze_dataset_by_school(self, workspace_name, school_id=None):
        file_iterator = self.load_file_iterator(sep=",")

        start_time = time.time()
        self.unique_schools = set()
        self.unique_class = set()
        self.unique_students = set()
        self.unique_problems = set()
        self.unique_steps = set()
        self.unique_kcs = set()
        self.unique_actions = set()
        self.unique_outcomes = set()
        self.unique_new_steps_w_action_attempt = set()
        self.unique_new_steps_w_kcs = set()
        self.unique_new_steps_w_action_attempt_kcs = set()
        
        for chunk_data in file_iterator:
            for school, school_group in chunk_data.groupby('CF (Anon School Id)'):
                # if school and school == school_id:
                self.unique_schools.add(school)
                for class_id, class_group in school_group.groupby('CF (Anon Class Id)'):
                    self.unique_class.add(class_id)
                    for student_id, std_group in class_group.groupby('Anon Student Id'):
                        self.unique_students.add(student_id)
                        for prob, prob_group in std_group.groupby('Problem Name'):
                            self.unique_problems.add(prob)
                            
                            step_names = set(prob_group['Step Name'])
                            sub_skills = set(prob_group['KC Model(MATHia)'])
                            actions = set(prob_group['Action'])
                            outcomes = set(prob_group['Outcome'])
                            
                            self.unique_steps.update(step_names)
                            self.unique_kcs.update(sub_skills)
                            self.unique_actions.update(actions)
                            self.unique_outcomes.update(outcomes)
                            
                            for step in step_names:                                
                                if pd.isna(step):
                                    step_group = prob_group[pd.isna(prob_group['Step Name'])]
                                else:
                                    step_group = prob_group[prob_group['Step Name']==step]
                                    
                                for kc in set(step_group['KC Model(MATHia)']):
                                    new_step = f"{step}:{kc}"
                                    self.unique_new_steps_w_kcs.add(new_step)

                                for action, action_group in step_group.groupby('Action'):
                                    for attempt, attempt_group in action_group.groupby('Attempt At Step'):
                                        new_step = f"{step}:{action}:{attempt}"
                                        self.unique_new_steps_w_action_attempt.add(new_step)

                                        for kc in set(attempt_group["KC Model(MATHia)"]):
                                            new_step = f"{step}:{action}:{attempt}:{kc}"
                                            self.unique_new_steps_w_action_attempt_kcs.add(new_step)
                                        

        end_time = time.time()
        print("Time Taken to analyze dataset = ", end_time - start_time)
        print("Workspace-> ",workspace_name)
        print("Length of unique students->", len(self.unique_students))
        print("Length of unique problems->", len(self.unique_problems))
        print("Length of unique classes->", len(self.unique_class))
        print("Length of unique step names ->", len(self.unique_steps))
        print("Length of unique knowledge components ->", len(self.unique_kcs))
        print("Length of unique actions ->", len(self.unique_actions))
        print("Length of unique outcomes ->", len(self.unique_outcomes))
        print("Length of unique new step names with actions and attempts ->", len(self.unique_new_steps_w_action_attempt))
        print("Length of unique new step names with actions, attempts and kcs ->", len(self.unique_new_steps_w_action_attempt_kcs))
        print("Length of unique new step names with kcs ->", len(self.unique_new_steps_w_kcs))

    def load_file_iterator(self, sep="\t"):
        chunk_iterator = pd.read_csv(self.input_file_path, sep=sep, header=0, iterator=True, chunksize=1000000)
        return chunk_iterator

