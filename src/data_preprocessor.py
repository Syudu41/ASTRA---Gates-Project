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

    def load_file_iterator(self):
        chunk_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)
        return chunk_iterator

