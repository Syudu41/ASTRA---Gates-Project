import gradio as gr
from huggingface_hub import hf_hub_download
import pickle
from gradio import Progress
import numpy as np
import subprocess
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
from matplotlib.figure import Figure
# Define the function to process the input file and model selection

def process_file(model_name,inc_slider,progress=Progress(track_tqdm=True)):
    # progress = gr.Progress(track_tqdm=True)

    progress(0, desc="Starting the processing") 
    # with open(file.name, 'r') as f:
    #     content = f.read()
    # saved_test_dataset = "train.txt"
    # saved_test_label = "train_label.txt"
    # saved_train_info="train_info.txt"
    # Save the uploaded file content to a specified location
    # shutil.copyfile(file.name, saved_test_dataset)
    # shutil.copyfile(label.name, saved_test_label)
    # shutil.copyfile(info.name, saved_train_info)
    parent_location="ratio_proportion_change3_2223/sch_largest_100-coded/finetuning/"
    test_info_location=parent_location+"fullTest/test_info.txt"
    test_location=parent_location+"fullTest/test.txt"
    if(model_name=="ASTRA-FT-HGR"):
        finetune_task="highGRschool10"
        # test_info_location=parent_location+"fullTest/test_info.txt"
        # test_location=parent_location+"fullTest/test.txt"
    elif(model_name== "ASTRA-FT-LGR" ):
        finetune_task="lowGRschoolAll"
        # test_info_location=parent_location+"lowGRschoolAll/test_info.txt"
        # test_location=parent_location+"lowGRschoolAll/test.txt"
    elif(model_name=="ASTRA-FT-FULL"):
        # test_info_location=parent_location+"fullTest/test_info.txt"
        # test_location=parent_location+"fullTest/test.txt"
        finetune_task="fullTest"
    else:
        finetune_task=None
    # Load the test_info file and the graduation rate file
    test_info = pd.read_csv(test_info_location, sep=',', header=None, engine='python')
    grad_rate_data = pd.DataFrame(pd.read_pickle('assests/school_grduation_rate.pkl'),columns=['school_number','grad_rate'])  # Load the grad_rate data

    # Step 1: Extract unique school numbers from test_info
    unique_schools = test_info[0].unique()

    # Step 2: Filter the grad_rate_data using the unique school numbers
    schools = grad_rate_data[grad_rate_data['school_number'].isin(unique_schools)]

    # Define a threshold for high and low graduation rates (adjust as needed)
    grad_rate_threshold = 0.9  

    # Step 4: Divide schools into high and low graduation rate groups
    high_grad_schools = schools[schools['grad_rate'] >= grad_rate_threshold]['school_number'].unique()
    low_grad_schools = schools[schools['grad_rate'] < grad_rate_threshold]['school_number'].unique()

    # Step 5: Sample percentage of schools from each group
    high_sample = pd.Series(high_grad_schools).sample(frac=inc_slider/100, random_state=1).tolist()
    low_sample = pd.Series(low_grad_schools).sample(frac=inc_slider/100, random_state=1).tolist()

    # Step 6: Combine the sampled schools
    random_schools = high_sample + low_sample

    # Step 7: Get indices for the sampled schools
    indices = test_info[test_info[0].isin(random_schools)].index.tolist()
    high_indices = test_info[(test_info[0].isin(high_sample))].index.tolist()
    low_indices = test_info[(test_info[0].isin(low_sample))].index.tolist()
    
    # Load the test file and select rows based on indices
    test = pd.read_csv(test_location, sep=',', header=None, engine='python')
    selected_rows_df2 = test.loc[indices]

    # Save the selected rows to a file
    selected_rows_df2.to_csv('fileHandler/selected_rows.txt', sep='\t', index=False, header=False, quoting=3, escapechar=' ')
    # ✅ Get the first 20% and last 20% of instances for each student ID within selected schools

    selected_test_info = test_info.loc[indices]

    # First 20%
    first_20_percent_indices = selected_test_info.groupby(3).apply(
        lambda x: x.head(int(len(x) * 0.2))
    ).index.get_level_values(1).tolist()

    # Last 20%
    last_20_percent_indices = selected_test_info.groupby(3).apply(
        lambda x: x.tail(int(len(x) * 0.2))
    ).index.get_level_values(1).tolist()

    # Select the corresponding rows from the test file
    first_20_percent_rows = test.loc[first_20_percent_indices]
    last_20_percent_rows = test.loc[last_20_percent_indices]

    # Save the first 20% instances per student to a file
    first_20_percent_rows.to_csv('fileHandler/selected_rows_first20.txt', sep='\t', index=False, header=False, quoting=3, escapechar=' ')

    # Save the last 20% instances per student to a file
    last_20_percent_rows.to_csv('fileHandler/selected_rows_last20.txt', sep='\t', index=False, header=False, quoting=3, escapechar=' ')

    # select the graduation groups
    graduation_groups = [
    'high' if idx in high_indices else 'low' for idx in selected_rows_df2.index
    ]
    # Group data by opt_task1 and opt_task2 based on test_info[6]
    opt_task_groups = ['opt_task1' if test_info.loc[idx, 6] == 0 else 'opt_task2' for idx in selected_rows_df2.index]
    progress(0.2, desc="Files create and saved!! Now Executing models")
    print("finetuned task: ",finetune_task)
    subprocess.run([
        "python", "new_test_saved_finetuned_model.py",
        "-workspace_name", "ratio_proportion_change3_2223/sch_largest_100-coded",
        "-finetune_task", finetune_task,
        "-test_dataset_path","../../../../fileHandler/selected_rows.txt",
        # "-test_label_path","../../../../train_label.txt",
        "-finetuned_bert_classifier_checkpoint", 
        "ratio_proportion_change3_2223/sch_largest_100-coded/output/highGRschool10/bert_fine_tuned.model.ep42",
        "-e",str(1),
        "-b",str(1000)
    ])
    progress(0.5,desc="Model execution completed!! Now performing analysis on the results")
    with open("fileHandler/roc_data2.pkl", 'rb') as file:
        data = pickle.load(file)
    t_label=data[0]
    p_label=data[1]
    # Step 1: Align graduation_group, t_label, and p_label
    aligned_labels = list(zip(graduation_groups, t_label, p_label))
    opt_task_aligned = list(zip(opt_task_groups, t_label, p_label))
    # Step 2: Separate the labels for high and low groups
    high_t_labels = [t for grad, t, p in aligned_labels if grad == 'high']
    low_t_labels = [t for grad, t, p in aligned_labels if grad == 'low']

    high_p_labels = [p for grad, t, p in aligned_labels if grad == 'high']
    low_p_labels = [p for grad, t, p in aligned_labels if grad == 'low']

    opt_task1_t_labels = [t for task, t, p in opt_task_aligned if task == 'opt_task1']
    opt_task1_p_labels = [p for task, t, p in opt_task_aligned if task == 'opt_task1']

    opt_task2_t_labels = [t for task, t, p in opt_task_aligned if task == 'opt_task2']
    opt_task2_p_labels = [p for task, t, p in opt_task_aligned if task == 'opt_task2']

    high_roc_auc = roc_auc_score(high_t_labels, high_p_labels) if len(set(high_t_labels)) > 1 else None
    low_roc_auc = roc_auc_score(low_t_labels, low_p_labels) if len(set(low_t_labels)) > 1 else None

    opt_task1_roc_auc = roc_auc_score(opt_task1_t_labels, opt_task1_p_labels) if len(set(opt_task1_t_labels)) > 1 else None
    opt_task2_roc_auc = roc_auc_score(opt_task2_t_labels, opt_task2_p_labels) if len(set(opt_task2_t_labels)) > 1 else None

    # For demonstration purposes, we'll just return the content with the selected model name

    # print(checkpoint)
    
    # if (inc_val<5):
    #     model_name="highGRschool10"
    # elif(inc_val>=5 & inc_val<10):
    #     model_name="highGRschool10"
    # else:
    #     model_name="highGRschool10"
    # Function to analyze each row
    def analyze_row(row):
    # Split the row into fields
        fields = row.split("\t")

        # Define tasks for OptionalTask_1, OptionalTask_2, and FinalAnswer
        optional_task_1_subtasks = ["DenominatorFactor", "NumeratorFactor", "EquationAnswer"]
        optional_task_2_subtasks = [
            "FirstRow2:1", "FirstRow2:2", "FirstRow1:1", "FirstRow1:2", 
            "SecondRow", "ThirdRow"
        ]

        # Helper function to evaluate task attempts
        def evaluate_tasks(fields, tasks):
            task_status = {}
            total_attempts = 0  # Counter for total number of attempts
            attempted_tasks = 0  # Counter for tasks attempted at least once
            successful_first_attempts = 0  # Counter for tasks successful on the first try
            for task in tasks:
                
                relevant_attempts = [f for f in fields if task in f]
                # if (task=="FinalAnswer"): print(relevant_attempts)
                attempt_count = len(relevant_attempts)
                total_attempts += attempt_count  # Add to the total attempts

                if attempt_count > 0:
                    attempted_tasks += 1  # Increment attempted tasks count

                    # Check the first attempt
                    first_attempt = relevant_attempts[0]
                    if "OK" in first_attempt and "ERROR" not in first_attempt and "JIT" not in first_attempt:
                        successful_first_attempts += 1


                if any("OK" in attempt for attempt in relevant_attempts):
                    task_status[task] = "Attempted (Successful)"
                    
                elif any("ERROR" in attempt for attempt in relevant_attempts):
                    task_status[task] = "Attempted (Error)"
                elif any("JIT" in attempt for attempt in relevant_attempts):
                    task_status[task] = "Attempted (JIT)"
                else:
                    task_status[task] = "Unattempted"
            return task_status,attempted_tasks, total_attempts,successful_first_attempts

        # Evaluate tasks for each category
        optional_task_1_status, opt1_attempted, opt1_total_attempts, opt1_successful_first_attempts  = evaluate_tasks(fields, optional_task_1_subtasks)
        optional_task_2_status, opt2_attempted, opt2_total_attempts, opt2_successful_first_attempts  = evaluate_tasks(fields, optional_task_2_subtasks)
        final_status, final_attempted, final_total_attempts,final_sucessful_first_attempts=evaluate_tasks(fields,["FinalAnswer-Attempt"])
        # print("/n",evaluate_tasks(fields,["FinalAnswer-Attempt"]))
        # Check if tasks have any successful attempt
        opt1_done = any(status == "Attempted (Successful)" for status in optional_task_1_status.values())
        opt2_done = any(status == "Attempted (Successful)" for status in optional_task_2_status.values())

        return (opt1_done, opt2_done,
                opt1_attempted, opt2_attempted, 
                opt1_total_attempts, opt2_total_attempts,
                opt1_successful_first_attempts, opt2_successful_first_attempts,final_total_attempts)

    # Read data from test_info.txt
    with open(test_info_location, "r") as file:
        data = file.readlines()

    # Assuming test_info[7] is a list with ideal tasks for each instance
    ideal_tasks = test_info[6]  # A list where each element is either 1 or 2

    # Initialize counters
    task_counts = {
    1: {"ER": 0, "ME": 0, "both": 0,"none":0},
    2: {"ER": 0, "ME": 0, "both": 0,"none":0}
    }
      # To store task completion counts per row
    # Analyze rows
    row_summary = []
    opt1_ratios = []
    opt2_ratios = []
    final_total=[]
    opt1_total=[]
    for i, row in enumerate(data):
        row = row.strip()
        if not row:
            continue

        ideal_task = ideal_tasks[i]  # Get the ideal task for the current row
        (
            opt1_done, opt2_done,
            opt1_attempted, opt2_attempted, 
            opt1_total_attempts, opt2_total_attempts,
            opt1_successful_first_attempts, opt2_successful_first_attempts,final_total_attemps
        ) = analyze_row(row)

        opt1_ratios.append( opt1_total_attempts / opt1_attempted if opt1_total_attempts > 0 else 0)
        opt2_ratios.append( opt2_total_attempts / opt2_attempted if opt2_total_attempts > 0 else 0)
        final_total.append(final_total_attemps)
        opt1_total.append(opt1_total_attempts)
    # create a summey for task:attempted, total attempts and succesful attempts for each row
    #     row_summary.append({
    #     "row_index": i + 1,
    #     "opt1": {
    #         "tasks_attempted": opt1_attempted,
    #         "total_attempts": opt1_total_attempts,
    #         "successful_attempts": opt1_successful_first_attempts,
    #     },
    #     "opt2": {
    #         "tasks_attempted": opt2_attempted,
    #         "total_attempts": opt2_total_attempts,
    #         "successful_attempts": opt2_successful_first_attempts,
    #     }
    # })
        if ideal_task == 0:
            if opt1_done and not opt2_done:
                task_counts[1]["ER"] += 1
            elif not opt1_done and opt2_done:
                task_counts[1]["ME"] += 1
            elif opt1_done and opt2_done:
                task_counts[1]["both"] += 1
            else:
                task_counts[1]["none"] +=1
        elif ideal_task == 1:
            if opt1_done and not opt2_done:
                task_counts[2]["ER"] += 1
            elif not opt1_done and opt2_done:
                task_counts[2]["ME"] += 1
            elif opt1_done and opt2_done:
                task_counts[2]["both"] += 1
            else:
                task_counts[2]["none"] +=1
    # Print a summary of task completions
    # for summary in row_summary:
    #     print(f"\nRow {summary['row_index']}:")
    #     print(f"  OptionalTask_1 - Tasks Attempted: {summary['opt1']['tasks_attempted']}, "
    #         f"Total Attempts: {summary['opt1']['total_attempts']}, "
    #         f"Successful Attempts: {summary['opt1']['successful_attempts']}")
    #     print(f"  OptionalTask_2 - Tasks Attempted: {summary['opt2']['tasks_attempted']}, "
    #         f"Total Attempts: {summary['opt2']['total_attempts']}, "
    #         f"Successful Attempts: {summary['opt2']['successful_attempts']}")


    # Create a string output for results
    # output_summary = "Task Analysis Summary:\n"
    # output_summary += "-----------------------\n"

    # for ideal_task, counts in task_counts.items():
    #     output_summary += f"Ideal Task = OptionalTask_{ideal_task}:\n"
    #     output_summary += f"  Only OptionalTask_1 done: {counts['ER']}\n"
    #     output_summary += f"  Only OptionalTask_2 done: {counts['ME']}\n"
    #     output_summary += f"  Both done: {counts['both']}\n"


    # Create figure
    fig_hist1 = go.Figure()

    # Add histogram for OptionalTask_1 (ER)
    fig_hist1.add_trace(go.Histogram(
        x=opt1_ratios,
        name="ER",
        marker=dict(color='blue'),
        opacity=1,
        xbins=dict(
            start=1.0,
            end=max(opt1_ratios) if max(opt1_ratios) < 15 else 15,
            size=1  # Bin width set to 0.1 for 10 bins
        )
    ))
    fig_hist2=go.Figure()
    # Add histogram for OptionalTask_2 (ME)
    fig_hist2.add_trace(go.Histogram(
        x=opt2_ratios,
        name="ME",
        marker=dict(color='red'),
        opacity=1,
        xbins=dict(
            start=1.0,
            end=max(opt1_ratios) if max(opt1_ratios) < 15 else 15,
            size=1  # Bin width set to 0.1 for 10 bins
        )
    ))

    # Update layout
    fig_hist1.update_layout(
        title="ER: Histogram of Attempts required per task",
        title_x=0.5,
        xaxis=dict(
            title="Success Ratio ( Total Attempts / Tasks Attempted )",
            tickmode="array",
            tickvals=list(range(1, 11)) + [15],  # 1,2,3,...10, 15+  
            ticktext=[str(i) for i in range(1, 11)] + ["10+"],
        ),
        yaxis=dict(
            title="Number of Instances"
        ),
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        legend=dict(
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            )
        ),
        barmode='overlay'  # Overlapping bars
    )
    fig_hist2.update_layout(
        title="ME: Histogram of Attempts required per task",
        title_x=0.5,
        xaxis=dict(
            title="Success Ratio (Total Attempts / Tasks Attempted)",
            tickmode="array",
            tickvals=list(range(1, 11)) + [15],  # 1,2,3,...10, 15+  
            ticktext=[str(i) for i in range(1, 11)] + ["10+"],
        ),
        yaxis=dict(
            title="Number of Instances"
        ),
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        legend=dict(
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            )
        ),
        barmode='overlay'  # Overlapping bars
    )
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors = ["#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9"]
    # print(opt1_ratios,opt2_ratios)
    # fig_scatter = go.Figure()

    # fig_scatter.add_trace(go.Scatter(
    #     x=final_total,
    #     y=opt1_total,
    #     mode='markers',
    #     marker=dict(size=8, color='blue', opacity=0.7),
    #     name="Student Data"
    # ))

    # # Update layout
    # fig_scatter.update_layout(
    #     title="Scatter Plot: Final Total Attempts vs OptionalTask_1 Attempts",
    #     title_x=0.5,
    #     xaxis=dict(title="Final Total Attempts"),
    #     yaxis=dict(title="OptionalTask_1 Total Attempts"),
    #     font=dict(family="sans-serif", size=12, color="black"),
    #     showlegend=True
    # )

    # fig_scatter.show()
  # Generate pie chart for Task 1
    task1_labels = list(task_counts[1].keys())
    task1_values = list(task_counts[1].values())

    # fig_task1 = Figure()
    # ax1 = fig_task1.add_subplot(1, 1, 1)
    # ax1.pie(task1_values, labels=task1_labels, autopct='%1.1f%%', startangle=90)
    # ax1.set_title('Ideal Task 1 Distribution')

    fig_task1 = go.Figure(data=[go.Pie(
        labels=task1_labels,
        values=task1_values,
        textinfo='percent+label',
        textposition='auto',
        marker=dict(colors=colors),
        sort=False
       
    )])

    fig_task1.update_layout(
        title='Problem Type: ER',
        title_x=0.5,
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )

    fig_task1.update_layout(
    legend=dict(
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
            ),
        )
    )
    


    # fig.show()

    # Generate pie chart for Task 2
    task2_labels = list(task_counts[2].keys())
    task2_values = list(task_counts[2].values())

    fig_task2 = go.Figure(data=[go.Pie(
        labels=task2_labels,
        values=task2_values,
        textinfo='percent+label',
        textposition='auto',
        marker=dict(colors=colors),
        sort=False
        # pull=[0, 0.2, 0, 0] # for pulling part of pie chart out (depends on position)
        
    )])

    fig_task2.update_layout(
        title='Problem Type: ME',
        title_x=0.5,
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )

    fig_task2.update_layout(
    legend=dict(
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
            ),
        )
    )


    # fig_task2 = Figure()
    # ax2 = fig_task2.add_subplot(1, 1, 1)
    # ax2.pie(task2_values, labels=task2_labels, autopct='%1.1f%%', startangle=90)
    # ax2.set_title('Ideal Task 2 Distribution')

    # print(output_summary)

    
    result = {}
    with open("fileHandler/result.txt", 'r') as file:
        for line in file:
            key, value = line.strip().split(': ', 1)
            # print(type(key))
            if key=='epoch':
                result[key]=value
            else:
                 result[key]=float(value)
    result["ROC score of HGR"]=high_roc_auc
    result["ROC score of LGR"]=low_roc_auc
# Create a plot
    with open("fileHandler/roc_data.pkl", "rb") as f:
        fpr, tpr, _ = pickle.load(f)
    # print(fpr,tpr)
    roc_auc = auc(fpr, tpr)


#  Create a matplotlib figure
    # fig = Figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'Receiver Operating Curve (ROC)')
    # ax.legend(loc="lower right")
    # ax.grid()

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Line(x = list(fpr), y = list(tpr), name=f'ROC curve (area = {roc_auc:.2f})',
                            line=dict(color='royalblue', width=3,
                                ) # dash options include 'dash', 'dot', and 'dashdot'
    ))
    fig.add_trace(go.Line(x = [0,1], y = [0,1], showlegend = False,
                            line=dict(color='firebrick', width=2,
                                dash='dash',) # dash options include 'dash', 'dot', and 'dashdot'
    ))

    # Edit the layout
    fig.update_layout(
            showlegend = True,
            title_x=0.5,
            title=dict(
                text='Receiver Operating Curve (ROC)'
            ),
            xaxis=dict(
                title=dict(
                    text='False Positive Rate'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='False Negative Rate'
                )
            ),
            font=dict(
            family="sans-serif",
            color="black"
            ),
        
    )
    fig.update_layout(
    legend=dict(
        x=0.75,
        y=0,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
            ),
        )
    )






    # Save plot to a file
    # plot_path = "plot.png"
    # fig.savefig(plot_path)
    # plt.close(fig)


    

    progress(1.0)
    # Prepare text output
    text_output = f"Model: {model_name}\nResult:\n{result}"
    # Prepare text output with HTML formatting
    text_output = f"""
    ---------------------------
    Model: {model_name}
    ---------------------------\n
    Time Taken: {result['time_taken_from_start']:.2f} seconds
    Total Schools in test: {len(unique_schools):.4f}
    Total number of instances having Schools with HGR : {len(high_sample):.4f}
    Total number of instances having Schools with LGR: {len(low_sample):.4f}

    ROC score of HGR: {high_roc_auc:.4f}
    ROC score of LGR: {low_roc_auc:.4f}


    ROC-AUC for problems of type ER: {opt_task1_roc_auc:.4f}
    ROC-AUC for problems of type ME: {opt_task2_roc_auc:.4f}
    """
    progress(0.5,desc="first k '%' sampling")
    subprocess.run([
    "python", "new_test_saved_finetuned_model.py",
    "-workspace_name", "ratio_proportion_change3_2223/sch_largest_100-coded",
    "-finetune_task", finetune_task,
    "-test_dataset_path","../../../../fileHandler/selected_rows_first20.txt",
    # "-test_label_path","../../../../train_label.txt",
    "-finetuned_bert_classifier_checkpoint", 
    "ratio_proportion_change3_2223/sch_largest_100-coded/output/highGRschool10/bert_fine_tuned.model.ep42",
    "-e",str(1),
    "-b",str(1000)
])
    with open("fileHandler/roc_data.pkl", "rb") as f:
        fpr, tpr, _ = pickle.load(f)
    # print(fpr,tpr)
    roc_auc_first_k = auc(fpr, tpr)
    print(roc_auc_first_k)


    progress(0.5,desc="last '%' sampling")
    subprocess.run([
    "python", "new_test_saved_finetuned_model.py",
    "-workspace_name", "ratio_proportion_change3_2223/sch_largest_100-coded",
    "-finetune_task", finetune_task,
    "-test_dataset_path","../../../../fileHandler/selected_rows_last20.txt",
    # "-test_label_path","../../../../train_label.txt",
    "-finetuned_bert_classifier_checkpoint", 
    "ratio_proportion_change3_2223/sch_largest_100-coded/output/highGRschool10/bert_fine_tuned.model.ep42",
    "-e",str(1),
    "-b",str(1000)
])
    with open("fileHandler/roc_data.pkl", "rb") as f:
        fpr, tpr, _ = pickle.load(f)
    # print(fpr,tpr)
    roc_auc_last_k = auc(fpr, tpr)
    print(roc_auc_last_k)



    text_output_sampled_auc = f"""
        ---------------------------
        Model: {model_name}
        ---------------------------\n

        ROC score of first 20% of problems: {roc_auc_first_k:.4f}
        ROC score of last 20% of problems: {roc_auc_last_k:.4f}

    """












    return text_output,text_output_sampled_auc,fig,fig_task1,fig_task2,fig_hist1,fig_hist2
# List of models for the dropdown menu

# models = ["ASTRA-FT-HGR", "ASTRA-FT-LGR", "ASTRA-FT-FULL"]
models = ["ASTRA-FT-HGR", "ASTRA-FT-FULL"]
content = """
<h1 style="color: black;">A S T R A</h1>
<h2 style="color: black;">An AI Model for Analyzing Math Strategies</h2>

<h3 style="color: white; text-align: center">
    <a href="https://drive.google.com/file/d/1lbEpg8Se1ugTtkjreD8eXIg7qrplhWan/view" style="color: gr.themes.colors.red; text-decoration: none;">Link To Paper</a> | 
    <a href="https://github.com/Syudu41/ASTRA---Gates-Project" style="color: #1E90FF; text-decoration: none;">GitHub: Astra Demo</a> | 
    <a href="https://sites.google.com/view/astra-research/home" style="color: #1E90FF; text-decoration: none;">Project Page</a> | 
    <a href="https://path-analysis.vercel.app/" style="color: #1E90FF; text-decoration: none;">Path Analysis Tool</a> | 
    <a href="https://github.com/CarnegieLearningWeb/PathAnalysis" style="color: #1E90FF; text-decoration: none;">Github: Path Analysis Tool</a> | 
    
    
</h3>

<p style="color: white;">Welcome to a demo of ASTRA. ASTRA is a collaborative research project between researchers at the 
<a href="https://sites.google.com/site/dvngopal/" style="color: #1E90FF; text-decoration: none;">University of Memphis</a> and 
<a href="https://www.carnegielearning.com" style="color: #1E90FF; text-decoration: none;">Carnegie Learning</a> 
to utilize AI to improve our understanding of math learning strategies.</p>

<p style="color: white;">This demo has been developed with a pre-trained model (based on an architecture similar to BERT ) that learns math strategies using data 
collected from hundreds of schools in the U.S. who have used Carnegie Learning’s MATHia (formerly known as Cognitive Tutor), the flagship Intelligent Tutor that is part of a core, blended math curriculum. 
For this demo, we have used data from a specific domain (teaching ratio and proportions) within 7th grade math. The fine-tuning based on the pre-trained model learns to predict which strategies lead to correct vs incorrect solutions. 
</p>

<p style="color: white;">In this math domain, students were given word problems related to ratio and proportions. Further, the students 
were given a choice of optional tasks to work on in parallel to the main problem to demonstrate  their thinking (metacognition). 
The optional tasks are designed based on solving problems using Equivalent Ratios (ER) and solving using Means and Extremes/cross-multiplication (ME).
When the equivalent ratios are easy to compute (integral values), ER is much more efficient compared to ME and switching between the tasks appropriately demonstrates cognitive flexibility.
</p>

<p style="color: white;">To use the demo, please follow these steps:</p>

<ol style="color: white;">
    <li style="color: white;">Select a fine-tuned model:
        <ul style="color: white;">
            <li style="color: white;">ASTRA-FT-HGR: Fine-tuned with a small sample of data from schools that have a high graduation rate.</li>
            <li style="color: white;">ASTRA-FT-Full: Fine-tuned with a small sample of data from a mix of schools that have high/low graduation rates.</li>
        </ul>
    </li>
    <li style="color: white;">Select a percentage of schools to analyze (selecting a large percentage may take a long time). Note that the selected percentage is applied to both High Graduation Rate (HGR) schools and Low Graduation Rate (LGR schools).
</li>
    <li style="color: white;">The results from the fine-tuned model are displayed in the dashboard:
        <ul>
            <li style="color: white;">The model accuracy is computed using the ROC-AUC metric.
</li>
            <li style="color: white;">The results are shown for HGR, LGR schools and  for different problem types (ER/ME). 
</li>
<li style="color: white;">The distribution over how students utilized the optional tasks (whether they utilized ER/ME, used both of them or none of them) is shown for each problem type. 
</li>
        </ul>
    </li>
</ol>
"""
# CSS styling for white text
# Create the Gradio interface
available_themes = {
    "default": gr.themes.Default(),
    "soft": gr.themes.Soft(),
    "monochrome": gr.themes.Monochrome(),
    "glass": gr.themes.Glass(),
    "base": gr.themes.Base(),
}

# Comprehensive CSS for all HTML elements
custom_css = '''
/* Import Fira Sans font */
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&family=Spectral+SC:wght@600&display=swap');
/* Container modifications for centering */
.gradio-container {
    color: var(--block-label-text-color) !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
    font-family: Arial, sans-serif !important;
}

/* Main title (ASTRA) */
#title {
    text-align: center !important;
    margin: 1rem auto !important;  /* Reduced margin */
    font-size: 2.5em !important;
    font-weight: 600 !important;
    font-family: "Spectral SC", 'Fira Sans', sans-serif !important;
    padding-bottom: 0 !important;  /* Remove bottom padding */
}

/* Subtitle (An AI Model...) */
h1 {
    text-align: center !important;
    font-size: 30pt !important;
    font-weight: 600 !important;
    font-family: "Spectral SC", 'Fira Sans', sans-serif !important;
    margin-top: 0.5em !important;  /* Reduced top margin */
    margin-bottom: 0.3em !important;
}

h2 {
    text-align: center !important;
    font-size: 22pt !important;
    font-weight: 600 !important;
    font-family: "Spectral SC",'Fira Sans', sans-serif !important;
    margin-top: 0.2em !important;  /* Reduced top margin */
    margin-bottom: 0.3em !important;
}

/* Links container styling */
.links-container {
    text-align: center !important;
    margin: 1em auto !important;
    font-family: 'Inter' ,'Fira Sans', sans-serif !important;
}

/* Links */
a {
    color: #2563eb !important;
    text-decoration: none !important;
    font-family:'Inter' , 'Fira Sans', sans-serif !important;
}

a:hover {
    text-decoration: underline !important;
    opacity: 0.8;
}

/* Regular text */
p, li, .description, .markdown-text {
    font-family: 'Inter', Arial, sans-serif !important;
    color: black !important;
    font-size: 11pt;
    line-height: 1.6;
    font-weight: 500 !important;
    color: var(--block-label-text-color) !important;
}

/* Other headings */
h3, h4, h5 {
    font-family: 'Fira Sans', sans-serif !important;
    color: var(--block-label-text-color) !important;
    margin-top: 1.5em;
    margin-bottom: 0.75em;
}


h3 { font-size: 1.5em; font-weight: 600; }
h4 { font-size: 1.25em; font-weight: 500; }
h5 { font-size: 1.1em; font-weight: 500; }

/* Form elements */
.select-wrap select, .wrap select,
input, textarea {
    font-family: 'Inter' ,Arial, sans-serif !important;
    color: var(--block-label-text-color) !important;
}

/* Lists */
ul, ol {
    margin-left: 0 !important;
    margin-bottom: 1.25em;
    padding-left: 2em;
}

li {
    margin-bottom: 0.75em;
}

/* Form container */
.form-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 1rem !important;
}

/* Dashboard */
.dashboard {
    margin-top: 2rem !important;
    padding: 1rem !important;
    border-radius: 8px !important;
}

/* Slider styling */
.gradio-slider-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.5em 0;
    max-width: 100% !important;
}

.gradio-slider {
    flex-grow: 1;
    margin-right: 15px;
}

.slider-percentage {
    font-family: 'Inter', Arial, sans-serif !important;
    flex-shrink: 0;
    min-width: 60px;
    font-size: 1em;
    font-weight: bold;
    text-align: center;
    background-color: #f0f8ff;
    border: 1px solid #004080;
    border-radius: 5px;
    padding: 5px 10px;
}

.progress-bar-wrap.progress-bar-wrap.progress-bar-wrap
{
	border-radius: var(--input-radius);
	height: 1.25rem;
	margin-top: 1rem;
	overflow: hidden;
	width: 70%;
    font-family: 'Inter', Arial, sans-serif !important;
}

/* Add these new styles after your existing CSS */

/* Card-like appearance for the dashboard */
.dashboard {
    background: #ffffff !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    margin-top: 2.5rem !important;
}

/* Enhance ROC graph container */
#roc {
    background: #ffffff !important;
    padding: 1.5rem !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    margin: 1.5rem 0 !important;
}

/* Style the dropdown select */
select {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

select:hover {
    border-color: #cbd5e1 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

/* Enhance slider appearance */
.progress-bar-wrap {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

/* Style metrics in dashboard */
.dashboard p {
    padding: 0.5rem 0 !important;
    border-bottom: 1px solid #f1f5f9 !important;
}

/* Add spacing between sections */
.dashboard > div {
    margin-bottom: 1.5rem !important;
}

/* Style the ROC curve title */
.dashboard h4 {
    color: #1e293b !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid #e2e8f0 !important;
}

/* Enhance link appearances */
a {
    position: relative !important;
    padding-bottom: 2px !important;
    transition: all 0.2s ease-in-out !important;
}

a:after {
    content: '' !important;
    position: absolute !important;
    width: 0 !important;
    height: 1px !important;
    bottom: 0 !important;
    left: 0 !important;
    background-color: #2563eb !important;
    transition: width 0.3s ease-in-out !important;
}

a:hover:after {
    width: 100% !important;
}

/* Add subtle dividers between sections */
.form-container > div {
    padding-bottom: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    border-bottom: 1px solid #f1f5f9 !important;
}

/* Style model selection section */
.select-wrap {
    background: #ffffff !important;
    padding: 1.5rem !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    margin-bottom: 2rem !important;
}

/* Style the metrics display */
.dashboard span {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #334155 !important;
}

/* Add subtle animation to interactive elements */
button, select, .slider-percentage {
    transition: all 0.2s ease-in-out !important;
}

/* Style the ROC curve container */
.plot-container {
    background: #ffffff !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

/* Add container styles for opt1 and opt2 sections */
#opt1, #opt2 {
    background: #ffffff !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    margin-top: 1.5rem !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

/* Style the distribution titles */
.distribution-title {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-bottom: 1rem !important;
    text-align: center !important;
}

'''

with gr.Blocks(theme='gstaff/sketch', css=custom_css) as demo:
    
    # gr.Markdown("<h1 id='title'>ASTRA</h1>", elem_id="title")
    gr.Markdown(content)
    
    with gr.Row():
        # file_input = gr.File(label="Upload a test file", file_types=['.txt'], elem_classes="file-box")
        # label_input = gr.File(label="Upload test labels", file_types=['.txt'], elem_classes="file-box")

        # info_input = gr.File(label="Upload test info", file_types=['.txt'], elem_classes="file-box")
        model_dropdown = gr.Dropdown(
            choices=models,
            label="Select Fine-tuned Model",
            elem_classes="dropdown-menu"
        )
        increment_slider = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            label="Schools Percentage",
            value=1,
            elem_id="increment-slider",
            elem_classes="gradio-slider"
        )
    
    with gr.Row():
        btn = gr.Button("Submit")

    gr.Markdown("<p class='description'>Dashboard</p>")

    with gr.Row():
        output_text = gr.Textbox(label="")
        # output_image = gr.Image(label="ROC")
    with gr.Row():
        plot_output = gr.Plot(label="ROC")

    with gr.Row():
        opt1_pie = gr.Plot(label="ER")
        opt2_pie = gr.Plot(label="ME")
        # output_summary = gr.Textbox(label="Summary")
    with gr.Row():
        histo1 = gr.Plot(label="Hist")
        histo2 = gr.Plot(label="Hist")
    with gr.Row():
        output_text_sampled_auc = gr.Textbox(label="")
  

    
 
    btn.click(
        fn=process_file, 
        inputs=[model_dropdown,increment_slider], 
        outputs=[output_text,output_text_sampled_auc,plot_output,opt1_pie,opt2_pie,histo1,histo2]
    )


# Launch the app
demo.launch()