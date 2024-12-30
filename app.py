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
        finetune_task="highGRschool10"
    else:
        finetune_task=None
    # Load the test_info file and the graduation rate file
    test_info = pd.read_csv(test_info_location, sep=',', header=None, engine='python')
    grad_rate_data = pd.DataFrame(pd.read_pickle('school_grduation_rate.pkl'),columns=['school_number','grad_rate'])  # Load the grad_rate data

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

    # Load the test file and select rows based on indices
    test = pd.read_csv(test_location, sep=',', header=None, engine='python')
    selected_rows_df2 = test.loc[indices]

    # Save the selected rows to a file
    selected_rows_df2.to_csv('selected_rows.txt', sep='\t', index=False, header=False, quoting=3, escapechar=' ')

   
    # For demonstration purposes, we'll just return the content with the selected model name

    # print(checkpoint)
    progress(0.1, desc="Files created and saved")
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
        final_answer_tasks = ["FinalAnswer"]

        # Helper function to evaluate task attempts
        def evaluate_tasks(fields, tasks):
            task_status = {}
            for task in tasks:
                relevant_attempts = [f for f in fields if task in f]
                # print(relevant_attempts)
                if any("OK" in attempt for attempt in relevant_attempts):
                    task_status[task] = "Attempted (Successful)"
                elif any("ERROR" in attempt for attempt in relevant_attempts):
                    task_status[task] = "Attempted (Error)"
                elif any("JIT" in attempt for attempt in relevant_attempts):
                    task_status[task] = "Attempted (JIT)"
                else:
                    task_status[task] = "Unattempted"
            return task_status

        # Evaluate tasks for each category
        optional_task_1_status = evaluate_tasks(fields, optional_task_1_subtasks)
        optional_task_2_status = evaluate_tasks(fields, optional_task_2_subtasks)
        final_answer_status = evaluate_tasks(fields, final_answer_tasks)

        # Combine results
        result = {
            "OptionalTask_1": optional_task_1_status,
            "OptionalTask_2": optional_task_2_status,
            "FinalAnswer": final_answer_status,
        }
        return result
    # Read data from test_info.txt
    with open(test_info_location, "r") as file:
        data = file.readlines()
    results = [analyze_row(row.strip()) for row in data if row.strip()]

    status_counts = {}


    for result in results:
        for task_category, task_statuses in result.items():
            for task, status in task_statuses.items():
                if task not in status_counts:
                    status_counts[task] = {"Attempted (Successful)": 0, "Attempted (Error)": 0, 
                                        "Attempted (JIT)": 0, "Unattempted": 0}
                status_counts[task][status] += 1

    # Create a string output for results
    output_summary = "Task Analysis Summary:\n"
    output_summary += "-----------------------\n"

    for task, statuses in status_counts.items():
        output_summary += f"Task: {task}\n"
        for status, count in statuses.items():
            output_summary += f"  {status}: {count}\n"


    progress(0.2, desc="analysis done!! Executing models")
    subprocess.run([
        "python", "new_test_saved_finetuned_model.py",
        "-workspace_name", "ratio_proportion_change3_2223/sch_largest_100-coded",
        "-finetune_task", finetune_task,
        "-test_dataset_path","../../../../selected_rows.txt",
        # "-test_label_path","../../../../train_label.txt",
        "-finetuned_bert_classifier_checkpoint", 
        "ratio_proportion_change3_2223/sch_largest_100-coded/output/highGRschool10/bert_fine_tuned.model.ep42",
        "-e",str(1),
        "-b",str(1000)
    ])
    progress(0.6,desc="Model execution completed")
    result = {}
    with open("result.txt", 'r') as file:
        for line in file:
            key, value = line.strip().split(': ', 1)
            # print(type(key))
            if key=='epoch':
                result[key]=value
            else:
                 result[key]=float(value)
# Create a plot
    with open("roc_data.pkl", "rb") as f:
        fpr, tpr, _ = pickle.load(f)

    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'(Receiver Operating Curve) ROC')
    ax.legend(loc="lower right")
    ax.grid()

    # Save plot to a file
    plot_path = "plot.png"
    fig.savefig(plot_path)
    plt.close(fig)
    progress(1.0)
    # Prepare text output
    text_output = f"Model: {model_name}\nResult:\n{result}"
    # Prepare text output with HTML formatting
    text_output = f"""
    Model: {model_name}\n
    -----------------\n

    Time Taken: {result['time_taken_from_start']:.2f} seconds\n
    Total Schools in test: {len(unique_schools):.4f}\n
    Total number of instances having Schools with HGR : {len(high_sample):.4f}\n
    Total number of instances having Schools with LGR: {len(low_sample):.4f}\n
    -----------------\n
    """
    return text_output,plot_path

# List of models for the dropdown menu

models = ["ASTRA-FT-HGR", "ASTRA-FT-LGR", "ASTRA-FT-FULL"]
content = """
<h1 style="color: white;">ASTRA: An AI Model for Analyzing Math Strategies</h1>

<h3 style="color: white;">
    <a href="https://drive.google.com/file/d/1lbEpg8Se1ugTtkjreD8eXIg7qrplhWan/view" style="color: #1E90FF; text-decoration: none;">Link To Paper</a> | 
    <a href="https://github.com/Syudu41/ASTRA---Gates-Project" style="color: #1E90FF; text-decoration: none;">GitHub</a> | 
    <a href="#" style="color: #1E90FF; text-decoration: none;">Project Page</a>
</h3>

<p style="color: white;">Welcome to a demo of ASTRA. ASTRA is a collaborative research project between researchers at the 
<a href="https://www.memphis.edu" style="color: #1E90FF; text-decoration: none;">University of Memphis</a> and 
<a href="https://www.carnegielearning.com" style="color: #1E90FF; text-decoration: none;">Carnegie Learning</a> 
to utilize AI to improve our understanding of math learning strategies.</p>

<p style="color: white;">This demo has been developed with a pre-trained model (based on an architecture similar to BERT) 
that learns math strategies using data collected from hundreds of schools in the U.S. who have used 
Carnegie Learning's MATHia (formerly known as Cognitive Tutor), the flagship Intelligent Tutor 
that is part of a core, blended math curriculum.</p>

<p style="color: white;">For this demo, we have used data from a specific domain (teaching ratio and proportions) within 
7th grade math. The fine-tuning based on the pre-trained models learns to predict which strategies 
lead to correct vs. incorrect solutions.</p>

<p style="color: white;">To use the demo, please follow these steps:</p>

<ol style="color: white;">
    <li style="color: white;">Select a fine-tuned model:
        <ul style="color: white;">
            <li style="color: white;">ASTRA-FT-HGR: Fine-tuned with a small sample of data from schools that have a high graduation rate.</li>
            <li style="color: white;">ASTRA-FT-LGR: Fine-tuned with a small sample of data from schools that have a low graduation rate.</li>
            <li style="color: white;">ASTRA-FT-Full: Fine-tuned with a small sample of data from a mix of schools that have high/low graduation rates.</li>
        </ul>
    </li>
    <li style="color: white;">Select a percentage of schools to analyze (selecting a large percentage may take a long time).</li>
    <li style="color: white;">View Results:
        <ul>
            <li style="color: white;">The results from the fine-tuned model are displayed on the dashboard.</li>
            <li style="color: white;">The results are shown separately for schools that have high and low graduation rates.</li>
        </ul>
    </li>
</ol>
"""
# CSS styling for white text
# Create the Gradio interface
with gr.Blocks(css="""
    body {
        background-color: #1e1e1e!important;
        font-family: 'Arial', sans-serif;
        color: #f5f5f5!important;;
    }

    .gradio-container {
        max-width: 850px!important;
        margin: 0 auto!important;;
        padding: 20px!important;;
        background-color: #292929!important;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .gradio-container-4-44-0 .prose h1 {
    font-size: var(--text-xxl);
    color: #ffffff!important;
}
    #title {
        color: white!important;
        font-size: 2.3em;
        font-weight: bold;
        text-align: center!important;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 1.1em;
        color: #bfbfbf;
        margin-bottom: 30px;
    }
    .file-box {
        max-width: 180px;
        padding: 5px;
        background-color: #444!important;
        border: 1px solid #666!important;
        border-radius: 6px;
        height: 80px!important;;  
        margin: 0 auto!important;; 
        text-align: center; 
        color: transparent;
    }
    .file-box span {
        color: #f5f5f5!important;
        font-size: 1em;
        line-height: 45px; /* Vertically center text */
    }
    .dropdown-menu {
        max-width: 220px;
        margin: 0 auto!important;
        background-color: #444!important;
        color:#444!important;
        border-radius: 6px;
        padding: 8px;
        font-size: 1.1em;
        border: 1px solid #666;
    }
    .button {
        background-color: #4CAF50!important;
        color: white!important;
        font-size: 1.1em;
        padding: 10px 25px;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out;
    }
    .button:hover {
        background-color: #45a049!important;
    }
    .output-text {
        background-color: #333!important;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #666;
        font-size: 1.1em;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 0.9em;
        color: #b0b0b0;
    }
    .svelte-12ioyct .wrap {
    display: none !important;
}
.file-label-text {
    display: none !important;
}

div.svelte-sfqy0y {
    display: flex;
    flex-direction: inherit;
    flex-wrap: wrap;
    gap: var(--form-gap-width);
    box-shadow: var(--block-shadow);
    border: var(--block-border-width) solid var(--border-color-primary);
    border-radius: var(--block-radius);
    background: #1f2937!important;
    overflow-y: hidden;
}

.block.svelte-12cmxck {
    position: relative;
    margin: 0;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: var(--block-border-color);
    border-radius: var(--block-radius);
    background: #1f2937!important;
    width: 100%;
    line-height: var(--line-sm);
}

    .svelte-12ioyct .wrap {
    display: none !important;
}
.file-label-text {
    display: none !important;
}
input[aria-label="file upload"] {
    display: none !important;
}

gradio-app .gradio-container.gradio-container-4-44-0 .contain .file-box span {
    font-size: 1em;
    line-height: 45px;
    color: #1f2937 !important;
}
.wrap.svelte-12ioyct {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: var(--size-60);
    color: #1f2937 !important;
    line-height: var(--line-md);
    height: 100%;
    padding-top: var(--size-3);
    text-align: center;
    margin: auto var(--spacing-lg);
}
span.svelte-1gfkn6j:not(.has-info) {
    margin-bottom: var(--spacing-lg);
    color: white!important;
}
label.float.svelte-1b6s6s {
    position: relative!important;
    top: var(--block-label-margin);
    left: var(--block-label-margin);
}
label.svelte-1b6s6s {
    display: inline-flex;
    align-items: center;
    z-index: var(--layer-2);
    box-shadow: var(--block-label-shadow);
    border: var(--block-label-border-width) solid var(--border-color-primary);
    border-top: none;
    border-left: none;
    border-radius: var(--block-label-radius);
    background: rgb(120 151 180)!important;
    padding: var(--block-label-padding);
    pointer-events: none;
    color: #1f2937!important;
    font-weight: var(--block-label-text-weight);
    font-size: var(--block-label-text-size);
    line-height: var(--line-sm);
}
.file.svelte-18wv37q.svelte-18wv37q {
    display: block!important;
    width: var(--size-full);
}

tbody.svelte-18wv37q>tr.svelte-18wv37q:nth-child(odd) {
    background: ##7897b4!important;
    color: white;
    background: #aca7b2;
}

.gradio-container-4-31-4 .prose h1, .gradio-container-4-31-4 .prose h2, .gradio-container-4-31-4 .prose h3, .gradio-container-4-31-4 .prose h4, .gradio-container-4-31-4 .prose h5 {

    color: white;
}
""") as demo:
    
    gr.Markdown("<h1 id='title'>ASTRA</h1>", elem_id="title")
    gr.Markdown(content)
    
    with gr.Row():
        # file_input = gr.File(label="Upload a test file", file_types=['.txt'], elem_classes="file-box")
        # label_input = gr.File(label="Upload test labels", file_types=['.txt'], elem_classes="file-box")

        # info_input = gr.File(label="Upload test info", file_types=['.txt'], elem_classes="file-box")
    
        model_dropdown = gr.Dropdown(choices=models, label="Select Fine-tuned Model", elem_classes="dropdown-menu")

    
    increment_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Schools Percentage", value=1)
    gr.Markdown("<p class='description'>Dashboard</p>")
    with gr.Row():
        output_text = gr.Textbox(label="")
        output_image = gr.Image(label="ROC")
        # output_summary = gr.Textbox(label="Summary")

    btn = gr.Button("Submit")
 
    btn.click(fn=process_file, inputs=[model_dropdown,increment_slider], outputs=[output_text,output_image])


# Launch the app
demo.launch()