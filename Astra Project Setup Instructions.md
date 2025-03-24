# Astra Project Setup Instructions

## Prerequisites
Make sure you have the following installed before proceeding:
- Python 3.12.4
- Git
- Git Large File Storage (LFS)

## Step 1: Install Git LFS
Git LFS (Large File Storage) is required for managing large files in the Astra project. Follow these steps to install Git LFS:

### Windows
1. Download the Git LFS installer from [Git LFS Releases](https://git-lfs.github.com/).
2. Run the installer and follow the setup instructions.
3. Open a terminal (Command Prompt or PowerShell) and run:
   ```sh
   git lfs install
   ```

### macOS
1. Install Git LFS using Homebrew:
   ```sh
   brew install git-lfs
   ```
2. Initialize Git LFS:
   ```sh
   git lfs install
   ```

### Linux
1. Install Git LFS using your package manager:
   - Debian/Ubuntu:
     ```sh
     sudo apt install git-lfs
     ```
   - Fedora:
     ```sh
     sudo dnf install git-lfs
     ```
   - Arch Linux:
     ```sh
     sudo pacman -S git-lfs
     ```
2. Initialize Git LFS:
   ```sh
   git lfs install
   ```

## Step 2: Install Python (Alternative: pyenv)
While Python 3.12.4 is required, it is recommended to use `pyenv` if you want to work with multiple Python versions or if you encounter errors while installing dependencies.

### Installing pyenv
#### macOS & Linux:
```sh
curl https://pyenv.run | bash
```
After installation, restart your terminal and install Python:
```sh
pyenv install 3.12.4
pyenv global 3.12.4
```

#### Windows:
Use [pyenv-win](https://github.com/pyenv-win/pyenv-win):
```sh
git clone https://github.com/pyenv-win/pyenv-win.git ~/.pyenv
setx PYENV "%USERPROFILE%\.pyenv"
setx PATH "%PYENV%\bin;%PYENV%\shims;%PATH%"
pyenv install 3.12.4
pyenv global 3.12.4
```

## Step 3: Clone the Repository
Clone the Astra project repository using Git:
```sh
git clone <repository_url>
cd astra
```

## Step 4: Install Dependencies
Install all required dependencies from the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

## Step 5: Verify Installation
Ensure all dependencies are installed correctly by running:
```sh
python --version
pip list
```

## Step 6: Run the Application or Test the Model
You have two options to proceed:

### Option 1: Run the Gradio App
To open the Gradio app in your web browser and interact with the application, run:
```sh
python app.py
```

### Option 2: Test the Model with a Sample File
To test the fine-tuned model using a sample file, navigate to the root folder of the project and run the following command:
```sh
cd <root_folder>
python new_test_saved_finetuned_model.py \
    -workspace_name "ratio_proportion_change3_2223/sch_largest_100-coded" \
    -finetune_task "<finetune_task>" \
    -test_dataset_path "../../../../fileHandler/selected_rows.txt" \
    -finetuned_bert_classifier_checkpoint "ratio_proportion_change3_2223/sch_largest_100-coded/output/highGRschool10/bert_fine_tuned.model.ep42" \
    -e 1 \
    -b 1000
```
Replace `<finetune_task>` with the actual fine-tuning task value.

Your Astra project should now be fully set up and ready to use!
