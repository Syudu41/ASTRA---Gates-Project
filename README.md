# ASTRA (AI for STRategy Analysis)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)


ASTRA is a collaborative research project between the University of Memphis and Carnegie Learning focused on leveraging AI to enhance our understanding of mathematical learning strategies.

## Project Overview

ASTRA utilizes a pre-trained model (based on a BERT-like architecture) to learn mathematics strategies from extensive data collected through Carnegie Learning's MATHia platform (formerly known as Cognitive Tutor). The data comes from hundreds of U.S. schools using this Intelligent Tutor as part of their core, blended math curriculum.

## Current Focus

The current demonstration focuses on:
- Domain: Ratio and Proportions
- Grade Level: 7th Grade
- Objective: Predicting strategies that lead to correct vs. incorrect solutions

## Available Models

ASTRA offers three fine-tuned models:

1. **ASTRA-FT-HGR**
   - Fine-tuned with data from schools with high graduation rates
   - Specialized for high-performing school environments

2. **ASTRA-FT-LGR**
   - Fine-tuned with data from schools with low graduation rates
   - Focused on understanding challenges in struggling schools

3. **ASTRA-FT-Full**
   - Fine-tuned with a mixed dataset from both high and low graduation rate schools
   - Provides a balanced perspective across different school environments

## Usage Instructions

1. **Model Selection**
   - Choose one of the three fine-tuned models based on your analysis needs
   - Each model offers unique insights into student learning strategies

2. **Data Scope Configuration**
   - Select the percentage of schools to include in your analysis
   - Note: Larger percentages may require longer processing times

3. **Results Analysis**
   - Access the dashboard to view model results
   - Results are segregated by school graduation rates (high vs. low)
   - Analyze strategy predictions and their effectiveness

## Technical Details

The system is built on:
- A BERT-like architecture for the base model
- Fine-tuning using real-world student data from MATHia
- Specialized training sets based on school graduation rates

## Partners

- University of Memphis
- Carnegie Learning (MATHia Platform)

## Note

This is a demonstration version of the ASTRA system. The models are trained on specific mathematical domains and grade levels, with a current focus on ratio and proportions in 7th-grade mathematics.

## 📦 Dependencies

- Python 3.7+
- Required machine learning libraries
- Access to Hugging Face platform

## ⚡ Quick Start

1. Visit `suryadevi/astra` on Hugging Face
2. Select your desired analysis type
3. Adjust parameters as needed
4. Generate instant visualizations

---

Made with ❤️ by the ASTRA team


