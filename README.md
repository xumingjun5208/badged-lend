# Stroke-Associated Pneumonia Prediction System
[English](README.md) | [简体中文](README_CN.md)
## Project Overview

This system is an AI-powered diagnostic assistant system designed to predict the risk of stroke-associated pneumonia in stroke patients. It supports prediction tasks for both ischemic and hemorrhagic stroke types by integrating medical record text and clinical indicators to provide decision support for doctors.
![](https://github.com/user-attachments/assets/465adf14-6cb9-41e4-b56b-776c077f2e25)

## Online Demo
[User Guide](docs/help_EN.md)
Try our online demo at: [http://www.badged-lend.com](http://www.badged-lend.com)
![](https://github.com/user-attachments/assets/fa8bf2b1-27ad-4e43-a296-dd64bcfca379)

## System Architecture

### Model Components

1. Text Prediction Models

   - Ischemic Stroke Text Model (ischemic_text_model.py)
     - Based on MacBERT-CNN architecture
     - Processes Chinese medical records
     - Outputs risk probability and text importance analysis
   - Hemorrhagic Stroke Text Model (hemorrhagic_text_model.py)
     - Same model architecture
     - Optimized for hemorrhagic stroke characteristics
     - Provides text analysis visualization

2. Structured Prediction Models

   - Ischemic Stroke Structured Model (ischemic_structured_model.py)
     - Based on soft voting ensemble learning
     - Processes 6 key clinical indicators
     - Binary classification with 0.2 threshold
   - Hemorrhagic Stroke Structured Model (hemorrhagic_structured_model.py)
     - Also based on ensemble learning
     - Processes 7 clinical indicators
     - Binary classification with 0.3 threshold

3. Combined Prediction Models
   - Ischemic Stroke Combined Model (ischemic_combined_model.py)
     - Integrates text and structured features
     - Text prediction probability as additional feature
     - SHAP values for prediction explanation
   - Hemorrhagic Stroke Combined Model (hemorrhagic_combined_model.py)
     - Similar fusion strategy
     - Targeted feature combination
     - Feature importance visualization

### Feature Description

1. Ischemic Stroke Features

   - Basic Information
     - Age: Continuous value
     - Number of resuscitations: Integer value
   - Clinical Manifestations
     - Dysphagia: Yes/No
     - Ventilator-associated pneumonia: Yes/No
     - Decubitus ulcer: Yes/No
     - Lung disease: Yes/No

2. Hemorrhagic Stroke Features
   - Basic Symptoms
     - Dysphagia: Yes/No
     - Ventilator-associated pneumonia: Yes/No
     - Decubitus ulcer: Yes/No
   - Special Indicators
     - Hydrocephalus: Yes/No
     - Brain hernia: Yes/No
     - Hyperleukocytosis: Yes/No
     - Gastrointestinal bleeding: Yes/No

### Prediction Logic

1. Single Mode Prediction

   - Text Input Only
     - Uses BERT for text feature extraction
     - CNN for feature fusion
     - Outputs risk probability and important text annotation
   - Indicators Only
     - Normalizes clinical indicators
     - Ensemble model prediction
     - SHAP value analysis for feature contribution

2. Combined Mode Prediction
   - Text Processing Flow
     - BERT-CNN extracts text features
     - Gets text prediction probability
   - Feature Fusion
     - Text prediction as new feature
     - Combined with clinical indicators
   - Final Prediction
     - Ensemble model comprehensive prediction
     - Generates SHAP explanation plot
     - Outputs risk assessment results

### Output Results

1. Risk Prediction

   - Ischemic Stroke
     - High Risk: Probability > 20%
     - Low Risk: Probability ≤ 20%
   - Hemorrhagic Stroke
     - High Risk: Probability > 30%
     - Low Risk: Probability ≤ 30%

2. Text Analysis

   - Importance Levels
     - Extremely High: Deep red marking
     - High: Red marking
     - Moderate: Yellow marking
     - Low: Gray marking
   - Interactive Features
     - Hover to show importance value

3. Feature Importance
   - SHAP Value Analysis
     - Red: Risk-increasing factors
     - Blue: Risk-decreasing factors
   - Visualization
     - Waterfall plot shows contributions
     - Annotates specific impact values

## Technical Implementation

### Frontend Interface

- Gradio Framework
  - Responsive layout
  - Dark mode support
  - Custom CSS styles
  - Interactive components
  - Bilingual support (Chinese/English)

### Backend Technology

- Deep Learning Framework
  - Python 3.8+
  - PyTorch 1.8+
  - Transformers 4.0+
- Machine Learning Tools
  - Scikit-learn
  - SHAP 0.40+
- Utility Libraries
  - Numpy
  - Pandas
  - Matplotlib

### System Features

- Task Switching
  - Ischemic/Hemorrhagic stroke selection
  - Dynamic interface update
- Data Input
  - Text input box
  - Clinical indicator selection
  - Example data filling
- Result Display
  - Risk prediction results
  - Text analysis visualization
  - SHAP value explanation plot

## Deployment Guide

### Requirements

- Python 3.8+
- CUDA 11.0+ (GPU version)
- 4GB+ RAM

### Installation

1. Clone repository

```bash
git clone https://github.com/your-username/badged-lendcare.git
cd badged-lendcare
```

2. Create and activate conda environment

```bash
conda create -n SAP python=3.8
conda activate SAP
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Download model files

Option 1: Automatic download
```bash
# Install huggingface_hub
pip install huggingface_hub

# Run download script
python scripts/download_models.py
```

Option 2: Manual download
Download model files from:
- HuggingFace repository: 1.[donghao1234/badged-lend](https://huggingface.co/donghao1234/badged-lend)
2.[hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)
Model file structure:
```
models/
├── ischemic/
│   ├── text/
│   │   └── ais_baseline_macbertnewend2cnn_3_1time_epoch3.pth
│   ├── structured/
│   │   └── ais_SoftVoting_6_mice1.pkl
│   └── combined/
│       └── ais_SoftVoting_7_mice1.pkl
├── hemorrhagic/
│   ├── text/
│   │   └── ich_baseline_macbertnewend1cnn_1time_epoch3.pth
│   ├── structured/
│   │   └── ich_SoftVoting_7_mice1.pkl
│   └── combined/
│       └── ich_SoftVoting_8_mice1.pkl
└── macbert/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    └── tokenizer_config.json
```

5. Start application

```bash
python app.py
```

6. Access Address
   - Local access: http://localhost:8080
   - Default port: 7860

## Development Team

### Research Institution

Medical Big Data and Artificial Intelligence Laboratory, School of Science, China Pharmaceutical University

### Contact Information

- 📧 Email: liaojun@cpu.edu.cn
- 🏛️ Address: School of Science, China Pharmaceutical University
- 🔬 Laboratory: Medical Big Data and Artificial Intelligence Laboratory
- ⏰ Working Hours: Monday to Friday 9:00-18:00

## License

[Apache 2.0](LICENSE)

## ICP Filing

- [苏公网安备 32010602011293号](https://beian.mps.gov.cn/#/query/webSearch)
- [苏ICP备2023023603](https://beian.miit.gov.cn/#/Integrated/index)
