# Stroke-Associated Pneumonia Prediction System

## Project Overview

This system is an AI-powered diagnostic assistant system designed to predict the risk of stroke-associated pneumonia in stroke patients. It supports prediction tasks for both ischemic and hemorrhagic stroke types by integrating medical record text and clinical indicators to provide decision support for doctors.

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
     - Click to view detailed information

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

### Environment Setup

1. System Requirements

   - OS: Linux/Windows
   - Python Version: 3.8+
   - CUDA Version: 11.0+ (GPU version)

2. Dependencies Installation
   ```bash
   pip install -r requirements.txt
   ```

3. Model Files
   - Download pre-trained models
   - Place in specified directory
   - Check file permissions

### Starting Service

1. Development Mode
   ```bash
   python dev.py
   ```

2. Production Mode
   ```bash
   python app.py
   ```

3. Access Address
   - Local access: http://localhost:7860
   - Default port: 7860

## Maintenance Guide

### Logging System

- Runtime Logs
  - Records prediction requests
  - Error tracking
  - Performance monitoring
- Error Handling
  - Input validation
  - Exception catching
  - User-friendly prompts

### Performance Optimization

- Model Optimization
  - Model preloading
  - Batch processing
  - Memory management
- Concurrent Processing
  - Request queue
  - Concurrency limits
  - Load balancing

### Version Control

- Code Version
  - Git management
  - Branch strategy
  - Release process
- Model Version
  - Version records
  - Compatibility check
  - Update mechanism

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

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU版本)
- 4GB+ RAM

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/your-username/badged-lendcare.git
cd badged-lendcare
```

2. 创建并激活conda环境
```bash
conda create -n SAP python=3.8
conda activate SAP
```

3. 安装PyTorch (带CUDA支持)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. 安装基础依赖
```bash
conda install numpy pandas scikit-learn matplotlib seaborn
conda install -c conda-forge transformers tokenizers jieba
conda install -c conda-forge shap lightgbm xgboost
pip install gradio==4.44.1
```

5. 安装其他依赖
```bash
pip install -r requirements.txt
```

6. 下载模型文件
```bash
# 安装huggingface_hub
pip install huggingface_hub

# 运行下载脚本
python scripts/download_models.py
```

7. 启动应用
```bash
python app.py
```

访问 http://localhost:7860 即可使用系统