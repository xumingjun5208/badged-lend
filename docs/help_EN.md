# 🏥 Stroke-Associated Pneumonia Prediction System User Guide

## 🎯 System Introduction

This system is an AI-assisted diagnostic tool that helps doctors assess the risk of stroke-associated pneumonia using deep learning technology. The system supports two types of stroke prediction tasks:

### Ischemic Stroke-Associated Pneumonia Prediction

- For ischemic stroke patients
- Based on clinical features and medical records
- Assesses the risk of associated pneumonia

### Hemorrhagic Stroke-Associated Pneumonia Prediction

- For hemorrhagic stroke patients
- Combines clinical indicators and medical information
- Predicts the risk of associated pneumonia

## 📝 Input Methods

### 1️⃣ Medical Record Text Input

> Enter patient medical record information in the left text box

#### 📋 Content Requirements

- **Chief Complaint**: Time of onset, main symptoms
- **Present Illness**: Disease progression process
- **Physical Examination**: Signs, neurological examination results

#### ⚡ Quick Use

- Use example buttons to quickly fill in sample text
- Supports copy and paste functionality
- Recommended text length: 100-1000 characters

### 2️⃣ Clinical Indicator Input

> Enter relevant clinical indicators on the right

#### 🔍 Ischemic Stroke Indicators

- **Dysphagia** `[Yes/No]`: Whether the patient has swallowing difficulties
- **Ventilator-Associated Pneumonia** `[Yes/No]`: Whether ventilator-associated pneumonia is present
- **Decubitus Ulcer** `[Yes/No]`: Whether bedsores are present
- **Age** `[Value]`: Patient's age
- **Resuscitation Count** `[Value]`: Number of resuscitations
- **Lung Disease** `[Yes/No]`: Whether lung disease is present

#### 🔍 Hemorrhagic Stroke Indicators

- **Dysphagia** `[Yes/No]`: Whether the patient has swallowing difficulties
- **Ventilator-Associated Pneumonia** `[Yes/No]`: Whether ventilator-associated pneumonia is present
- **Decubitus Ulcer** `[Yes/No]`: Whether bedsores are present
- **Hydrocephalus** `[Yes/No]`: Whether hydrocephalus is present
- **Brain Herniation** `[Yes/No]`: Whether brain herniation is present
- **Leukocytosis** `[Yes/No]`: Whether leukocytosis is present
- **Gastrointestinal Bleeding** `[Yes/No]`: Whether GI bleeding is present

## 📊 Understanding Prediction Results

### 1️⃣ Risk Prediction

> Displays predicted risk level and specific probability

Ischemic Stroke:

- 🔴 **High Risk**: Probability > 20%
  - Note: Close monitoring needed, take preventive measures promptly
- 🟢 **Low Risk**: Probability ≤ 20%
  - Note: Relatively low risk, continue routine monitoring

Hemorrhagic Stroke:

- 🔴 **High Risk**: Probability > 30%
  - Note: Close monitoring needed, take preventive measures promptly
- 🟢 **Low Risk**: Probability ≤ 30%
  - Note: Relatively low risk, continue routine monitoring

### 2️⃣ Text Analysis

> Analyzes the importance of input text

Color Marking Guide:

- 🟥 **Deep Red**: Extremely high correlation clinical information
- 🟥 **Red**: Highly correlated clinical information
- 🟨 **Yellow**: Moderately correlated clinical information
- ⬜ **Gray**: Slightly correlated clinical information

### 3️⃣ Feature Importance Analysis

> Shows the contribution of each factor to the prediction result

- 📈 Red Area: Risk-increasing factors
- 📉 Blue Area: Risk-decreasing factors
- Bar Length: Magnitude of impact

## ⚠️ Important Notes

1. 💡 **Data Quality**

   - Ensure accuracy of input information
   - Provide complete medical record information when possible

2. 🔄 **Usage Modes**

   - Can use text input alone
   - Can use clinical indicators alone
   - Best results achieved using both inputs

3. 📌 **Result Interpretation**
   - Predictions are for reference only
   - Follow medical advice for specific treatment plans
   - Regularly assess patient condition

## 🆘 Technical Support

If you encounter problems or need help:

- 📧 Contact Email: liaojun@cpu.edu.cn
- 🏛️ Institution: School of Science, China Pharmaceutical University
- 🔬 Laboratory: Laboratory of Medical Big Data and Artificial Intelligence
- ⏰ Working Hours: Monday to Friday 9:00-18:00 

## 💻 Source Code

- 📦 GitHub Repository: [badged-lend](https://github.com/xumingjun5208/badged-lend)
- 🔄 Latest Version: v1.0
- 📄 License: MIT License
