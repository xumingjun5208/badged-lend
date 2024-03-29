# coding: UTF-8

import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import shap
import warnings
import matplotlib as plt
import argparse
warnings.filterwarnings("ignore")

# prefix = "C:\\Users\\Alienware\\Desktop\\demo\\py\\"
# prefix_image = prefix + "images\\"

prefix = "/root/py-linux/"
prefix_image = prefix + "images/"

parser = argparse.ArgumentParser(
        prog='WriteJSON',
        description='Writing the input json data to the corresponding .xls template file and save'
    )

parser.add_argument("--Dysphagia1", type=int, default=0)
parser.add_argument("--Endotracheal_intubation1", type=int, default=0)
parser.add_argument("--Decubitus_ulcer1", type=int, default=0)
parser.add_argument("--Age1", type=int, default=0)
parser.add_argument("--Number_of_resuscitations1", type=int, default=0)
parser.add_argument("--Lung_disease1", type=int, default=0)
args = parser.parse_args()
Dysphagia = (args.Dysphagia1-0.1078)/0.0962
Endotracheal_intubation = (args.Endotracheal_intubation1-0.0166)/0.0163
Decubitus_ulcer = (args.Decubitus_ulcer1-0.0132)/0.013
Age = (args.Age1-68.856)/132.063
Number_of_resuscitations = (args.Number_of_resuscitations1-0.0653)/0.0611
Lung_disease = (args.Lung_disease1-0.3595)/0.2303

# 定义输入变量
#情况二：只输入数值
input_data = [[Endotracheal_intubation, Lung_disease,Dysphagia,  Number_of_resuscitations, Age, Decubitus_ulcer]]
input_data_shap=input_data_shap = [[args.Endotracheal_intubation1, args.Lung_disease1, args.Dysphagia1, args.Number_of_resuscitations1,args.Age1, args.Decubitus_ulcer1]]
# 加载模型文件
# with open('C:\\java\\project\\plantform\\src\\main\\resources\\python\\ais_SoftVoting_6_mice1.pkl', 'rb') as file:
# with open('/data/python/ais_SoftVoting_6_mice1.pkl', 'rb') as file:
# with open('./ais_SoftVoting_6_mice1.pkl', 'rb') as file:
with open(prefix + 'ais_SoftVoting_6_mice1.pkl', 'rb') as file:
    model = pickle.load(file)
# 定义特征名
input_names = ['Endotracheal intubation','Lung disease', 'Dysphagia', 'Number of resuscitations', 'Age', 'Decubitus ulcer']
yuzhi=0.2
#根据input数量选择model
prob = model.predict_proba(input_data)[:, 1][0]
# 输出预测结果
if prob > yuzhi:
    print('该患者在入院后七天内可能会发生卒中相关肺炎,发生风险为{:.1%}'.format(prob))
else:
    print('该患者不太可能会发生卒中相关肺炎,发生风险为{:.1%}'.format(prob))
data1=pd.DataFrame(input_data,columns=input_names)
data2=pd.DataFrame(input_data_shap,columns=input_names)
explainer = shap.TreeExplainer(model.estimators_[0])
shap_values = explainer.shap_values(data1.iloc[0,:])
shap.initjs()
p=shap.force_plot(base_value=0, shap_values=shap_values[0],  show=False,features=data2.iloc[0,:],feature_names=input_names, matplotlib=True)
p.savefig(prefix_image + 'example_plot.png')
