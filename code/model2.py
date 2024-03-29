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

parser.add_argument("--Dysphagia", type=int, default=0)
parser.add_argument("--Endotracheal_intubation", type=int, default=0)
parser.add_argument("--Decubitus_ulcer", type=int, default=0)
parser.add_argument("--Hydrocephalus", type=int, default=0)
parser.add_argument("--Brain_hernia", type=int, default=0)
parser.add_argument("--Hyperleukocytosis", type=int, default=0)
parser.add_argument("--Gastrointestinal_bleeding", type=int, default=0)
args = parser.parse_args()


# 定义输入变量
#情况二：只输入数值
Dysphagia0 = (args.Dysphagia-0.0778)/0.0717
Endotracheal_intubation0 = (args.Endotracheal_intubation-0.2053)/0.1631
Decubitus_ulcer0 = (args.Decubitus_ulcer-0.028)/0.02723
Hydrocephalus0 = (args.Hydrocephalus-0.3219)/0.2183
Brain_hernia0 = (args.Brain_hernia-0.3114)/0.2144
Hyperleukocytosis0 = (args.Hyperleukocytosis-0.1177)/0.10383
Gastrointestinal_bleeding0 = (args.Gastrointestinal_bleeding-0.1116)/0.09914


# 加载模型文件
with open(prefix + 'ich_SoftVoting_7_mice1.pkl', 'rb') as file:
    model = pickle.load(file)
# 定义特征
input_data = [[Brain_hernia0, Decubitus_ulcer0,Dysphagia0, Endotracheal_intubation0, Gastrointestinal_bleeding0, Hydrocephalus0, Hyperleukocytosis0]]
input_data_shap = [[args.Brain_hernia, args.Decubitus_ulcer,args.Dysphagia, args.Endotracheal_intubation, args.Gastrointestinal_bleeding, args.Hydrocephalus, args.Hyperleukocytosis]]
input_names = [ 'Brain hernia', 'Decubitus ulcer','Dysphagia', 'Endotracheal intubation', 'Gastrointestinal bleeding', 'Hydrocephalus', 'Hyperleukocytosis']
yuzhi=0.3
#根据input数量选择model
prob = model.predict_proba(input_data)[:, 1][0]
# 输出预测结果
if prob > 0.3:
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
