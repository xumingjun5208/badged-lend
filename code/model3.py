# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizerFast
import torch.nn.functional as F
import re
import pandas as pd
import numpy as np
import pickle
import shap
import argparse
import warnings

prefix = "/root/py-linux/"
prefix_image = prefix + "images/"
pathd = prefix + "mac_bert/"
warnings.filterwarnings("ignore")
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
parser.add_argument("--line", type=str, default="")
args = parser.parse_args()
line = args.line
line=line.replace(",","，")
Dysphagia0 = (args.Dysphagia-0.0778)/0.0717
Endotracheal_intubation0 = (args.Endotracheal_intubation-0.2053)/0.1631
Decubitus_ulcer0 = (args.Decubitus_ulcer-0.028)/0.02723
Hydrocephalus0 = (args.Hydrocephalus-0.3219)/0.2183
Brain_hernia0 = (args.Brain_hernia-0.3114)/0.2144
Hyperleukocytosis0 = (args.Hyperleukocytosis-0.1177)/0.10383
Gastrointestinal_bleeding0 = (args.Gastrointestinal_bleeding-0.1116)/0.09914

#构造模型结构
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(pathd)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, 1)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        output= self.bert(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'],output_attentions=True )
        out=self.fc(output[1])
        out1=self.sig(out)
        return out1,output
class bertcnn(nn.Module):

    def __init__(self, path):
        super(bertcnn, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.dropout = 0.1
        self.num_classes=1 # 类别数
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden_size )) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)

        self.fc_cnn = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        self.sig = nn.Sigmoid()
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        #context = x[0]  # 输入的句子
        #mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        outputs= self.bert(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'],output_attentions=True)
        encoder_out=outputs[0]
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        out1 = self.sig(out)
        return out1,outputs
# 加载模型文件
def get_seed(seed=15):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

#get_seed(11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(pathd)
model = bertcnn(pathd)
with open(prefix + 'ich_SoftVoting_8_mice1.pkl', 'rb') as file:
    model3 = pickle.load(file)
model.load_state_dict(torch.load(prefix + 'ich_baseline_macbertnewend1cnn_1time_epoch3.pth', map_location=device)['net'])
model.eval()
model.to(device)
lines=[line]
colors=[26,39,123,159,231]#颜色列表
all_list=[]
#colors=[34,82,83,155,231]
all_char=''
# 使用模型进行预测
def label(score):
    split = np.percentile(scores, q=(20, 30, 40, 60))
    if score <= split[0]:
        return -1
    elif score <= split[1]:
        return -2
    elif score <= split[2]:
        return -3
    elif score <= split[3]:
        return -4
    else:
        return -5


for line in lines:
    line = re.sub(r"[*]|[\s]|[\\r]|[\\n]|[\\t]|[\\f]|[\\v]", '', line)
    inputs = tokenizer(line, max_length=512, return_offsets_mapping=True, add_special_tokens=True, truncation=True,
                       padding=True,
                       return_tensors="pt")
    # inputs=tokenizer.encode_plus(line,return_offsets_mapping=True,add_special_tokens=False,return_tensors='pt')
    # tokens=tokenizer.tokenize(line)
    token_span = inputs['offset_mapping'][0]
    inputs.to(device)
    out, outputs = model(inputs)
    attscore = outputs['attentions']
    interscore = attscore[-1][:, :, 0, :].mean(1)  # 取最后一层bert的平均注意力分数
    scores = interscore.detach().cpu().numpy()
    scores = scores[0]
    token_span = token_span.detach().cpu().numpy()

    charlist = []
    linelist = list(line)
    for rg in token_span:  # 将Token和offsets对应起来
        charlist.append(line[rg[0]:rg[1]])

    assert len(charlist) == len(scores)

    all_char = ''
    for char, score in zip(charlist, scores):
        colorlabel = colors[label(score)]
        tem = "\033[48;5;{}m".format(colorlabel) + char
        all_char += tem
    all_list.append(all_char)
    admission_score=out.detach().numpy()
for ll in all_list:
    print(ll)
    print('\n')

# 定义特征名
input_data = [[(round(admission_score.item(),2)-0.2666)/0.0464, Brain_hernia0,Decubitus_ulcer0, Dysphagia0, Endotracheal_intubation0, Gastrointestinal_bleeding0,  Hydrocephalus0, Hyperleukocytosis0]]
input_data_shap = [[round(admission_score.item(),2),args.Brain_hernia,args.Decubitus_ulcer,  args.Dysphagia, args.Endotracheal_intubation, args.Gastrointestinal_bleeding, args.Hydrocephalus, args.Hyperleukocytosis]]
input_names = ['Admission score', 'Brain hernia', 'Decubitus ulcer','Dysphagia', 'Endotracheal intubation', 'Gastrointestinal bleeding', 'Hydrocephalus', 'Hyperleukocytosis']
yuzhi=0.3
prob = model3.predict_proba(input_data)[:, 1][0]
# 输出预测结果
if prob > yuzhi:
    print('The patient may develop stroke-associated pneumonia within seven days of admission, with a risk of occurrence at {:.1%}'.format(prob))
else:
    print('The patient has a low risk of developing stroke-related pneumonia, with a risk of occurrence at {:.1%}'.format(prob))
data1=pd.DataFrame(input_data,columns=input_names)
data2=pd.DataFrame(input_data_shap,columns=input_names)
explainer = shap.TreeExplainer(model3.estimators_[0])
shap_values = explainer.shap_values(data1.iloc[0,:])
shap.initjs()
p=shap.force_plot(base_value=0, shap_values=shap_values[0],  show=False,features=data2.iloc[0,:],feature_names=input_names, matplotlib=True)
p.savefig(prefix_image + 'example_plot.png')

