# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizerFast
import torch.nn.functional as F
import numpy as np
import warnings
import re
import argparse
from PIL import Image

prefix = "/root/py-linux/"
prefix_image = prefix + "images/"

pathd = prefix + "mac_bert/"
warnings.filterwarnings("ignore")
img = Image.new('RGB', (200, 300), 'white')
img.save(prefix_image + 'example_plot.png')
parser = argparse.ArgumentParser(
        prog='WriteJSON',
        description='Writing the input json data to the corresponding .xls template file and save'
    )
parser.add_argument("--line1", type=str, default="")
args = parser.parse_args()
line = args.line1
line=line.replace(",","，")
# 构造输入数据
line1='入院时情况:(简要病史、阳性体征、有关实验室及器械检查结果)患者，因"右侧肢体乏力15天"入院。患者于2020-12-02日16时左右打牌时弯腰捡东西后出现头痛，伴右侧肢体乏力，伴言语含糊，无头晕、呕吐，无意识障碍、肢体抽搐，无大小便失禁，被家人送至查颅脑CT示脑出血，家人将其转至住院治疗，测血压222/131mmHg，右侧肢体肌力3级，查颅脑CT示左侧基底节区出血，予止血、营养神经、降压等治疗。12-08患者出现右下肢红肿、疼痛，并出现昏睡、不能言语、进食呛咳，右侧肢体乏力加重，复查颅脑CT示脑出血较前相仿。12-13患者出现左上肢抽搐，好发于夜间，每次发作约1分钟，且进食量少、尿量减少，大便未解，复查颅脑CT示脑出血较前吸收。患者家人要求转我院治疗，2020-12-17家属将其送至我院急诊抢救室，查电解质示:钠167.7mmol/L，氯132.4mmol/L，肌酐322.9umol/L。予头孢唑肘抗感染、营养神经、保肝、营养支持、维持水电解质平衡等治疗，为进一步诊治收住我科。既往有脑出血病史12年，于当地医院住院治疗，未留有后遗症，在住院期间发现高血压及肾功能不全（肌酐在100-208umo1/L，后复查恢复正常)，血压最高达190/120mmHg，不规则口服缴沙坦非洛地平治疗，未监测血压及肾功能;有痛风病史6年;有长期吸烟史。体格检查:体温36.8℃C，脉搏80次/分，呼吸22次/分，血压163/93mHg。神志嗜睡，精神一般，鼻饲管在位，颈软，双瞳孔d=3.0mm，直接、间接对光反射灵敏，心率80次/分，律齐，未及杂音;两肺呼吸音粗，双肺未闻及干湿性啰音，双下肢不肿，右侧上下肢力1级，肌张力正常，右侧巴宾斯基征阳性。'

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

model.load_state_dict(torch.load(prefix + 'ais_baseline_macbertnewend2cnn_3_1time_epoch3.pth', map_location=device)['net'])
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
    prob=out.detach().numpy()
for ll in all_list:
    print(ll)
    print('\n')
yuzhi=0.2
# 输出预测结果
if prob > yuzhi:
    print('The patient may develop stroke-associated pneumonia within seven days of admission, with a risk of occurrence at {:.1%}'.format(prob.item()))
else:
    print('The patient has a low risk of developing stroke-related pneumonia, with a risk of occurrence at {:.1%}'.format(prob.item()))
