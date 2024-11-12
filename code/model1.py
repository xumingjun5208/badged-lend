# coding: UTF-8
from transformers import AdamW
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizerFast
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import *
import torch.nn.functional as F
import time
import os
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
parser.add_argument("--line", type=str, default="")
args = parser.parse_args()
line = args.line
line=line.replace(",","，")
# 构造输入数据
#line="入院时情况:(简要病史、阳性体征、    有关实验室及***器械检查结果)患者，因'右侧肢体*乏力15天'入院。患者于2020-12-02日16时左右打牌时弯腰捡东西后出现头痛，伴右侧肢体乏力，伴言语含糊，无头晕、呕吐，无意识障碍、肢体抽搐，无大小便失禁，被家人送至查颅脑CT示脑出血，家人将其转至住院治疗，测血压222/131mmHg，右侧肢体肌力3级，查颅脑CT示左侧基底节区出血，予止血、营养神经、降压等治疗。12-08患者出现右下肢红肿、疼痛，并出现昏睡、不能言语、进食呛咳，右侧肢体乏力加重，复查颅脑CT示脑出血较前相仿。12-13患者出现左上肢抽搐，好发于夜间，每次发作约1分钟，且进食量少、尿量减少，大便未解，复查颅脑CT示脑出血较前吸收。患者家人要求转我院治疗，2020-12-17家属将其送至我院急诊抢救室，查电解质示:钠167.7mmol/L，氯132.4mmol/L，肌酐322.9umol/L。予头孢唑肘抗感染、营养神经、保肝、营养支持、维持水电解质平衡等治疗，为进一步诊治收住我科。既往有脑出血病史12年，于当地医院住院治疗，未留有后遗症，在住院期间发现高血压及肾功能不全（肌酐在100-208umo1/L，后复查恢复正常)，血压最高达190/120mmHg，不规则口服缴沙坦非洛地平治疗，未监测血压及肾功能;有痛风病史6年;有长期吸烟史。体格检查:体温36.8℃C，脉搏80次/分，呼吸22次/分，血压163/93mHg。神志嗜睡，精神一般，鼻饲管在位，颈软，双瞳孔d=3.0mm，直接、间接对光反射灵敏，心率80次/分，律齐，未及杂音;两肺呼吸音粗，双肺未闻及干湿性啰音，双下肢不肿，右侧上下肢力1级，肌张力正常，右侧巴宾斯基征阳性。"
#line='患者因“突发意识不清7小时余”入院。查体：体温37.1℃，脉搏80次/分，呼吸20次/分，血压158/91mmHg，头颅大小正常，昏迷，刺痛无法睁眼，查体不合作，双瞳等大，直径约3mm，对光反射迟钝，颈软，气管插管在位，双肺呼吸音粗，可闻及干湿性啰音,诊断为肺炎。心界不大，心律齐，未闻及器质性杂音。腹平软，无明显肌卫，移动性浊音检查不合作，肠鸣音1分钟未闻及，脊柱四肢无畸形，四肢刺痛无自主活动，肌力检查不合作，肌张力不高，生理反射弱，病理反射未引出。'
#line='患者因“突发头痛头晕一小时余”入院，体格检查:神志清晰，精神欠佳，GCS14分，E3V5M6，头颅无畸形，双侧瞳孔等大等圆,直径为2.5mm，对光反射灵敏，口鼻腔及双外耳无活动性流液，伸舌居中，口角无歪斜，颈软，无抵抗感，四肢肌力5级，无感觉异常。膝跳反射，跟腱反射正常，双侧巴氏征、克氏征未引出。辅助检查：CT：左侧枕叶可见团块状高密度影，两侧半卵圆区可见点状低密度影，两侧侧脑室周围脑白质内见片状低密度影，脑沟增宽，脑池扩大，中线结构无移位。'
#line='患者，男性，64岁，因“行走不稳伴记忆力减退一月余”入院，患者半年余前有不慎摔倒致头部外伤史，患者一月余前出现行走不稳，伴记亿力减退，精神轻度萎靡，偶感头痛，患者遂至外院就诊，行头CT(2018-08-06，外院)示:左侧额颞顶枕部可见新月形等高密度影，左侧侧脑室受压，中线轻微右移。余脑沟、脑裂未见明显异常。现患者及其家属为求进一步诊治，遂来我院就诊，门诊拟“左侧额颓顶枕部慢性硬膜下血肿”收住入院。病程中，患者神志清，精神轻度萎靡，食纳、睡眠一般，大小便正常。患者既往有高血压病史5年余，口服施慧达1#QD、撷沙坦1#QD控制，控制效果一般。无肝炎结核等传染病史。无外伤手术史。无输血史。无食物、药物过敏。查体：神志清，对答切题，双瞳等大等圆，直径2.5mm，对光反射灵敏，伸舌居中，口角无偏斜，颈软无抵抗，心肺听诊未及明显异常，腹部平软，无明显压痛及反跳痛，双下肢肌力V-，双上肢肌力正常，肌张力正常，生理反射存在，病理反射未引出。辅助检查∶头颅CT(2018-08-06，外院)示∶左侧额颞顶枕部可见新月形等高密度影，左侧侧脑室受压，中线轻微右移。余脑沟、脑裂未见明显异常。'
#line='患者，男性，57岁，因“右侧肢体活动障碍伴麻木4年余，加重一月”入院。查体：T：BP：122/76mmHg，神志清楚，精神萎，查体合作，言语清晰，回答切题。胸廓对称，双肺呼吸音清，未闻及干湿罗音。心率69次/分，律齐，各瓣膜听诊区未闻及明显病理性杂音。腹软，未见肠型及蠕动波，无压痛、反跳痛，肝脾肋下未及，未及明显包块;肝肾区无叩击痛，移动性浊音阴性，肠鸣音正常。双下肢无水肿。专科检查：交流、理解能力稍下降。嗅觉、视野、视力粗测无明显异常，双侧瞳孔等大等圆，直径约3mm，对光反射灵敏，眼球运动自如，无明显眼震，双侧眼裂、额纹对称，双侧鼻唇沟对称，口角不歪，粗查右耳听力正常，双侧软腭上抬对等，悬壅垂居中，咽反射正常，双侧转颈，耸肩对等有力，伸舌偏右，无舌肌萎缩、舌肌震颤。右侧肌张力略偏高，肌力5级，腱反射()。双下肢病理征阴性。右侧肢体针刺觉较对侧减退，四肢深感觉正常。右侧指鼻试验、轮替试验、跟-膝-胫试验欠稳准，左侧共济试验正常。颈软，Keig征、Budziski征阴性。辅助检查，头颅CT平扫示：1、左额顶叶软化灶并左侧脑室穿通畸形，2、左蝶窦粘膜下囊肿。心电图示窦性心律，一度房室传导阻滞，ST段异常(抬高)。'
#line='患者因突发头晕一天余入院。患者昨日中午12点务午饭后准备上床体息，突然发作头是，天旅地转，伴四肢无力，瘫倒在地，并恶心呕吐，呕吐物为胃内容物，无意识不清及肢体抽搐，急诊送入我院，查头颅CT示多发腔隙性脑梗死，给予银杏达莫、奥美拉唑等药物治疗后头是有所好转，但患者诉向左侧转头时可诱发眩是，今日为进一步治疗收住我科。病程中无畏寒发热，无咳嗽咳痰，无胸闷心慌，无腹痛腹泻，纳食睡眠可，大小便正常，近期体重无明显变化。平素时有左耳耳鸣，似火车暴鸣声，否认有听力异常。既往千余年前有过发作头量，但症状相对较轻。有高血压病史20余年，口服拜新同降压，自诉血压控制在130/80mmhg左岩；否认外伤史，无输血史；否认烟酒嗜好；有善霉素 链霉素及前列地尔过敏史，体人史、家族史无特殊。查体：体温：36.2℃,脉搏：81次/分，呼吸：20次/分，血压：175/92mmHg,体重指数：63kg/m2。神志清楚，查体合作，皮肤、巩膜无明显黄染，浅表淋巴结未及肿大，颈软，写管居中，两侧甲状腺未及肿大，颈部血管未闻及杂音，两肺呼吸音清，未闻及干湿啰音。心音有力，心率81次/分，律齐，各瓣膜听诊区未闻及明显病理性杂音。腹平软，肝脾肋下未及，移动性浊音阳性，肠鸣音正常，双下肢无水肿，双下肢拾高试验(-)。NE:神志清楚，口齿清，双侧瞳孔等大等圆直径2.5mm,对光反射灵敏，眼球运动正常，无水平眼霄，用头试验阳性，双侧额纹、晶唇沟对称，伸舌居中，双侧咽反射存在，四肢肌张力正常，肌力5级，四肢腱反射正常，双侧巴氏征未引出。双侧指鼻试验准确，双侧跟膝胫试验(二),闭目难立征不能配合，深浅感觉检查正常，颜软，吉民征’布氏征(-)。检查：头颅CT(南京市鼓楼医院，2019-01-16):1.两次放射冠腔隙性脑梗死2.脑萎缩，两次侧脑室旁白质慢性缺氧性改变3.若侧蝶实轻度炎症。'
#line='1.患者昨日始无明显诱因下出现头晕伴右下肢无力，在当地卫生室输液治疗，病情无好转，今入本院就诊，头颅CT示多发脑梗死（部分陈旧性)、脑萎缩，急诊拟"脑梗死"收住院治疗。病程中患者稍气喘，无昏迷抽搐，无恶心呕吐，无呕血，无复视耳鸣，无发热畏寒，无心慌胸闷胸痛，无明显口干多饮多尿，无腹痛腹泻等。目前患者食纳夜眠正常，无二便失禁。2.既往患有高血压、糖尿病史十年余，一直药物控制，控制效果一般，2年前有青光眼病史，行手术治疗，现双眼视力明显减退，只有光感，否认食物及药物过敏史。有慢性支气管炎、肺气肿、肺心病病史数十年，曾因肺性脑病在本院重症监护室住院治疗好转出院。3.查体;T36.2℃,P84次分，R18次分，BP16090mmHg，神志清楚，精神差，腹型肥胖，推车入病房，查体合作，轻度构音障碍，无明显失语，双侧额纹对称，双眼球活动自如，双瞳等大等圆，直径约3mm，对光反射灵敏，右侧鼻唇沟变浅，伸舌偏右，咽反射存在，颈软，呼吸稍促，桶状胸，双肺呼吸音低，两肺可闻及干湿罗音，心率84次分，律齐，心音中等，各瓣膜区未闻及杂音。腹软，无压痛，肝脾肋下未及，肠鸣音正常，双下肢不肿，右上肢肌力4级，右下肢肌力3级，肌张力正常，左侧肢体肌力肌张力正常，双侧病理征(-)。4.辅助检查:头颅CT示两侧基底节区、放射冠区、半卵圆中心多发脑梗死，脑白质变性，脑萎缩;心电图示窦性心律。'
#line='入院时情况:(简要病史、阳性体征、有关实验室及器械检查结果)患者，因"右侧肢体乏力15天"入院。患者于2020-12-02日16时左右打牌时弯腰捡东西后出现头痛，伴右侧肢体乏力，伴言语含糊，无头晕、呕吐，无意识障碍、肢体抽搐，无大小便失禁，被家人送至查颅脑CT示脑出血，家人将其转至住院治疗，测血压222/131mmHg，右侧肢体肌力3级，查颅脑CT示左侧基底节区出血，予止血、营养神经、降压等治疗。12-08患者出现右下肢红肿、疼痛，并出现昏睡、不能言语、进食呛咳，右侧肢体乏力加重，复查颅脑CT示脑出血较前相仿。12-13患者出现左上肢抽搐，好发于夜间，每次发作约1分钟，且进食量少、尿量减少，大便未解，复查颅脑CT示脑出血较前吸收。患者家人要求转我院治疗，2020-12-17家属将其送至我院急诊抢救室，查电解质示:钠167.7mmol/L，氯132.4mmol/L，肌酐322.9umol/L。予头孢唑肘抗感染、营养神经、保肝、营养支持、维持水电解质平衡等治疗，为进一步诊治收住我科。既往有脑出血病史12年，于当地医院住院治疗，未留有后遗症，在住院期间发现高血压及肾功能不全（肌酐在100-208umo1/L，后复查恢复正常)，血压最高达190/120mmHg，不规则口服缴沙坦非洛地平治疗，未监测血压及肾功能;有痛风病史6年;有长期吸烟史。体格检查:体温36.8℃C，脉搏80次/分，呼吸22次/分，血压163/93mHg。神志嗜睡，精神一般，鼻饲管在位，颈软，双瞳孔d=3.0mm，直接、间接对光反射灵敏，心率80次/分，律齐，未及杂音;两肺呼吸音粗，双肺未闻及干湿性啰音，双下肢不肿，右侧上下肢力1级，肌张力正常，右侧巴宾斯基征阳性。'


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
    prob=out.detach().numpy()
for ll in all_list:
    print(ll)
    print('\n')
yuzhi=0.3
# 输出预测结果
if prob > yuzhi:
    print('The patient may develop stroke-associated pneumonia within seven days of admission, with a risk of occurrence at {:.1%}'.format(prob.item()))
else:
    print('The patient has a low risk of developing stroke-related pneumonia, with a risk of occurrence at {:.1%}'.format(prob.item()))
