# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        #self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1


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

        outputs= self.bert(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'])
        encoder_out=outputs[0]
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        out1 = self.sig(out)
        return out1
