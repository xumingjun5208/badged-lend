import torch.nn as nn
from transformers import AdamW
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification,BertModel
from transformers import BertTokenizer,BertTokenizerFast
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import *
import torch.nn.functional as F
import time
import os
import warnings
import BertCNN
import BertLSTM
import BertCNN
import BertDPCNN
import BertRCNN
warnings.filterwarnings("ignore")
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if not os.path.exists('paperlog_sap/' + log_name):
        os.mkdir('paperlog_sap/' + log_name)

    with open('paperlog_sap/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__), 'r', encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    path = 'paperlog_sap/' + log_name
    with open(path + '/log1.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path
def log(text, path):
    with open(path + '/log1.txt', 'a', encoding='utf-8') as f:
        if "valid" in text:
            f.write('************{}***********'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write(text)
            f.write('\n')
        else:
            f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write(text)
            f.write('\n')
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
def main(name,pathd,time_c,path):
    # 设定随机数
    def get_seed(seed=15):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    get_seed(8)#11
    # 超参数
    hidden_dropout_prob = 0.3
    num_labels = 4
    learning_rate = 1e-5#学习率大于1e-4时会出现验证集全预测为0的情况
    weight_decay = 1e-2
    epochs = 5
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "sentiment/"
    max_len=512
    acc_step = 5
    # 定义 Dataset
    class SentimentDataset(Dataset):
        def __init__(self, data):
            self.dataset = data
            print(self.dataset.isnull().sum())

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # 根据 idx 分别找到 text 和 label
            text = self.dataset.loc[idx, "IN_HP_ILL_"]
            label = self.dataset.loc[idx, "sap7"]
            sample = {"text": text, "label": label}
            # 返回一个 dict
            return sample

    class Model(nn.Module):

        def __init__(self):
            super(Model, self).__init__()
            self.bert = BertModel.from_pretrained(pathd)
            for param in self.bert.parameters():
                param.requires_grad = True

            self.fc = nn.Linear(768, 1)



            self.sig=nn.Sigmoid()


        def forward(self, x):
            #context = x['input_ids']  # 输入的句子
            #mask = x['attention_mask']  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
            output= self.bert(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'] )
            out=self.fc(output[1])


            out1=self.sig(out)
            return out1
        def initialize(self):
            nn.init.xavier_normal_(self.fc.weight.data)
            #nn.init.kaiming_normal_(self.fc2.weight.data)
    class Focal_loss(nn.Module):
        def __init__(self, weight=torch.Tensor([[1.0],[1.0],[1.0]]).to(device), gamma=2):
            super(Focal_loss, self).__init__()
            self.gamma = gamma
            self.weight = weight
            self.soft=nn.Softmax(dim=1)
        def forward(self, preds, labels):
            '''
            :param preds: 还未softmax输出结果
            :param labels:真实值
            :return:
            '''
            preds = self.soft(preds)
            labels_onehot=F.one_hot(labels.long(),preds.shape[1])
            eps = 1e-7
            y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)
            target = labels_onehot.view(y_pred.size())  # B*C*H*W->B*C*(H*W)
            ce = -1 * torch.log(y_pred + eps) * target
            floss = torch.pow((1 - y_pred), self.gamma) * ce
            floss = torch.mul(floss, self.weight)
            floss = torch.sum(floss, dim=1)
            return torch.mean(floss)
    # 加载并划分数据集
    from sklearn.model_selection import train_test_split
    import pandas as pd
    #texttype = 'PRES_HX' if name == "macbertcnn1" else 'IN_HP_ILL_'
    #sentiment_train_set = SentimentDataset('sentiment/outhp_train_fufa1.csv')
    dataall=pd.read_csv('/home/lab/sap_link_inhp_66467_345395.csv')
    if "macbertnewend1" in name:
        pd_chuxue = dataall[dataall['Stroke Type'] == 1]
    else:
        pd_chuxue = dataall[dataall['Stroke Type'] == 2]
    datatrain=pd_chuxue[pd_chuxue['validation_organ']==0]
    datatrain.reset_index(drop=True,inplace=True)
    datatest=pd_chuxue[pd_chuxue['validation_organ']==1]
    datatest.reset_index(drop=True, inplace=True)
    dataoutvalid = pd_chuxue[pd_chuxue['validation_organ'] == 2]
    dataoutvalid.reset_index(drop=True, inplace=True)

    sentiment_train_set = SentimentDataset(datatrain)
    sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # 加载验证集
    #sentiment_valid_set = SentimentDataset('sentiment/outhp_valid_fufa1.csv')
    sentiment_valid_set = SentimentDataset(datatest)
    sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # 加载外部验证集
    entiment_test_set = SentimentDataset(dataoutvalid)
    sentiment_test_loader= DataLoader(entiment_test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # 定义 tokenizer，传入词汇表
    # tokenizer = BertTokenizer(vocab_file)

    # 加载模型
    '''
    config = BertConfig.from_pretrained("../../bert-base-chinese", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained("../../bert-base-chinese", config=config)
    '''
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

    tokenizer = BertTokenizerFast.from_pretrained(pathd)

    #model = BertForSequenceClassification.from_pretrained(pathd,num_labels=3)
    if "lstm" in name:
        model = BertLSTM.bertlstm(pathd)
    elif "cnn" in name:
        model = BertCNN.bertcnn(pathd)
    else:
        model = Model()
    #model = BertCNN.bertcnn(pathd)
    #model.initialize()
    model.to(device)

    # 定义优化器和损失函数
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    #criterion = nn.CrossEntropyLoss()
    #criterion=Focal_loss()
    criterion = nn.BCELoss()
    def Focal_loss_raw():
        def __init__(self,weight,gamma=2):
            super(Focal_loss,self).__init__()
            self.gamma=gamma
            self.weight=weight
        def forward(self,preds,labels):
            '''
            :param preds: softmax输出结果
            :param labels:真实值
            :return:
            '''
            eps=1e-7
            y_pred=preds.view((preds.size()[0],preds.size()[1],-1))#B*C*H*W->B*C*(H*W)
            target=labels.view(y_pred.size())#B*C*H*W->B*C*(H*W)
            ce=-1*torch.log(y_pred+eps)*target
            floss=torch.pow((1-y_pred),self.gamma)*ce
            floss=torch.mul(floss,self.weight)
            floss=torch.sum(floss,dim=1)
            return torch.mean(floss)
    # log


    import os


    # 定义训练的函数
    log_name = name

    #log lunshu
    log('NO {} TIME TRAIN'.format(time_c+1),path)
    log('lr:{},weight_decay:{},maxlen:{},batchsize:{},'.format(learning_rate,weight_decay,max_len,batch_size*acc_step), path)
    def train(model, dataloader, optimizer, criterion, device):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        label_all = []
        preds_all=[]
        f1 = 0
        i = 1
        optimizer.zero_grad()


        for batch in dataloader:

            # 标签形状为 (batch_size, 1)
            label = batch["label"]
            y_test = label
            label = label.to(device)
            text = batch["text"]
            # tokenized_text 包括 input_ids， token_type_ids， attention_mask

            tokenized_text = tokenizer(text, max_length=max_len, add_special_tokens=True, truncation=True,
                                       padding="max_length",
                                       return_tensors="pt")

            tokenized_text = tokenized_text.to(device)

            # 梯度清零
            #optimizer.zero_grad()

            # output: (loss), logits, (hidden_states), (attentions)

            output = model(tokenized_text)

            # y_pred_prob = logits : [batch_size, num_labels]
            y_pred_prob = output
            y_pred_label = [1 if item >0.5else 0 for item in y_pred_prob ]

            # 计算loss
            # 这个 loss 和 output[0] 是一样的
            # loss = criterion(y_pred_prob.view(-1, 4), label.view(-1))
            #loss = output[0]
            label=torch.unsqueeze(label,1)
            label=label.to(torch.float)
            loss=criterion(output,label)

            # f1score rocauc
            y_pred = y_pred_label

            # 计算acc,f1,rocauc

            # predict_prob = predict_prob.extend(y_pred_prob.cpu().numpy()[:,1])
            label_all.extend(y_test)
            preds_all.extend(y_pred)
            # loss 每次是一个 batch 的平均 loss
            epoch_loss += loss.item()
            # 反向传播
            loss.backward()
            #if (i+1) % 8 == 0
            #optimizer.step()
            #optimizer.zero_grad()
            #更新参数,4*16=64
            optimizer.step()
            optimizer.zero_grad()
            # epoch 中的 loss 和 acc 累加

            # acc 是一个 batch 的 acc 总和


            if (i+1) % 100 == 0:
                # 查看是否有梯度回传
                #for name, parms in model.named_parameters():
                #    print("-->name", name, "-->grad_requires", parms.requires_grad, "--weight", torch.mean(parms.data),
                #         "-->grad_value", torch.mean(parms.grad))
                acc = accuracy_score(label_all, preds_all)
                f1=f1_score(label_all,preds_all)
                print("current loss:", epoch_loss / (i + 1), "\t", "current acc:", acc ,
                      "current f1:", f1 )

            i += 1

        # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
        # rocauc=roc_auc_score(label_all,predict_prob)
        acc = accuracy_score(label_all, preds_all)
        f1 = f1_score(label_all, preds_all)
        return epoch_loss / len(dataloader), acc , f1

    def evaluate(model, iterator, device):
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        f1= 0
        auc=0
        predict_prob = []
        label_all = []
        n = 0
        predict_all=[]
        y_pred_score=[]
        labels_all=[]
        scores_all=[]
        with torch.no_grad():
            for batch in iterator:
                n += 1
                label = batch["label"]
                label_all.extend(label)
                y_test = label
                label = label.to(device)
                text = batch["text"]
                tokenized_text = tokenizer(text, max_length=max_len, add_special_tokens=True, truncation=True,
                                           padding="max_length",
                                           return_tensors="pt")
                tokenized_text = tokenized_text.to(device)

                output = model(tokenized_text)
                y_pred_label = [1 if item >0.5else 0 for item in output ]
                label = torch.unsqueeze(label, 1)
                label = label.to(torch.float)
                loss = criterion(output, label)


                # epoch 中的 loss 和 acc 累加
                # loss 每次是一个 batch 的平均 loss
                epoch_loss += loss.item()
                # acc 是一个 batch 的 acc 总和
                #epoch_acc += acc
                # f1score
                y_pred_score=output
                y_pred = y_pred_label
                f1 += f1_score(y_test, y_pred)
                labels = label.data.cpu().numpy()
                predic = y_pred_label
                score = y_pred_score.cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

                scores_all = np.append(scores_all, score)
                #predict_prob.extend(output.cpu().numpy())
        # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量

        # rocauc = roc_auc_score(label_all, predict_prob,multi_class='ovo',average='macro')
        return epoch_loss / len(iterator), labels_all,predict_all,scores_all

    def test(model, iterator, device):
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        f1= 0
        auc=0
        predict_prob = []
        label_all = []
        n = 0
        predict_all=[]
        y_pred_score=[]
        labels_all=[]
        scores_all=[]
        with torch.no_grad():
            for batch in iterator:
                n += 1
                label = batch["label"]
                label_all.extend(label)
                y_test = label
                label = label.to(device)
                text = batch["text"]
                tokenized_text = tokenizer(text, max_length=max_len, add_special_tokens=True, truncation=True,
                                           padding="max_length",
                                           return_tensors="pt")
                tokenized_text = tokenized_text.to(device)

                output = model(tokenized_text)
                y_pred_label = [1 if item > 0.5 else 0 for item in output]
                label = torch.unsqueeze(label, 1)
                label = label.to(torch.float)
                loss = criterion(output, label)

                # epoch 中的 loss 和 acc 累加
                # loss 每次是一个 batch 的平均 loss
                epoch_loss += loss.item()
                # acc 是一个 batch 的 acc 总和
                # epoch_acc += acc
                # f1score
                y_pred_score = output
                y_pred = y_pred_label
                f1 += f1_score(y_test, y_pred)
                labels = label.data.cpu().numpy()
                predic = y_pred_label
                score = y_pred_score.cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

                scores_all = np.append(scores_all, score)
                #predict_prob.extend(output.cpu().numpy())
        # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量

        # rocauc = roc_auc_score(label_all, predict_prob,multi_class='ovo',average='macro')
        return epoch_loss / len(iterator), labels_all,predict_all,scores_all
    # 开始训练和验证
    import time
    begin_time = time.time()
    best_score = 0

    for i in range(epochs):
        train_loss, train_acc, f1s = train(model, sentiment_train_loader, optimizer, criterion, device)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc, 'f1', f1s)
        log_1 = "train loss: {},train acc:{},f1:{}".format(train_loss, train_acc, f1s)
        log(log_1, path)
        valid_loss, labels_all, predict_all,scores_all = evaluate(model, sentiment_valid_loader, device)
        #cal metrics
        valid_acc,f1,roc_auc,pr_auc,result=cal_metrics(labels_all,predict_all,scores_all)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc, "f1:", f1,'roc_auc',roc_auc,'pr_auc',pr_auc)
        log2 = "valid loss: {},valid acc:{},f1:{},roc_auc:{},pr_auc:{}".format(valid_loss, valid_acc, f1,roc_auc,pr_auc)
        log(log2, path)
        result.to_csv('result_sap/{}_result_invalid_{}_epoch{}.csv'.format(name, time_c + 1,i+1), index=False)


        test_loss, labels_all, predict_all, scores_all = test(model, sentiment_test_loader, device)
        # cal metrics
        test_acc, f1, roc_auc, pr_auc, result = cal_metrics(labels_all, predict_all, scores_all)
        log2 = "test loss: {},test acc:{},f1:{},roc_auc:{},pr_auc:{}".format(test_loss, test_acc, f1, roc_auc, pr_auc)
        log(log2, path)
        # 保存test结果
        result.to_csv('result_sap/{}_result_test_{}_epoch{}.csv'.format(name, time_c + 1,i+1), index=False)

        sentiment_train_loader2 = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loss, labels_all, predict_all, scores_all = test(model, sentiment_train_loader2, device)
        # cal metrics
        test_acc, f1, roc_auc, pr_auc, result = cal_metrics(labels_all, predict_all, scores_all)
        log2 = "test loss: {},test acc:{},f1:{},roc_auc:{},pr_auc:{}".format(test_loss, test_acc, f1, roc_auc, pr_auc)
        log(log2, path)
        # 保存train结果
        result.to_csv('result_sap/{}_result_train_{}_epoch{}.csv'.format(name, time_c + 1,i+1), index=False)

        if i in [2,3,4] :
            state = {'net': model.state_dict(), 'epoch': epochs}
            modelpath = '/data/LM_code/ChineseStrokeBert/StrokeTask/sapmodels/baseline_{}_{}time_epoch{}.pth'.format(name,time_c + 1,i+1)
            torch.save(state,modelpath)

    end_time = time.time()
    run_time = round(end_time - begin_time)
    hour = run_time // 3600
    mins = (run_time - hour * 3600) // 60
    s = run_time - 3600 * hour - 60 * mins
    print('run time:{}h{}m{}s'.format(hour, mins, s))
def cal_metrics(labels_all,predict_all,scores_all):
    from scipy.special import softmax
    #scores_all2 = np.reshape(scores_all, (-1, 2))
    #scores = softmax(scores_all2, axis=1)
    #scores=scores[:,1]
    acc=accuracy_score(labels_all, predict_all)
    f1=f1_score(labels_all,predict_all)
    report = classification_report(labels_all, predict_all, target_names=['0','1'], digits=4)
    confusion = confusion_matrix(labels_all, predict_all)
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    rocauc=roc_auc_score(labels_all,scores_all)
    precision,recall,thresholds=precision_recall_curve(labels_all,scores_all)
    prauc=auc(recall,precision)

    # 保存结果
    result = pd.DataFrame()
    result['labels_all'] = labels_all
    result['predict_all'] = predict_all
    result['scores'] = scores_all

    return acc,f1,rocauc,prauc,result
if __name__ == '__main__':
    for name in ["macbertnewend2cnn_3"]:#CSbert pres 16*8 CSbert2 pres 16*8 chongpao CSbert3 ct 16*8 NEW_DATA
        #CSbert4 new pres 16*8 f CSbert5 new inhp 16*8
        #CSbert6 allnew pres
        #CSbert7 prev,csbert8 inhos
        #macbertcnn1 pres
        #macbertcnn2 inhos
        #macbertend chuxue
        #macbertend2 quexue
        if name=='CSbert9':
            pathd="/data/LM_code/ChineseStrokeBert/Pretrain/user_data/model_param/pretrained_model_param/ChineseStrokeBert_2"
        elif "macbert" in name:
            pathd = "/data/LM_code/ChineseStrokeBert/Pretrain/user_data/model_param/pretrain_model_param/mac_bert"
        for i in range(1):
            path_log = log_start(name)
            main(name, pathd,i,path_log)
