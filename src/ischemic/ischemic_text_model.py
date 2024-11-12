# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F
import numpy as np
import warnings
import re
import os
import jieba
import sys

# 定义路径
prefix = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pathd = os.path.join(prefix, "models", "macbert")

warnings.filterwarnings("ignore")
def validate_text_input(text: str):
    """验证文本输入"""
    if not text or not text.strip():
        raise ValueError("文本输入不能为空")
    if len(text.strip()) < 10:
        raise ValueError("文本长度太短，请输入更详细的病历信息")
    return True
# 停止词列表
STOP_WORDS = {
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
    '之', '于', '但', '并', '等', '却', '还', '以', '把', '说',
    '到', '被', '又', '也', '即', '既', '让', '她', '他', '你',
    '我', '它', '这', '那', '些', '么', '什', '啊', '哪', '吗',
    '呢', '吧', '啦', '呀', '嘛', '哇', '呵', '嗯', '哦', '哈',
    '在', '有', '个', '能', '来', '去', '过', '会', '患者', '而且',
    '或者', '一个', '一些', '这个', '那个', '这些', '那些','入院',
    '出院','查体','检查','诊断''随访','死亡','医院','病房','病历',
    '病例', '病史', '既往史', '家族史', 
    '现病史', '个人史', '月', '年', '日', '时', '分', '次',
    '上述', '建议', '医嘱', '治疗', '用药', '服用', '口服',
    '静脉', '肌肉', '注射', '输液', '护理', '观察', '记录',
    '病程', '病情', '症状', '体征', '复查', '复诊', '门诊',
    '住院', '手术', '术后', '术前'
}

# BERT-CNN模型定义
class IschemicBertCNN(nn.Module):
    def __init__(self, path):
        super(IschemicBertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.dropout_rate = 0.1
        self.num_classes = 1
        
        for param in self.bert.parameters():
            param.requires_grad = True
            
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden_size)) for k in self.filter_sizes]
        )
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.fc_cnn = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        self.sig = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        outputs = self.bert(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids'],
            output_attentions=True
        )
        encoder_out = outputs[0]
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout_layer(out)
        out = self.fc_cnn(out)
        out1 = self.sig(out)
        return out1, outputs

# 加载模型相关函数
def load_ischemic_text_model():
    """加载缺血性卒中文本预测模型"""
    try:
        print("Loading ischemic text model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer
        print("Loading tokenizer...")
        tokenizer = BertTokenizerFast.from_pretrained(pathd)
        if tokenizer is None:
            raise ValueError("Failed to initialize tokenizer")
        
        # 加载模型
        print("Loading model...")
        model = IschemicBertCNN(pathd)
        model_path = os.path.join(
            prefix, 'models', 'ischemic', 'text', 'ais_baseline_macbertnewend2cnn_3_1time_epoch3.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=False)
        model.bert.config.attn_implementation = "eager"
        model.eval()
        model.to(device)
        
        # 预热模型
        print("Warming up ischemic text model...")
        test_texts = [
            "这是第一个预热文本。",
            "这是一个较长的预热文本，包含更多的临床信息。",
            "患者，男性，60岁，因突发意识障碍2小时入院。"
        ]
        for text in test_texts:
            inputs = tokenizer(text, max_length=512, return_offsets_mapping=True,
                             add_special_tokens=True, truncation=True,
                             padding=True, return_tensors="pt")
            inputs.to(device)
            with torch.no_grad():
                out, _ = model(inputs)
        
        print("Ischemic text model loaded and warmed up successfully!")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading ischemic text model: {e}")
        raise

# 全局变量存储模型
_ischemic_model = None
_ischemic_tokenizer = None
_ischemic_device = None

def get_ischemic_model():
    """获取已加载的模型"""
    global _ischemic_model, _ischemic_tokenizer, _ischemic_device
    if _ischemic_model is None:
        _ischemic_model, _ischemic_tokenizer, _ischemic_device = load_ischemic_text_model()
    return _ischemic_model, _ischemic_tokenizer, _ischemic_device

_jieba_initialized = False

def init_jieba():
    """初始化jieba分词器"""
    global _jieba_initialized
    if not _jieba_initialized:
        jieba.initialize()
        test_text = "预热分词器的测试文本"
        list(jieba.cut(test_text))
        _jieba_initialized = True

def predict_ischemic_text(text: str):
    """缺血性卒中文本预测函数"""
    try:
        validate_text_input(text)

        model, tokenizer, device = get_ischemic_model()
        
        # 保存原始文本，仅对处理文本进行清洗
        original_text = text
        processed_text = re.sub(r"[*]|[\s]|[\\r]|[\\n]|[\\t]|[\\f]|[\\v]", '', text)
        
        # 使用jieba分词
        words = list(jieba.cut(original_text))
        
        # 模型测试
        inputs = tokenizer(
            processed_text,
            max_length=512,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs.to(device)
        
        out, outputs = model(inputs)
        prob = out.detach().cpu().numpy().item()
        
        # 计算token级别的重要度
        attscore = outputs['attentions'][-1][:, :, 0, :].mean(1)
        token_scores = attscore.detach().cpu().numpy()[0]
        token_spans = inputs['offset_mapping'][0].detach().cpu().numpy()
        
        # 创建字符到分数的映射
        char_scores = {}
        current_pos = 0
        
        # 为每个分词后的词分配分数
        for word in words:
            word_end = current_pos + len(word)
            
            # 如果是停止词，直接设置为0分
            if word in STOP_WORDS:
                for pos in range(current_pos, word_end):
                    char_scores[pos] = 0.0
                current_pos = word_end
                continue
            
            # 收集该词对应的所有token分数
            word_token_scores = []
            for i, (tok_start, tok_end) in enumerate(token_spans):
                if tok_start == tok_end:  # 跳过特殊token
                    continue
                # 检查token是否与当前词有重叠
                if tok_start < word_end and tok_end > current_pos:
                    word_token_scores.append(token_scores[i])
            
            # 计算词的分数（使用最大值）
            word_score = max(word_token_scores) if word_token_scores else 0.0
            
            # 将相同的分数分配给词中的每个字符
            for pos in range(current_pos, word_end):
                char = original_text[pos]
                # 如果是标点符号，分数设为0
                if re.match(r'[^\u4e00-\u9fff\u0030-\u0039\u0041-\u005a\u0061-\u007a]', char):
                    char_scores[pos] = 0.0
                else:
                    char_scores[pos] = word_score
            
            current_pos = word_end
        
        # 转换为数组并归一化
        scores_array = np.array([char_scores.get(i, 0.0) for i in range(len(original_text))])
        
        # 只对非零分数进行归一化
        non_zero_mask = scores_array > 0
        if non_zero_mask.any():
            valid_scores = scores_array[non_zero_mask]
            min_score = np.min(valid_scores)
            max_score = np.max(valid_scores)
            if max_score > min_score:
                normalized_scores = np.zeros_like(scores_array)
                normalized_scores[non_zero_mask] = (scores_array[non_zero_mask] - min_score) / (max_score - min_score)
            else:
                normalized_scores = scores_array
        else:
            normalized_scores = scores_array
        
        # 生成带颜色文本，保持原始文本的每个字符
        tokens_with_scores = []
        for i, char in enumerate(original_text):
            tokens_with_scores.append((char, normalized_scores[i]))
        
        # 定义颜色映射函数
        def get_color_scheme(score):
            """根据重要度值返回颜色方案"""
            abs_score = abs(score)
            if abs_score > 0.75:  # 极高相关
                return {
                    "bg": "rgba(183, 28, 28, 0.9)",  # 深红色
                    "text": "#ffffff",
                    "level": "Extremely High"
                }
            elif abs_score > 0.5:  # 高度相关
                return {
                    "bg": "rgba(239, 83, 80, 0.8)",  # 红色
                    "text": "#ffffff",
                    "level": "High"
                }
            elif abs_score > 0.25:  # 中度相关
                return {
                    "bg": "rgba(255, 152, 0, 0.8)",  # 黄色
                    "text": "#ffffff",
                    "level": "Moderate"
                }
            elif abs_score > 0.15:  # 轻度相关
                return {
                    "bg": "rgba(158, 158, 158, 0.8)",  # 灰色
                    "text": "#ffffff",
                    "level": "Low"
                }
            else:  # 相关性很低
                return None
        
        # 生成HTML文本
        html_text = ""
        for token, score in tokens_with_scores:
            colors = get_color_scheme(score)
            if colors is None:
                html_text += f'<span>{token}</span>'
            else:
                # 添加悬停提示，显示重要度级别和具体数值
                tooltip = f"{colors['level']} (Importance: {abs(score):.3f})"
                html_text += (
                    f'<span style="background-color: {colors["bg"]}; '
                    f'color: {colors["text"]}; padding: 2px 4px; '
                    f'border-radius: 4px; margin: 0 1px; '
                    f'transition: all 0.3s ease;" '
                    f'title="{tooltip}">'
                    f'{token}</span>'
                )
        
        return html_text, prob
    
    except Exception as e:
        raise e
if __name__ == "__main__":
    print(predict_ischemic_text("患者，男，60岁，因突发意识丧失1小时入院，既往有高血压病史，无糖尿病史，无吸烟史，无饮酒史，无药物过敏史，无手术史，无外伤史，无家族遗传病史。"))
