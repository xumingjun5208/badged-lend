# 卒中相关性肺炎预测系统
[English](README.md) | [简体中文](README_CN.md)
## 项目概述

本系统是一个基于深度学习的智能辅助诊断系统，用于预测卒中患者发生相关性肺炎的风险。系统支持缺血性卒中和出血性卒中两种类型的预测任务，通过整合病历文本和临床指标，为医生提供辅助决策支持。

![](https://github.com/user-attachments/assets/465adf14-6cb9-41e4-b56b-776c077f2e25)

## 在线演示
[🎯用户指南](docs/help_EN.md)

访问在线演示： [⚡demo](http://www.badged-lend.com)

![sap前端中文](https://github.com/user-attachments/assets/ece96a31-5566-4036-a123-c5221ba51214)

## 系统架构

### 模型组成

1. 文本预测模型

   - 缺血性卒中文本模型 (ischemic_text_model.py)
     - 基于 MacBERT-CNN 架构
     - 处理中文病历文本
     - 输出风险概率和文本重要性分析
   - 出血性卒中文本模型 (hemorrhagic_text_model.py)
     - 相同的模型架构
     - 针对出血性卒中特点优化
     - 提供文本分析可视化

2. 结构化预测模型

   - 缺血性卒中结构化模型 (ischemic_structured_model.py)
     - 基于软投票集成学习
     - 处理 6 个关键临床指标
     - 阈值为 0.2 的二分类预测
   - 出血性卒中结构化模型 (hemorrhagic_structured_model.py)
     - 同样基于集成学习
     - 处理 7 个临床指标
     - 阈值为 0.3 的二分类预测

3. 组合预测模型
   - 缺血性卒中组合模型 (ischemic_combined_model.py)
     - 融合文本和结构化特征
     - 文本预测概率作为额外特征
     - SHAP 值解释预测结果
   - 出血性卒中组合模型 (hemorrhagic_combined_model.py)
     - 类似的融合策略
     - 针对性的特征组合
     - 可视化特征重要性

### 特征说明

1. 缺血性卒中特征

   - 基础信息
     - 年龄：连续值
     - 抢救次数：整数值
   - 临床表现
     - 吞咽困难：是/否
     - 呼吸机相关性肺炎：是/否
     - 褥疮：是/否
     - 肺部疾病：是/否

2. 出血性卒中特征
   - 基础症状
     - 吞咽困难：是/否
     - 呼吸机相关性肺炎：是/否
     - 褥疮：是/否
   - 特殊指标
     - 脑积水：是/否
     - 脑疝：是/否
     - 白细胞增多：是/否
     - 消化道出血：是/否

### 预测逻辑

1. 单一模式预测

   - 仅文本输入
     - 使用 BERT 提取文本特征
     - CNN 进行特征融合
     - 输出风险概率和重要文本标注
   - 仅指标输入
     - 标准化临床指标
     - 集成模型预测
     - SHAP 值分析特征贡献

2. 组合模式预测
   - 文本处理流程
     - BERT-CNN 提取文本特征
     - 获取文本预测概率
   - 特征融合
     - 文本预测值作为新特征
     - 与临床指标组合
   - 最终预测
     - 集成模型综合预测
     - 生成 SHAP 解释图
     - 输出风险评估结果

### 输出结果

1. 风险预测

   - 缺血性卒中
     - 高风险：概率 > 20%
     - 低风险：概率 ≤ 20%
   - 出血性卒中
     - 高风险：概率 > 30%
     - 低风险：概率 ≤ 30%

2. 文本分析

   - 重要性等级
     - 极高相关：深红色标注
     - 高度相关：红色标注
     - 中度相关：黄色标注
     - 轻度相关：灰色标注
   - 交互功能
     - 悬停显示重要度值

3. 特征重要性
   - SHAP 值分析
     - 红色：增加风险的因素
     - 蓝色：降低风险的因素
   - 可视化展示
     - 瀑布图展示贡献度
     - 标注具体影响值

## 技术实现

### 前端界面

- Gradio 框架
  - 响应式布局
  - 深色模式支持
  - 自定义 CSS 样式
  - 交互式组件
  - 中英文双语支持

### 后端技术

- 深度学习框架
  - Python 3.8+
  - PyTorch 1.8+
  - Transformers 4.0+
- 机器学习工具
  - Scikit-learn
  - SHAP 0.40+
- 工具库
  - Numpy
  - Pandas
  - Matplotlib

### 系统功能

- 任务切换
  - 缺血性/出血性卒中选择
  - 界面动态更新
- 数据输入
  - 文本输入框
  - 临床指标选择
  - 示例数据填充
- 结果展示
  - 风险预测结果
  - 文本分析可视化
  - SHAP 值解释图

## 部署说明

### 环境配置

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

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 下载模型文件

选项 1：自动下载
```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 运行下载脚本
python scripts/download_models.py
```

选项 2：手动下载
从以下地址下载模型文件：
- HuggingFace 仓库：1.[donghao1234/badged-lend](https://huggingface.co/donghao1234/badged-lend)
2.[hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)

模型文件结构：
```
models/
├── ischemic/
│   ├── text/
│   │   └── ais_baseline_macbertnewend2cnn_3_1time_epoch3.pth
│   ├── structured/
│   │   └── ais_SoftVoting_6_mice1.pkl
│   └── combined/
│       └── ais_SoftVoting_7_mice1.pkl
├── hemorrhagic/
│   ├── text/
│   │   └── ich_baseline_macbertnewend1cnn_1time_epoch3.pth
│   ├── structured/
│   │   └── ich_SoftVoting_7_mice1.pkl
│   └── combined/
│       └── ich_SoftVoting_8_mice1.pkl
└── macbert/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    └── tokenizer_config.json
```

5. 启动应用

```bash
python app.py
```

6. 访问地址
   - 本地访问：http://localhost:8080
   - 默认端口：7860

## 开发团队

### 研究机构

中国药科大学理学院医药大数据与人工智能实验室

### 联系方式

- 📧 邮箱：liaojun@cpu.edu.cn
- 🏛️ 地址：中国药科大学理学院
- 🔬 实验室：医药大数据与人工智能实验室
- ⏰ 工作时间：周一至周五 9:00-18:00

## 许可证

[Apache 2.0](LICENSE)

## 备案信息

- [苏公网安备 32010602011293 号](https://beian.mps.gov.cn/#/query/webSearch)
- [苏 ICP 备 2023023603](https://beian.miit.gov.cn/#/Integrated/index)
