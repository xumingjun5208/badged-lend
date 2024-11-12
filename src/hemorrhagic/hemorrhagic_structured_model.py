# coding: UTF-8
import pandas as pd
import pickle
import shap
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")

# 定义路径
prefix = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定义特征名和标准化参数
HEMORRHAGIC_FEATURE_NAMES = [
    'Dysphagia', 'Ventilator-associated pneumonia', 'Decubitus ulcer',
    'Hydrocephalus', 'Brain hernia', 'Hyperleukocytosis',
    'Gastrointestinal bleeding'
]

NORMALIZATION_PARAMS = {
    'Dysphagia': (0.0778, 0.0717),
    'Ventilator-associated pneumonia': (0.2053, 0.1631),
    'Decubitus ulcer': (0.028, 0.02723),
    'Hydrocephalus': (0.3219, 0.2183),
    'Brain hernia': (0.3114, 0.2144),
    'Hyperleukocytosis': (0.1177, 0.10383),
    'Gastrointestinal bleeding': (0.1116, 0.09914)
}

def load_hemorrhagic_structured_model():
    """加载出血性卒中结构化预测模型"""
    try:
        print("Loading hemorrhagic structured model...")
        model_path = os.path.join(prefix, 'models', 'hemorrhagic', 'structured', 'ich_SoftVoting_7_mice1.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = pickle.load(open(model_path, 'rb'))
        base_model = model.estimators_[0]
        explainer = shap.TreeExplainer(base_model)
        
        # 预热模型
        print("Warming up hemorrhagic structured model...")
        test_features = {
            "Dysphagia": 1,
            "Ventilator-associated pneumonia": 1,
            "Decubitus ulcer": 0,
            "Hydrocephalus": 0,
            "Brain hernia": 0,
            "Hyperleukocytosis": 1,
            "Gastrointestinal bleeding": 0
        }
        
        # 标准化特征
        X = [(test_features[name] - NORMALIZATION_PARAMS[name][0]) /
             NORMALIZATION_PARAMS[name][1] for name in HEMORRHAGIC_FEATURE_NAMES]
        
        # 预测
        _ = model.predict_proba(np.array(X).reshape(1, -1))
        
        # 计算SHAP值
        data = pd.DataFrame([test_features], columns=HEMORRHAGIC_FEATURE_NAMES)
        _ = explainer.shap_values(data)
        
        print("Hemorrhagic structured model loaded and warmed up successfully!")
        return model, base_model, explainer
    except Exception as e:
        print(f"Error loading hemorrhagic structured model: {e}")
        raise

# 全局变量于存储加载的模型
_hemorrhagic_structured_model = None
_base_model = None
_explainer = None

def get_hemorrhagic_structured_model():
    """获取已加载的模型，如果未加载则加载模型"""
    global _hemorrhagic_structured_model, _base_model, _explainer
    if _hemorrhagic_structured_model is None:
        _hemorrhagic_structured_model, _base_model, _explainer = load_hemorrhagic_structured_model()
    return _hemorrhagic_structured_model, _base_model, _explainer

def predict_hemorrhagic_structured(features):
    """结构化数据预测函数"""
    model, base_model, explainer = get_hemorrhagic_structured_model()
    
    try:
        # 确保特征顺序一致
        features_dict = {}
        for i, name in enumerate(HEMORRHAGIC_FEATURE_NAMES):
            features_dict[name] = int(features[i])

        # 标准化特征
        X = np.array([(features_dict[name] - NORMALIZATION_PARAMS[name][0]) /
                     NORMALIZATION_PARAMS[name][1] for name in HEMORRHAGIC_FEATURE_NAMES])

        # 预测
        prob = model.predict_proba(X.reshape(1, -1))[0][1]

        # 创建DataFrame用于SHAP解释
        data = pd.DataFrame(np.array([int(features_dict[name])
                            for name in HEMORRHAGIC_FEATURE_NAMES]).reshape(1, -1), columns=HEMORRHAGIC_FEATURE_NAMES)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 创建SHAP图，添加 base_value
        plt.figure(figsize=(10, 8))
        shap.force_plot(
            explainer.expected_value,
            shap_values,
            data.iloc[0, :],
            feature_names=HEMORRHAGIC_FEATURE_NAMES,
            matplotlib=True,
            show=False,
            contribution_threshold=0.01
        )
        
        # 保存为PNG格式
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # 读取为PIL图像
        from PIL import Image
        image = Image.open(buf)

        # 创建特征重要性字典
        feature_importance = {
            name: float(value) 
            for name, value in zip(HEMORRHAGIC_FEATURE_NAMES, shap_values.flatten())
        }

        return {
            "probability": float(prob),
            "shap_plot": image,  # 返回PIL图像对象
            "feature_importance": feature_importance
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "probability": 0.5,
            "shap_plot": None,
            "feature_importance": {}
        }
    finally:
        plt.close('all')  # 确保清理所有图形

# 添加特征信息字典
feature_info = {
    "names": HEMORRHAGIC_FEATURE_NAMES,
    "descriptions": {
        "Dysphagia": "吞咽困难",
        "Ventilator-associated pneumonia": "呼吸机相关性肺炎",
        "Decubitus ulcer": "褥疮",
        "Hydrocephalus": "脑积水",
        "Brain hernia": "脑疝",
        "Hyperleukocytosis": "白细胞增多",
        "Gastrointestinal bleeding": "消化道出血"
    }
}

def get_feature_info():
    """获取特征信息"""
    return {
        "names": HEMORRHAGIC_FEATURE_NAMES,
        "descriptions": {
            "Dysphagia": "吞咽困难",
            "Ventilator-associated pneumonia": "呼吸机相关性肺炎",
            "Decubitus ulcer": "褥疮",
            "Hydrocephalus": "脑积水",
            "Brain hernia": "脑疝",
            "Hyperleukocytosis": "白细胞增多",
            "Gastrointestinal bleeding": "消化道出血"
        }
    }

# 在文件开头导出函数
__all__ = [
    'predict_hemorrhagic_structured',
    'get_hemorrhagic_structured_model',
    'feature_info',
    'HEMORRHAGIC_FEATURE_NAMES'
]

#添加main
if __name__ == "__main__":
    print(predict_hemorrhagic_structured({
        "Dysphagia": 1, 
        "Ventilator-associated pneumonia": 1,
        "Decubitus ulcer": 0,
        "Hydrocephalus": 0,
        "Brain hernia": 0,
        "Hyperleukocytosis": 1,
        "Gastrointestinal bleeding": 0
    }))
