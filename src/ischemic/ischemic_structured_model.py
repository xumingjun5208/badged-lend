# coding: UTF-8
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shap
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端

# 定义路径
prefix = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# 定义特征名和标准化参数
ISCHEMIC_FEATURE_NAMES = [
    'Age', 'Number of resuscitations', 'Dysphagia',
    'Ventilator-associated pneumonia', 'Decubitus ulcer', 'Lung disease'
]

NORMALIZATION_PARAMS = {
    'Dysphagia': (0.1078, 0.0962),
    'Ventilator-associated pneumonia': (0.0166, 0.0163),
    'Decubitus ulcer': (0.0132, 0.013),
    'Age': (68.856, 132.063),
    'Number of resuscitations': (0.0653, 0.0611),
    'Lung disease': (0.3595, 0.2303)
}


def load_ischemic_structured_model():
    """加载缺血性卒中结构化预测模型"""
    try:
        print("Loading ischemic structured model...")
        model_path = os.path.join(prefix, 'models', 'ischemic', 'structured', 'ais_SoftVoting_6_mice1.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = pickle.load(open(model_path, 'rb'))
        base_model = model.estimators_[0]
        explainer = shap.TreeExplainer(base_model)
        
        # 预热模型
        print("Warming up ischemic structured model...")
        test_features = {
            "Age": 60,
            "Number of resuscitations": 1,
            "Dysphagia": 0,
            "Ventilator-associated pneumonia": 1,
            "Decubitus ulcer": 0,
            "Lung disease": 1
        }
        
        # 标准化特征
        X = [(test_features[name] - NORMALIZATION_PARAMS[name][0]) /
             NORMALIZATION_PARAMS[name][1] for name in ISCHEMIC_FEATURE_NAMES]
        
        # 预测
        _ = model.predict_proba(np.array(X).reshape(1, -1))
        
        # 计算SHAP值
        data = pd.DataFrame([test_features], columns=ISCHEMIC_FEATURE_NAMES)
        _ = explainer.shap_values(data)
        
        print("Ischemic structured model loaded and warmed up successfully!")
        return model, base_model, explainer
    except Exception as e:
        print(f"Error loading ischemic structured model: {e}")
        raise


# 全局变量存储模型
_ischemic_structured_model = None
_ischemic_base_model = None
_ischemic_explainer = None


def get_ischemic_structured_model():
    """获取已加载的模型，如果未加载则加载模型"""
    global _ischemic_structured_model, _ischemic_base_model, _ischemic_explainer
    if _ischemic_structured_model is None:
        _ischemic_structured_model, _ischemic_base_model, _ischemic_explainer = load_ischemic_structured_model()
    return _ischemic_structured_model, _ischemic_base_model, _ischemic_explainer


def predict_ischemic_structured(features):
    """缺血性卒中结构化数据预测函数"""
    model, base_model, explainer = get_ischemic_structured_model()

    try:
        # 确保特征顺序一致
        features_dict = {}
        for i, name in enumerate(ISCHEMIC_FEATURE_NAMES):
            features_dict[name] = int(features[i])

        # 标准化特征
        X = np.array([(features_dict[name] - NORMALIZATION_PARAMS[name][0]) /
                     NORMALIZATION_PARAMS[name][1] for name in ISCHEMIC_FEATURE_NAMES])

        prob = model.predict_proba(X.reshape(1, -1))[0][1]

        # 创建DataFrame用于SHAP解释
        data = pd.DataFrame(np.array([int(features_dict[name])
                            for name in ISCHEMIC_FEATURE_NAMES]).reshape(1, -1), columns=ISCHEMIC_FEATURE_NAMES)

        # 检查explainer是否有效
        if explainer is not None:
            try:
                # 计算SHAP值
                shap_values = explainer.shap_values(data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                # 创建SHAP图
                plt.figure(figsize=(10, 8))
                if hasattr(explainer, 'expected_value'):
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        data.iloc[0, :],
                        feature_names=ISCHEMIC_FEATURE_NAMES,
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
                else:
                    image = None
            except Exception as e:
                print(f"Error in SHAP calculation: {e}")
                image = None
        else:
            image = None

        return {
            "probability": float(prob),
            "shap_plot": image,
            "feature_importance": {}
        }

    except Exception as e:
        print(f"Error in ischemic structured prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "probability": 0.5,
            "shap_plot": None,
            "feature_importance": {}
        }
    finally:
        plt.close('all')


# 添加特征信息字典
feature_info = {
    "names": ISCHEMIC_FEATURE_NAMES,
    "descriptions": {
        "Dysphagia": "吞咽困难",
        "Ventilator-associated pneumonia": "呼吸机相关性肺炎",
        "Decubitus ulcer": "褥疮",
        "Age": "年龄",
        "Number of resuscitations": "抢救次数",
        "Lung disease": "肺部疾病"
    }
}


def get_feature_info():
    """获取特征信息"""
    return feature_info


# 在文件开头导出函数
__all__ = [
    'predict_ischemic_structured',
    'get_ischemic_structured_model',
    'feature_info',
    'ISCHEMIC_FEATURE_NAMES'
]

# 添加main
if __name__ == "__main__":
    print(predict_ischemic_structured({
        "Age": 60,
        "Number of resuscitations": 1,
        "Dysphagia": 0,
        "Ventilator-associated pneumonia": 1,
        "Decubitus ulcer": 0,
        "Lung disease": 1
    }))
