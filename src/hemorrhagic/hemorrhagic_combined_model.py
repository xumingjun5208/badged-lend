# coding: UTF-8
import pandas as pd
import pickle
import shap
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import sys
prefix = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(prefix)
from src.hemorrhagic.hemorrhagic_text_model import predict_hemorrhagic_text
from src.hemorrhagic.hemorrhagic_structured_model import HEMORRHAGIC_FEATURE_NAMES, NORMALIZATION_PARAMS

def load_hemorrhagic_combined_model():
    """加载出血性卒中组合预测模型"""
    try:
        print("Loading hemorrhagic combined model...")
        model_path = os.path.join(prefix, 'models', 'hemorrhagic', 'combined', 'ich_SoftVoting_8_mice1.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = pickle.load(open(model_path, 'rb'))
        base_model = model.estimators_[0]
        explainer = shap.TreeExplainer(base_model)
        
        # 预热模型
        print("Warming up hemorrhagic combined model...")
        test_features = [0.5] + [1, 1, 0, 0, 0, 1, 0]  # text_prob + structured_features
        input_data = np.array(test_features).reshape(1, -1)
        
        # 预测
        _ = model.predict_proba(input_data)
        
        # 计算SHAP值
        combined_feature_names = ['Admission score'] + HEMORRHAGIC_FEATURE_NAMES
        data = pd.DataFrame(input_data, columns=combined_feature_names)
        _ = explainer.shap_values(data)
        
        print("Hemorrhagic combined model loaded and warmed up successfully!")
        return model, base_model, explainer
    except Exception as e:
        print(f"Error loading hemorrhagic combined model: {e}")
        raise

# 全局变量存储模型
_hemorrhagic_combined_model = None
_hemorrhagic_combined_base_model = None
_hemorrhagic_combined_explainer = None

def get_hemorrhagic_combined_model():
    """获取已加载的组合模型"""
    global _hemorrhagic_combined_model, _hemorrhagic_combined_base_model, _hemorrhagic_combined_explainer
    if _hemorrhagic_combined_model is None:
        _hemorrhagic_combined_model, _hemorrhagic_combined_base_model, _hemorrhagic_combined_explainer = load_hemorrhagic_combined_model()
    return _hemorrhagic_combined_model, _hemorrhagic_combined_base_model, _hemorrhagic_combined_explainer

def predict_hemorrhagic_combined(text, features):
    """组合预测函数"""
    try:
        # 将features从列表转换为字典
        features_dict = {}
        for i, name in enumerate(HEMORRHAGIC_FEATURE_NAMES):
            features_dict[name] = int(features[i])

        # 标准化特征
        X = [(features_dict[name] - NORMALIZATION_PARAMS[name][0]) /
             NORMALIZATION_PARAMS[name][1] for name in HEMORRHAGIC_FEATURE_NAMES]
        # 获取文本模型预测值
        _, text_prob = predict_hemorrhagic_text(text)

        # 准备组合模型的输入
        combined_features = [round(float(text_prob), 2)] + X
        input_data = np.array(combined_features).reshape(1, -1)
        
        # 获取组合模型
        model, base_model, explainer = get_hemorrhagic_combined_model()
        
        # 预测
        prob = float(model.predict_proba(input_data)[0][1])
        
        # 创建DataFrame用于SHAP解释
        combined_feature_names = ['Admission score'] + HEMORRHAGIC_FEATURE_NAMES
        shape_input = np.array([round(float(text_prob), 2)] + [int(features_dict[name]) for name in HEMORRHAGIC_FEATURE_NAMES]).reshape(1, -1)
        data = pd.DataFrame(shape_input, columns=combined_feature_names)
        
        # 计算SHAP值并生成图像
        try:
            shap_values = explainer.shap_values(data)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # 创建SHAP图，添加 base_value
            plt.figure(figsize=(10, 6))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                data.iloc[0, :],
                feature_names=combined_feature_names,
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

            return {
                "probability": prob,
                "shap_plot": image,  # 返回PIL图像对象
                "feature_importance": dict(zip(
                    combined_feature_names,
                    [float(v) for v in shap_values[0]] if shap_values is not None else []
                ))
            }

        except Exception as e:
            print(f"Error generating SHAP plot: {e}")
            import traceback
            traceback.print_exc()
            return {
                "probability": prob,
                "shap_plot": None,
                "feature_importance": {}
            }
        finally:
            plt.close('all')  # 确保清理所有图形

    except Exception as e:
        print(f"Error in hemorrhagic combined prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "probability": 0.5,
            "shap_plot": None,
            "feature_importance": {}
        }

if __name__ == "__main__":
    print(predict_hemorrhagic_combined(
        "患者，男，60岁，因突发意识丧失1小时入院，既往有高血压病史，无糖尿病史，无吸烟史，无饮酒史，无药物过敏史，无手术史，无外伤史，无家族遗传病史。",
        [1, 1, 0, 0, 0, 1, 0]
    ))
