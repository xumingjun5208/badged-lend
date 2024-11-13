# 定义项目路径
from regex import F
from tomlkit import value
from src.hemorrhagic.hemorrhagic_combined_model import get_hemorrhagic_combined_model, predict_hemorrhagic_combined
from src.hemorrhagic.hemorrhagic_structured_model import (
    get_hemorrhagic_structured_model,
    predict_hemorrhagic_structured,
    feature_info as hemorrhagic_feature_info,
    HEMORRHAGIC_FEATURE_NAMES
)
from src.hemorrhagic.hemorrhagic_text_model import get_hemorrhagic_text_model, predict_hemorrhagic_text, init_jieba as init_hemorrhagic_jieba
from src.ischemic.ischemic_combined_model import get_ischemic_combined_model, predict_ischemic_combined
from src.ischemic.ischemic_structured_model import (
    get_ischemic_structured_model,
    predict_ischemic_structured,
    feature_info as ischemic_feature_info,
    ISCHEMIC_FEATURE_NAMES
)
from src.ischemic.ischemic_text_model import get_ischemic_model, predict_ischemic_text, init_jieba
from src.utils.theme import ThemeConfig
from src.utils.logger import Logger
import os
import gradio as gr
import sys
import time
import json
import gc
import torch

def cleanup():
    """清理内存资源"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
prefix = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prefix)

# 获取主题CSS
theme_css = ThemeConfig.get_theme_css()
theme_js = ThemeConfig.get_theme_toggle_js()
# 读取自定义CSS文件
with open(os.path.join(prefix, "src", "static", "css", "custom.css"), "r", encoding="utf-8") as f:
    custom_css = f.read()

# 初始化custom_css
custom_css = theme_css


def handle_prediction_error(e):
    """统一的预测错误处理"""
    outputs = {
        "text_analysis": gr.update(value=None, visible=False),
        "result": gr.update(value=f"""
            <div style="padding: 1em; border-radius: 0.5em; background: #ef5350; color: white;">
                <h3 style="margin: 0">预测失败</h3>
                <p style="margin: 0.5em 0">发生错误: {str(e)}</p>
            </div>
        """, visible=True),
        "shap_plot": gr.update(value=None, visible=False),
        "text_analysis_group": gr.update(visible=False),
        "shap_group": gr.update(visible=False)
    }
    return list(outputs.values())  # 这里需要确保返回顺序与预期一致


# 在文件开头添加示例文本
EXAMPLE_TEXTS = [
    """患者，男，68岁，因"突发意识碍4小时"入院。患4小时前明显诱因突发意识障碍，言语不能，右侧肢体活动障碍。查体：志昏睡，BP 175/95mmHg，心率80次/分，双瞳等大同圆，对光反射迟钝，右侧肢体肌力2级。""",

    """患者，女性，75岁。因"突发右侧肢体无力、言语不清2小时"入院。患者2小时前无明显诱因出现右侧肢体无力，言语含糊不清。既往有高血压病史10年。查体：神志清，言语不清，右侧肢体肌力3级，病理征阳性。""",

    """患者，男，58岁，因"头痛呕吐后意识丧失3小时"入院。患者3小时前突发剧烈头痛，随后出现呕吐3次，后逐渐出现意识障碍。查体：昏迷，GCS 6分，BP 190/110mmHg，双瞳不等大，右侧瞳孔散大。"""
]

# 读取中英文帮助文档内容
with open(prefix + "/docs/help.md", "r", encoding="utf-8") as f:
    help_content_zh = f.read()

with open(prefix + "/docs/help_EN.md", "r", encoding="utf-8") as f:
    help_content_en = f.read()

# 保留作为备用的默认翻译
TRANSLATIONS = {
    "zh": {
        "title": "🏥 卒中相关性肺炎预测系统",
        "subtitle": "基于深度学习的智能辅助诊断系统",
        "task_selector": {
            "label": "选择预测任务",
            "info": "请选择要进行的预测任务类型",
            "choices": ["缺血性卒中相关肺炎预测", "出血性卒中相关肺炎预测"]
        },
        "guide_button": "📖 使用指南",
        "text_input": {
            "label": "病历文本",
            "placeholder": "请输入患者病历文本，包括主诉、现病史、查体等信息..."
        },
        "input_guide": """
            ### 📝 输入指南
            - 建议包含：主诉、现病史、查体等信息
            - 建议长度：100-1000
            - 关键信息：意识状态、生命体征、神经系统体征等
        """,
        "example_button": "示例",
        "clinical_indicators": "### 📊 临床指标",
        "predict_button": "预测分析",
        "clear_button": "清除重置",

        "output": {
            "prediction_result": "预测结果",
            "high_risk": "高风险",
            "low_risk": "低风险",
            "risk_probability": "发生风险",
            "ischemic_high_risk_note": "患者可能在入院后7天内发生缺血性卒中相关性肺炎",
            "ischemic_low_risk_note": "患者发生缺血性卒中相关性肺炎的风险较低",
            "hemorrhagic_high_risk_note": "患者可能在入院后7天内发生出血性卒中相关性肺炎",
            "hemorrhagic_low_risk_note": "患者发生出血性卒中相关性肺炎的风险较低",
            "prediction_error": "请至少输入病历文本或临床指标",

            "text_analysis": {
                "title": "📊 文本分析结果",
                "legend": {
                    "extremely_high": "极高相关性",
                    "high": "高度相关性",
                    "moderate": "中度相关性",
                    "low": "轻度相关性"
                },
                "legend_html": {
                    "title": "📊 文本重要度说明：",
                    "legend": {
                        "extremely_high": "极高相关性",
                        "high": "高度相关性",
                        "moderate": "中度相关性",
                        "low": "轻度相关性"
                    },
                    "note": "提示：鼠标悬停在标记文本上可查看具体重要度值"
                }
            },

            "feature_importance": {
                "title": "📈 特征重要性分析",
                "legend": {
                    "increase_risk": "增加风险的因素",
                    "decrease_risk": "降低风险的因素",
                    "impact": "影响程度"
                },
            }
        }
    },
    "en": {
        "title": "🏥 Stroke-Associated Pneumonia Prediction System",
        "subtitle": "AI-Powered Diagnostic Assistant System",
        "task_selector": {
            "label": "Select Prediction Task",
            "info": "Please select the type of prediction task",
            "choices": ["Ischemic Stroke Pneumonia Prediction", "Hemorrhagic Stroke Pneumonia Prediction"]
        },
        "guide_button": "📖 User Guide",
        "text_input": {
            "label": "Medical Record",
            "placeholder": "Please enter patient's medical record, including chief complaint, present illness, physical examination..."
        },
        "input_guide": """
            ### 📝 Input Guide
            - Recommended content: Chief complaint, present illness, physical examination
            - Recommended length: 100-1000 words
            - Key information: Consciousness, vital signs, neurological signs
        """,
        "example_button": "Example",
        "clinical_indicators": "### 📊 Clinical Indicators",
        "predict_button": "Predict",
        "clear_button": "Clear",

        "output": {
            "prediction_result": "Prediction Result",
            "high_risk": "High Risk",
            "low_risk": "Low Risk",
            "risk_probability": "Risk Probability",
            "ischemic_high_risk_note": "Patient may develop ischemic stroke-associated pneumonia within 7 days after admission",
            "ischemic_low_risk_note": "Patient has a low risk of developing ischemic stroke-associated pneumonia",
            "hemorrhagic_high_risk_note": "Patient may develop hemorrhagic stroke-associated pneumonia within 7 days after admission",
            "hemorrhagic_low_risk_note": "Patient has a low risk of developing hemorrhagic stroke-associated pneumonia",
            "prediction_error": "Please enter at least one of the medical record or clinical indicators",

            "text_analysis": {
                "title": "📊 Text Analysis Results",
                "legend": {
                    "extremely_high": "Extremely high correlation",
                    "high": "High correlation",
                    "moderate": "Moderate correlation",
                    "low": "Low correlation"
                },
                "legend_html": {
                    "title": "📊 Text Importance Explanation:",
                    "legend": {
                        "extremely_high": "Extremely high correlation",
                        "high": "High correlation",
                        "moderate": "Moderate correlation",
                        "low": "Low correlation"
                    },
                    "note": "Hover over marked text to view specific importance values"}
            },

            "feature_importance": {
                "title": "📈 Feature Importance Analysis",
                "legend": {
                    "increase_risk": "Risk-increasing factors",
                    "decrease_risk": "Risk-decreasing factors",
                    "impact": "Impact magnitude"
                },

            }
        }
    }
}


def load_language(lang: str) -> dict:
    """加载语言配置文件"""
    lang_file = "zh-cn.json" if lang == "zh" else "en.json"
    lang_path = os.path.join(prefix, "lang", lang_file)
    try:
        with open(lang_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading language file {lang_file}: {e}")
        # 如果加载失败，返回内置的默认翻译
        return TRANSLATIONS[lang]


# 加载默认语言（中文）
current_texts = load_language("zh")

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=("Inter", "system-ui", "sans-serif")
    ),
    css=custom_css,
    elem_classes=["animated-interface"]
) as demo:
    # 添加主题切换脚本
    gr.HTML(f"<script>{theme_js}</script>")

    # 使用加载的语言配置
    language = gr.Radio(
        choices=["中文", "English"],
        value="中文",
        label="🌐 Language / 语言",
        elem_classes=["language-selector", "fade-in"]
    )

    title_html = gr.HTML(
        f"""
        <div class="header" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">{current_texts['title']}</h1>
            <p style="font-size: 1.1rem; color: var(--secondary-color);">
                {current_texts['subtitle']}
            </p>
        </div>
        """,
        elem_classes=["header", "slide-down"]
    )

    # 任务选择
    with gr.Row(elem_classes=["task-selector-container", "fade-in"]):
        task_selector = gr.Radio(
            choices=TRANSLATIONS["zh"]["task_selector"]["choices"],
            value=TRANSLATIONS["zh"]["task_selector"]["choices"][0],
            label=TRANSLATIONS["zh"]["task_selector"]["label"],
            info=TRANSLATIONS["zh"]["task_selector"]["info"],
            elem_classes=["task-selector"]
        )

    # 修改帮助文档部分
    with gr.Accordion("📖 使用指南", open=False, elem_classes=["help-accordion", "slide-up"]) as help_accordion:
        help_content_markdown = gr.Markdown(
            help_content_zh,
            elem_classes=["fade-in"]
        )

    # 缺血性卒界面
    with gr.Column(visible=True, elem_classes=["input-section", "slide-right"]) as ischemic_interface:
        with gr.Row(equal_height=True):
            # 左侧：文本输入
            with gr.Column(scale=2, min_width=400):
                ischemic_text = gr.Textbox(
                    label=TRANSLATIONS["zh"]["text_input"]["label"],
                    placeholder=TRANSLATIONS["zh"]["text_input"]["placeholder"],
                    lines=10,
                    max_lines=20,
                    show_copy_button=True,
                    elem_classes=["text-input"]
                )

                # 输入指南
                ischemic_input_guide = gr.Markdown(
                    TRANSLATIONS["zh"]["input_guide"],
                    elem_classes=["input-guide"]
                )

                # 示例按钮
                with gr.Row(variant="compact"):
                    ischemic_example_buttons = []
                    for i, text in enumerate(EXAMPLE_TEXTS):
                        button = gr.Button(
                            f"示例 {i+1}",
                            size="sm",
                            elem_classes=["example-button"]
                        )
                        button.click(
                            lambda t=text: t,
                            outputs=[ischemic_text]
                        )
                        ischemic_example_buttons.append(button)

            # 右侧：临床指标
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["clinical-indicators"]):
                    ischemic_indicators_title = gr.Markdown(
                        TRANSLATIONS["zh"]["clinical_indicators"]
                    )

                    # 使用Container组件包装指标
                    with gr.Column(variant="panel"):
                        ischemic_features = []

                        # 年龄滑块
                        ischemic_features.append(
                            gr.Slider(
                                minimum=18,
                                maximum=100,
                                step=1,
                                label="年龄",
                                info="Age (years)",
                                value=False,
                                elem_classes=["slider-input"],
                                container=True
                            )
                        )

                        # 抢救次数
                        ischemic_features.append(
                            gr.Slider(
                                minimum=0,
                                maximum=10,
                                step=1,
                                label="抢救次数",
                                info="Resuscitation Count",
                                value=False,
                                elem_classes=["slider-input"],
                                container=True
                            )
                        )

                        # 布尔特征使用紧凑布局
                        with gr.Column(variant="compact"):
                            for name in ['Dysphagia', 'Ventilator-associated pneumonia', 'Decubitus ulcer', 'Lung disease']:
                                ischemic_features.append(
                                    gr.Checkbox(
                                        label=ischemic_feature_info["descriptions"][name],
                                        info=name,
                                        elem_classes=["checkbox-input"],
                                        container=True
                                    )
                                )

        # 按钮组使用紧凑布局
        with gr.Row(variant="compact", equal_height=True):
            ischemic_predict = gr.Button(
                TRANSLATIONS["zh"]["predict_button"],
                variant="primary",
                size="lg",
                elem_classes=["predict-button", "pulse"]
            )
            ischemic_clear = gr.Button(
                TRANSLATIONS["zh"]["clear_button"],
                size="lg",
                elem_classes=["clear-button"]
            )

        # 输出区使用响应式布局
        with gr.Row(equal_height=True) as ischemic_output:
            # 左侧：预测结果
            with gr.Column(scale=1, min_width=300):
                ischemic_result = gr.HTML(
                    label=TRANSLATIONS["zh"]["output"]["prediction_result"],
                    visible=False,
                    elem_classes=["prediction-result"]
                )

            # 右侧：分析结果
            with gr.Column(scale=2, min_width=400):
                # 文本分析
                with gr.Group(visible=False) as ischemic_text_analysis_group:
                    ischemic_text_analysis_title = gr.Markdown(
                        TRANSLATIONS["zh"]["output"]["text_analysis"]["title"]
                    )
                    ischemic_text_analysis = gr.HTML(
                        elem_classes=["text-analysis"]
                    )

                # SHAP图
                with gr.Group(visible=False) as ischemic_shap_group:
                    ischemic_shap_title = gr.Markdown(
                        TRANSLATIONS["zh"]["output"]["feature_importance"]["title"]
                    )
                    ischemic_shap = gr.Image(
                        label="",
                        show_label=False,
                        visible=False,
                        elem_classes=["shap-plot"],
                        container=True
                    )

    # 修改缺血性卒中的预测事件处理
    def process_ischemic_prediction(lang_choice, text, *features):
        """处理缺血性卒中预测"""
        lang = "en" if lang_choice == "English" else "zh"
        # 添加图例说明
        legend_html = f"""
        <div style="margin-top: 16px; padding: 12px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div style="display: flex; align-items: center; gap: 24px;">
                <div style="font-weight: 500; color: #334155;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["title"]}</div>
                <div style="display: flex; gap: 24px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(183, 28, 28, 0.9); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["extremely_high"]}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(239, 83, 80, 0.8); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["high"]}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(255, 152, 0, 0.8); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["moderate"]}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(158, 158, 158, 0.8); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["low"]}</span>
                    </div>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 0.9em; color: #64748b;">
                {TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["note"]}
            </div>
        </div>
        """
        try:
            # 将特征列表转换为正确的格式
            feature_list = list(features)

            # 检查是否有文本输入和特征输入
            has_text = bool(text and text.strip())
            has_features = any(feature_list)

            if not (has_text or has_features):
                raise ValueError(
                    TRANSLATIONS[lang]["output"]["prediction_error"])

            # 根据输入情况选择预测方式
            if has_text and not has_features:
                # 仅文本输入：使用文本模型
                text_html, prob = predict_ischemic_text(text)
                text_html = f"""
                <div class="text-analysis-container">
                    <div class="text-content" style="line-height: 1.8; font-size: 1em; color: #1e293b;">
                        {text_html}
                    </div>
                    {legend_html}
                </div>
                """
                result_outputs = [
                    gr.update(value=text_html, visible=True),     # 文本分析结果
                    None,                                         # SHAP图
                    gr.update(visible=True),                      # 文本分析组
                    gr.update(visible=False)                      # SHAP组
                ]
            elif not has_text and has_features:
                # 仅特征输入：使用结构化模型
                results = predict_ischemic_structured(feature_list)
                prob = results["probability"]
                result_outputs = [
                    gr.update(value=None, visible=False),         # 文本分析结果
                    gr.update(value=results["shap_plot"],
                              visible=True),  # SHAP图
                    gr.update(visible=False),                     # 文本分析组
                    gr.update(visible=True)                       # SHAP组
                ]
            else:
                # 同时输入：使用组合模型
                results = predict_ischemic_combined(text, feature_list)
                text_html, _ = predict_ischemic_text(text)
                text_html = f"""
                <div class="text-analysis-container">
                    <div class="text-content" style="line-height: 1.8; font-size: 1em; color: #1e293b;">
                        {text_html}
                    </div>
                    {legend_html}
                </div>
                """
                prob = results["probability"]
                result_outputs = [
                    gr.update(value=text_html, visible=True),     # 文本分析结果
                    gr.update(value=results["shap_plot"],
                              visible=True),  # SHAP图
                    gr.update(visible=True),                      # 文本分析组
                    gr.update(visible=True)                       # SHAP组
                ]

            # 生成预测结果HTML
            threshold = 0.2
            risk_level = TRANSLATIONS[lang]["output"]["high_risk"] if prob > threshold else TRANSLATIONS[lang]["output"]["low_risk"]
            risk_color = "#ef5350" if prob > threshold else "#4caf50"
            risk_note = TRANSLATIONS[lang]["output"]["ischemic_high_risk_note"] if prob > threshold else TRANSLATIONS[lang]["output"]["ischemic_low_risk_note"]

            result_html = f"""
            <div style="padding: 1em; border-radius: 0.5em; background: {risk_color}; color: white;">
                <h3 style="margin: 0">{TRANSLATIONS[lang]["output"]["prediction_result"]}: {risk_level}</h3>
                <p style="margin: 0.5em 0">{TRANSLATIONS[lang]["output"]["risk_probability"]}: {prob:.1%}</p>
                <p style="margin: 0.5em 0">{risk_note}</p>
            </div>
            """
            # 清理加载过程中的临时内存
            cleanup()
            # 返回所有输出
            return [
                result_outputs[0],                               # 文本分析结果
                gr.update(value=result_html, visible=True),      # 预测结果
                result_outputs[1],                               # SHAP图
                result_outputs[2],                               # 文本分析组
                result_outputs[3]                                # SHAP组
            ]
        except Exception as e:
            # 清理加载过程中的临时内存
            cleanup()
            print(f"Error in ischemic prediction: {e}")
            return handle_prediction_error(e)

    ischemic_predict.click(
        fn=process_ischemic_prediction,
        inputs=[language] + [ischemic_text] + ischemic_features,
        outputs=[
            ischemic_text_analysis,
            ischemic_result,
            ischemic_shap,
            ischemic_text_analysis_group,
            ischemic_shap_group
        ]
    )

    # 添加缺血性卒中的清除事件处理
    def clear_ischemic():
        """清除缺血性卒中界面的所有输入和输出"""
        return [
            None,  # 文本输入
            *[gr.update(value=False) for _ in ischemic_features],  # 特征输入
            gr.update(value=None, visible=False),  # 文本分析
            gr.update(value=None, visible=False),  # 预测结果
            gr.update(value=None, visible=False),  # SHAP图
            gr.update(visible=False),  # 文本分析组
            gr.update(visible=False)   # SHAP组
        ]

    ischemic_clear.click(
        fn=clear_ischemic,
        outputs=[ischemic_text] + ischemic_features + [
            ischemic_text_analysis,
            ischemic_result,
            ischemic_shap,
            ischemic_text_analysis_group,
            ischemic_shap_group
        ]
    )

    # 出血性卒中界面
    with gr.Column(visible=False, elem_classes=["input-section", "slide-right"]) as hemorrhagic_interface:
        with gr.Row(equal_height=True):
            # 左侧：文本输入
            with gr.Column(scale=2, min_width=400):
                hemorrhagic_text = gr.Textbox(
                    label=TRANSLATIONS["zh"]["text_input"]["label"],
                    placeholder=TRANSLATIONS["zh"]["text_input"]["placeholder"],
                    lines=10,
                    max_lines=20,
                    show_copy_button=True,
                    elem_classes=["text-input"]
                )

                # 输入指南
                hemorrhagic_input_guide = gr.Markdown(
                    TRANSLATIONS["zh"]["input_guide"],
                    elem_classes=["input-guide"]
                )

                # 示例按钮
                with gr.Row(variant="compact"):
                    hemorrhagic_example_buttons = []
                    for i, text in enumerate(EXAMPLE_TEXTS):
                        button = gr.Button(
                            f"示例 {i+1}",
                            size="sm",
                            elem_classes=["example-button"]
                        )
                        button.click(
                            lambda t=text: t,
                            outputs=[hemorrhagic_text]
                        )
                        hemorrhagic_example_buttons.append(button)

            # 右侧：临床指标
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["clinical-indicators"]):
                    hemorrhagic_indicators_title = gr.Markdown(
                        TRANSLATIONS["zh"]["clinical_indicators"]
                    )

                    # 使用Container组件包装指标
                    with gr.Column(variant="panel"):
                        hemorrhagic_features = []

                        # 布尔特征使用紧凑布局
                        with gr.Column(variant="compact"):
                            for name in ['Dysphagia', 'Ventilator-associated pneumonia', 'Decubitus ulcer',
                                         'Hydrocephalus', 'Brain hernia', 'Hyperleukocytosis', 'Gastrointestinal bleeding']:
                                hemorrhagic_features.append(
                                    gr.Checkbox(
                                        label=hemorrhagic_feature_info["descriptions"][name],
                                        info=name,
                                        elem_classes=["checkbox-input"],
                                        container=True
                                    )
                                )

        # 按钮组使用紧凑布局
        with gr.Row(variant="compact", equal_height=True):
            hemorrhagic_predict = gr.Button(
                TRANSLATIONS["zh"]["predict_button"],
                variant="primary",
                size="lg",
                elem_classes=["predict-button", "pulse"]
            )
            hemorrhagic_clear = gr.Button(
                TRANSLATIONS["zh"]["clear_button"],
                size="lg",
                elem_classes=["clear-button"]
            )

        # 输出区域使用响应式布局
        with gr.Row(equal_height=True) as hemorrhagic_output:
            # 左侧：预测结果
            with gr.Column(scale=1, min_width=300):
                hemorrhagic_result = gr.HTML(
                    label=TRANSLATIONS["zh"]["output"]["prediction_result"],
                    visible=False,
                    elem_classes=["prediction-result"]
                )

            # 右侧：分析结果
            with gr.Column(scale=2, min_width=400):
                # 文本分析
                with gr.Group(visible=False) as hemorrhagic_text_analysis_group:
                    hemorrhagic_text_analysis_title = gr.Markdown(
                        TRANSLATIONS["zh"]["output"]["text_analysis"]["title"]
                    )
                    hemorrhagic_text_analysis = gr.HTML(
                        elem_classes=["text-analysis"]
                    )

                # SHAP图
                with gr.Group(visible=False) as hemorrhagic_shap_group:
                    hemorrhagic_shap_title = gr.Markdown(
                        TRANSLATIONS["zh"]["output"]["feature_importance"]["title"]
                    )
                    hemorrhagic_shap = gr.Image(
                        label="",
                        show_label=False,
                        visible=False,
                        elem_classes=["shap-plot"],
                        container=True
                    )

    # 修改出血性卒中的预测事件处理
    def process_hemorrhagic_prediction(lang_choice, text, *features):
        """处理出血性卒中预测"""
        lang = "en" if lang_choice == "English" else "zh"
        # 添加图例说明
        legend_html = f"""
        <div style="margin-top: 16px; padding: 12px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div style="display: flex; align-items: center; gap: 24px;">
                <div style="font-weight: 500; color: #334155;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["title"]}</div>
                <div style="display: flex; gap: 24px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(183, 28, 28, 0.9); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["extremely_high"]}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(239, 83, 80, 0.8); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["high"]}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(255, 152, 0, 0.8); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["moderate"]}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: rgba(158, 158, 158, 0.8); border-radius: 4px;"></span>
                        <span style="color: #475569;">{TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["legend"]["low"]}</span>
                    </div>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 0.9em; color: #64748b;">
                {TRANSLATIONS[lang]["output"]["text_analysis"]["legend_html"]["note"]}
            </div>
        </div>
        """
        try:
            # 将特征列表转换为正确的格式
            feature_list = list(features)

            # 检查是否有文本输入和特征输入
            has_text = bool(text and text.strip())
            has_features = any(feature_list)

            if not (has_text or has_features):
                raise ValueError(
                    TRANSLATIONS[lang]["output"]["prediction_error"])

            # 根据输入情况选择预测方式
            if has_text and not has_features:
                # 仅文本输入：使用文本模型
                text_html, prob = predict_hemorrhagic_text(text)
                text_html = f"""
                <div class="text-analysis-container">
                    <div class="text-content" style="line-height: 1.8; font-size: 1em; color: #1e293b;">
                        {text_html}
                    </div>
                    {legend_html}
                </div>
                """
                result_outputs = [
                    gr.update(value=text_html, visible=True),     # 文本分析结果
                    None,                                         # SHAP图
                    gr.update(visible=True),                      # 文本分析组
                    gr.update(visible=False)                      # SHAP组
                ]
            elif not has_text and has_features:
                # 仅特征输入：使用结构化模型
                results = predict_hemorrhagic_structured(feature_list)
                prob = results["probability"]
                result_outputs = [
                    gr.update(value=None, visible=False),         # 文本分析结果
                    gr.update(value=results["shap_plot"],
                              visible=True),  # SHAP图
                    gr.update(visible=False),                     # 文本分析组
                    gr.update(visible=True)                       # SHAP组
                ]
            else:
                # 同时输入：使用组合模型
                results = predict_hemorrhagic_combined(text, feature_list)
                text_html, _ = predict_hemorrhagic_text(text)
                text_html = f"""
                <div class="text-analysis-container">
                    <div class="text-content" style="line-height: 1.8; font-size: 1em; color: #1e293b;">
                        {text_html}
                    </div>
                    {legend_html}
                </div>
                """
                prob = results["probability"]
                result_outputs = [
                    gr.update(value=text_html, visible=True),     # 文本分析结果
                    gr.update(value=results["shap_plot"],
                              visible=True),  # SHAP图
                    gr.update(visible=True),                      # 文本分析组
                    gr.update(visible=True)                       # SHAP组
                ]

            # 生成预测结果HTML
            threshold = 0.3
            risk_level = TRANSLATIONS[lang]["output"]["high_risk"] if prob > threshold else TRANSLATIONS[lang]["output"]["low_risk"]
            risk_color = "#ef5350" if prob > threshold else "#4caf50"
            risk_note = TRANSLATIONS[lang]["output"]["hemorrhagic_high_risk_note"] if prob > threshold else TRANSLATIONS[lang]["output"]["hemorrhagic_low_risk_note"]

            result_html = f"""
            <div style="padding: 1em; border-radius: 0.5em; background: {risk_color}; color: white;">
                <h3 style="margin: 0">{TRANSLATIONS[lang]["output"]["prediction_result"]}: {risk_level}</h3>
                <p style="margin: 0.5em 0">{TRANSLATIONS[lang]["output"]["risk_probability"]}: {prob:.1%}</p>
                <p style="margin: 0.5em 0">{risk_note}</p>
            </div>
            """
            # 清理预测过程中的临时内存
            cleanup()
            # 返回所有输出
            return [
                result_outputs[0],                               # 文本分析结果
                gr.update(value=result_html, visible=True),      # 预测结果
                result_outputs[1],                               # SHAP图
                result_outputs[2],                               # 文本分析组
                result_outputs[3]                                # SHAP组
            ]
        except Exception as e:
            # 清理预测过程中的临时内存
            cleanup()
            print(f"Error in hemorrhagic prediction: {e}")
            return handle_prediction_error(e)

    hemorrhagic_predict.click(
        fn=process_hemorrhagic_prediction,
        inputs=[language] + [hemorrhagic_text] + hemorrhagic_features,
        outputs=[
            hemorrhagic_text_analysis,
            hemorrhagic_result,
            hemorrhagic_shap,
            hemorrhagic_text_analysis_group,
            hemorrhagic_shap_group
        ]
    )

    hemorrhagic_clear.click(
        fn=lambda: [
            None,  # 文本输入
            *[gr.update(value=False) for _ in hemorrhagic_features],  # 特征输入
            gr.update(value=None, visible=False),  # 文本分析
            gr.update(value=None, visible=False),  # 预测结果
            gr.update(value=None, visible=False),  # SHAP图
            gr.update(visible=False),  # 文本分析组
            gr.update(visible=False)   # SHAP组
        ],
        outputs=[hemorrhagic_text] + hemorrhagic_features + [
            hemorrhagic_text_analysis,
            hemorrhagic_result,
            hemorrhagic_shap,
            hemorrhagic_text_analysis_group,
            hemorrhagic_shap_group
        ]
    )

    # 修改示例按钮的语言切换处理
    def update_example_buttons(lang_choice):
        """更新示例按钮的文本"""
        lang = "en" if lang_choice == "English" else "zh"
        texts = TRANSLATIONS[lang]

        # 返回所有按钮的更新值
        updates = []
        for i in range(2):
            updates.extend([
                gr.update(value=f"{texts['example_button']} 1"),
                gr.update(value=f"{texts['example_button']} 2"),
                gr.update(value=f"{texts['example_button']} 3")
            ])
        return updates

    # 绑定示例按钮的语言切换事件
    language.change(
        fn=update_example_buttons,
        inputs=[language],
        outputs=ischemic_example_buttons + hemorrhagic_example_buttons  # 所有示例按钮
    )

    # 在语言切换事件之后添加任务切换功能

    # 修改任务切换函数
    def switch_task(choice):
        """任务切换处函数"""
        # 检查中英文任务名称
        is_ischemic = (
            # 中文
            choice == TRANSLATIONS["zh"]["task_selector"]["choices"][0] or
            # 英文
            choice == TRANSLATIONS["en"]["task_selector"]["choices"][0]
        )

        if is_ischemic:  # 缺血性卒中
            return [
                gr.update(visible=True),   # 显示缺血性卒中界面
                gr.update(visible=False)   # 隐藏出血性卒中界面
            ]
        else:  # 出血性卒中
            return [
                gr.update(visible=False),  # 隐藏缺血性卒中界面
                gr.update(visible=True)    # 显示出血性卒中界面
            ]

    # 绑定任务切换事件
    task_selector.change(
        fn=switch_task,
        inputs=[task_selector],
        outputs=[
            ischemic_interface,
            hemorrhagic_interface
        ]
    )

    # 修改语言切换函数
    def switch_language(lang_choice):
        """语言切换处理函数"""
        lang = "en" if lang_choice == "English" else "zh"
        texts = load_language(lang)  # 使用语言文件
        help_text = help_content_en if lang == "en" else help_content_zh

        # 添加帮助文档标题的翻译
        help_titles = {
            "zh": "📖 使用指南",
            "en": "📖 User Guide"
        }

        return [
            # 更新标题和说明
            f"""
            <div class="header" style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">{texts['title']}</h1>
                <p style="font-size: 1.1rem; color: var(--secondary-color);">
                    {texts['subtitle']}
                </p>
            </div>
            """,
            # 更新任务选择器
            gr.update(
                choices=texts["task_selector"]["choices"],
                value=texts["task_selector"]["choices"][0],
                label=texts["task_selector"]["label"],
                info=texts["task_selector"]["info"]
            ),
            # 更新缺血性卒中界面文本输入框
            gr.update(
                label=texts["text_input"]["label"],
                placeholder=texts["text_input"]["placeholder"]
            ),
            # 更新缺血性卒中界面输入指南
            gr.update(value=texts["input_guide"]),
            # 更新缺血性卒中临床指标标题
            gr.update(value=texts["clinical_indicators"]),
            # 更新缺血性卒中预测和清除按钮
            gr.update(value=texts["predict_button"]),
            gr.update(value=texts["clear_button"]),
            # 更新缺血性卒中文本分析标题
            gr.update(value=texts["output"]["text_analysis"]["title"]),
            # 更新缺血性卒中SHAP图标题
            gr.update(value=texts["output"]["feature_importance"]["title"]),
            # 更新出血性卒中界面文本输入框
            gr.update(
                label=texts["text_input"]["label"],
                placeholder=texts["text_input"]["placeholder"]
            ),
            # 更新出血性卒中界面输入指南
            gr.update(value=texts["input_guide"]),
            # 更新出血性卒中临床指标标题
            gr.update(value=texts["clinical_indicators"]),
            # 更新出血性卒中预测和清除按钮
            gr.update(value=texts["predict_button"]),
            gr.update(value=texts["clear_button"]),
            # 更新出血性卒中文本分析标题
            gr.update(value=texts["output"]["text_analysis"]["title"]),
            # 更新出血性卒中SHAP图标题
            gr.update(value=texts["output"]["feature_importance"]["title"]),
            # 更新帮助文档内容
            gr.update(value=help_text),
            # 更新帮助文档标题
            gr.update(label=help_titles[lang]),
        ]

    # 绑定语言切换事件
    language.change(
        fn=switch_language,
        inputs=[language],
        outputs=[
            title_html,                    # 标题和说明
            task_selector,                 # 任务选择器
            ischemic_text,                 # 缺血性卒中文本输入
            ischemic_input_guide,          # 缺血性卒中输入指南
            ischemic_indicators_title,     # 缺血性卒中指标标题
            ischemic_predict,              # 缺血性卒中预测按钮
            ischemic_clear,                # 缺血性卒中清除按钮
            ischemic_text_analysis_title,  # 缺血性卒中文本分析标题
            ischemic_shap_title,           # 缺血性卒中SHAP图标题
            hemorrhagic_text,              # 出血性卒中文输入
            hemorrhagic_input_guide,       # 出血性卒中输入指南
            hemorrhagic_indicators_title,  # 出血性卒中指标标题
            hemorrhagic_predict,           # 出血性卒中预测按钮
            hemorrhagic_clear,             # 出血性卒中清除按钮
            hemorrhagic_text_analysis_title,  # 出血性卒中文本分析标题
            hemorrhagic_shap_title,           # 出血性卒中SHAP图标题
            help_content_markdown,         # 帮助文档内容
            help_accordion,                # 帮助文档标题
        ]
    )

    # 在最后添加备案信息
    gr.HTML(
        """
        <div class="footer" style="
            text-align: center;
            padding: 1rem;
            margin-top: 2rem;
            border-top: 1px solid var(--border-color-primary);
            color: var(--body-text-color);
            opacity: 0.8;
            font-size: 0.9rem;
        ">
            <a href="https://beian.mps.gov.cn/#/query/webSearch" 
               target="_blank" 
               style="text-decoration: none; color: inherit; margin-right: 1rem;">
                <img src="https://www.beian.gov.cn/img/new/gongan.png" 
                     style="vertical-align: middle; margin-right: 3px;">
                苏公网安备 32010602011293号
            </a>
            <a href="https://beian.miit.gov.cn/#/Integrated/index" 
               target="_blank" 
               style="text-decoration: none; color: inherit;">
                苏ICP备2023023603
            </a>
        </div>
        """
    )

# 在启动前检查所有必需的模型文件


def check_model_files():
    model_paths = {
        'ischemic': {
            'text': 'models/ischemic/text/ais_baseline_macbertnewend2cnn_3_1time_epoch3.pth',
            'structured': 'models/ischemic/structured/ais_SoftVoting_6_mice1.pkl',
            'combined': 'models/ischemic/combined/ais_SoftVoting_7_mice1.pkl'
        },
        'hemorrhagic': {
            'text': 'models/hemorrhagic/text/ich_baseline_macbertnewend1cnn_1time_epoch3.pth',
            'structured': 'models/hemorrhagic/structured/ich_SoftVoting_7_mice1.pkl',
            'combined': 'models/hemorrhagic/combined/ich_SoftVoting_8_mice1.pkl'
        }
    }

    missing_files = []
    for stroke_type, models in model_paths.items():
        for model_type, path in models.items():
            full_path = os.path.join(prefix, path)
            if not os.path.exists(full_path):
                missing_files.append(f"{stroke_type}/{model_type}: {path}")

    if missing_files:
        raise FileNotFoundError(
            f"Missing model files:\n" + "\n".join(missing_files))


def preload_models():
    """预加载和预热所有模型"""
    logger = Logger("model_preload").get_logger()
    try:
        logger.info("Preloading models...")
        
        # 使用轻量级配置
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
            
        # 初始化jieba分词器
        print("Initializing jieba...")
        init_jieba()

        # 加载缺血性卒中模型
        print("Loading ischemic stroke models...")
        # 文本模型
        _, _, _ = get_ischemic_model()
        # 结构化模型
        _, _, _ = get_ischemic_structured_model()
        # 组合模型
        _, _, _ = get_ischemic_combined_model()

        # 加载出血性卒中模型
        print("Loading hemorrhagic stroke models...")
        # 文本模型
        _, _, _ = get_hemorrhagic_text_model()
        # 结构化模型
        _, _, _ = get_hemorrhagic_structured_model()
        # 组合模型
        _, _, _ = get_hemorrhagic_combined_model()

        # 清理加载过程中的临时内存
        cleanup()
        logger.info("All models loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error preloading models: {str(e)}")
        cleanup()
        return False


# 启动应用
if __name__ == "__main__":
    try:
        # 检查模型文件
        check_model_files()

        # 预加载模型
        if not preload_models():
            raise Exception("Failed to preload models")

        # 启动服务
        demo.queue(
            max_size=20,         # 限制队列长度
            api_open=False       # 关闭API访问以减少负载
        )
        demo.launch(
            server_name="0.0.0.0",
            server_port=8080,  # 使用动态端口
            share=False,
            show_error=False,
            max_threads=10,
        )
    except Exception as e:
        print(f"Error starting server: {e}")
