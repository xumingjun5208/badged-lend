# 测试数据
ISCHEMIC_TEST_CASES = [
    {
        "text": "患者，男，68岁，因'突发意识障碍4小时'入院。患者4小时前无明显诱因突发意识障碍，言语不能，右侧肢体活动障碍。查体：神志昏睡，BP 175/95mmHg，心率80次/分，双瞳等大同圆，对光反射迟钝，右侧肢体肌力2级。",
        "features": {
            "Dysphagia": 1,
            "Endotracheal intubation": 0,
            "Decubitus ulcer": 0,
            "Age": 68,
            "Number of resuscitations": 1,
            "Lung disease": 1
        },
        "expected_risk": "high"
    }
]

HEMORRHAGIC_TEST_CASES = [
    {
        "text": "患者，男，58岁，因'头痛、呕吐后意识丧失3小时'入院。患者3小时前突发剧烈头痛，随后出现呕吐3次，后逐渐出现意识障碍。查体：昏迷，GCS 6分，BP 190/110mmHg，双瞳不等大，右侧瞳孔散大。",
        "features": {
            "Dysphagia": 1,
            "Endotracheal intubation": 1,
            "Decubitus ulcer": 0,
            "Hydrocephalus": 1,
            "Brain hernia": 0,
            "Hyperleukocytosis": 1,
            "Gastrointestinal bleeding": 0
        },
        "expected_risk": "high"
    }
] 