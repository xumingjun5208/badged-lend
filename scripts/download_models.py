import os
import sys
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def download_from_hf(repo_id: str, filename: str, local_dir: str):
    """从HuggingFace下载文件"""
    try:
        print(f"Downloading {filename}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded {filename}")
        return local_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def main():
    """主函数"""
    # 创建模型目录
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # HuggingFace仓库信息
    repo_id = "donghao1234/badged-lend"
    
    # 模型文件列表
    model_files = {
        'ischemic': {
            'text': 'ais_baseline_macbertnewend2cnn_3_1time_epoch3.pth',
            'structured': 'ais_SoftVoting_6_mice1.pkl',
            'combined': 'ais_SoftVoting_7_mice1.pkl'
        },
        'hemorrhagic': {
            'text': 'ich_baseline_macbertnewend1cnn_1time_epoch3.pth',
            'structured': 'ich_SoftVoting_7_mice1.pkl',
            'combined': 'ich_SoftVoting_8_mice1.pkl'
        },
        'macbert': {
            'config.json': 'config.json',
            'pytorch_model.bin': 'pytorch_model.bin',
            'vocab.txt': 'vocab.txt',
            'tokenizer_config.json': 'tokenizer_config.json'
        }
    }
    
    # 下载模型文件
    for model_type, files in model_files.items():
        type_dir = os.path.join(models_dir, model_type)
        os.makedirs(type_dir, exist_ok=True)
        
        if model_type == 'macbert':
            # MacBERT文件直接放在macbert目录下
            for filename in files:
                filepath = os.path.join(type_dir, filename)
                if not os.path.exists(filepath):
                    downloaded_path = download_from_hf(repo_id, f"macbert/{filename}", type_dir)
                    if downloaded_path and downloaded_path != filepath:
                        os.rename(downloaded_path, filepath)
                else:
                    print(f"{filename} already exists, skipping...")
        else:
            # 其他模型文件按类型分目录
            for model_subtype, filename in files.items():
                model_dir = os.path.join(type_dir, model_subtype)
                os.makedirs(model_dir, exist_ok=True)
                
                filepath = os.path.join(model_dir, filename)
                if not os.path.exists(filepath):
                    downloaded_path = download_from_hf(repo_id, filename, model_dir)
                    if downloaded_path and downloaded_path != filepath:
                        os.rename(downloaded_path, filepath)
                else:
                    print(f"{filename} already exists, skipping...")

if __name__ == "__main__":
    main() 