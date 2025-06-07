import os
import requests
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, filename: str):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # 确保 models 目录存在
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # 模型文件信息
    model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    model_filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # 检查文件是否已存在
    if os.path.exists(os.path.join('models', model_filename)):
        logger.info(f"模型文件 {model_filename} 已存在，跳过下载")
        return
    
    logger.info(f"开始下载模型文件: {model_filename}")
    try:
        download_file(model_url, model_filename)
        logger.info("模型下载完成")
    except Exception as e:
        logger.error(f"下载模型时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 