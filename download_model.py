"""
模型下载工具
支持从 HuggingFace 下载各种格式的模型文件
"""
import os
import json
import requests
import re
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from config import (
    LLM_MODEL_NAME, LLM_MODEL_PATH, EMBEDDING_MODEL,
    HF_TOKEN, CHUNK_SIZE, CHUNK_OVERLAP
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, token: Optional[str] = None):
        """初始化下载器"""
        self.api = HfApi(token=token)
        self.token = token
    
    def _normalize_model_name(self, model_name: str) -> str:
        """
        标准化模型名称，移除版本号等后缀
        
        Args:
            model_name: 原始模型名称
            
        Returns:
            str: 标准化后的模型名称
        """
        # 移除版本号（如 -v1.5, -v2 等）
        model_name = re.sub(r'-v\d+(\.\d+)*$', '', model_name)
        # 移除其他常见后缀
        model_name = re.sub(r'-(base|large|small|tiny|mini)$', '', model_name)
        return model_name
    
    def _get_possible_repos(self, org: str, model: str, base_model: str) -> List[str]:
        """
        获取可能的仓库名称列表
        
        Args:
            org: 原始组织名
            model: 原始模型名
            base_model: 标准化后的模型名
            
        Returns:
            List[str]: 可能的仓库名称列表
        """
        repos = []
        
        # 原始仓库
        repos.append(f"{org}/{model}")
        
        # TheBloke 的 GGUF 版本
        repos.extend([
            f"TheBloke/{base_model}-GGUF",
            f"TheBloke/{model}-GGUF",
            f"TheBloke/{base_model}-7B-GGUF",
            f"TheBloke/{base_model}-13B-GGUF",
            f"TheBloke/{base_model}-Chat-GGUF"
        ])
        
        # 原始组织的 GGUF 版本
        repos.extend([
            f"{org}/{base_model}-GGUF",
            f"{org}/{model}-GGUF",
            f"{org}/{base_model}-7B-GGUF",
            f"{org}/{base_model}-Chat-GGUF"
        ])
        
        # 特殊处理 Qwen 模型
        if "qwen" in model.lower():
            repos.extend([
                "Qwen/Qwen-7B-Chat-GGUF",
                "Qwen/Qwen-7B-GGUF",
                "Qwen/Qwen-Chat-GGUF"
            ])
        
        return list(set(repos))  # 去重
    
    def get_model_info(self, model_name: str) -> Tuple[str, str]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称，格式为 "org/model-name"
            
        Returns:
            Tuple[str, str]: (仓库名, 模型名)
        """
        # 检查是否是完整仓库名
        if "/" in model_name:
            parts = model_name.split("/")
            if len(parts) != 2:
                raise ValueError(f"无效的模型名称格式: {model_name}")
            org, model = parts
        else:
            raise ValueError(f"无效的模型名称格式: {model_name}")
        
        # 标准化模型名称
        base_model = self._normalize_model_name(model)
        
        # 获取可能的仓库列表
        possible_repos = self._get_possible_repos(org, model, base_model)
        
        # 尝试每个可能的仓库名
        for repo in possible_repos:
            try:
                files = self.api.list_repo_files(repo)
                if files:  # 确保仓库有文件
                    logger.info(f"找到可用仓库: {repo}")
                    return repo.split("/")
            except Exception:
                continue
        
        # 如果都找不到，使用原始模型
        logger.info(f"未找到 GGUF 版本，使用原始模型: {model_name}")
        return org, model
    
    def list_model_files(self, repo_id: str, model_file: Optional[str] = None) -> List[str]:
        """
        列出模型仓库中的文件
        
        Args:
            repo_id: 仓库ID
            model_file: 可选的特定文件名
            
        Returns:
            List[str]: 文件列表
        """
        try:
            files = list_repo_files(repo_id, token=self.token)
            if model_file:
                # 如果指定了文件名，进行模糊匹配
                # 将文件名转换为正则表达式模式
                pattern = model_file.lower()
                pattern = pattern.replace(".", "\\.").replace("*", ".*")
                # 添加常见的量化后缀
                pattern = f"{pattern}|{pattern.replace('.gguf', '.*.gguf')}"
                pattern = re.compile(pattern, re.IGNORECASE)
                return [f for f in files if pattern.search(f.lower())]
            return files
        except Exception as e:
            logger.error(f"获取文件列表失败: {str(e)}")
            return []
    
    def download_file(self, repo_id: str, filename: str, local_dir: str = "models") -> Optional[str]:
        """
        下载单个文件
        
        Args:
            repo_id: 仓库ID
            filename: 文件名
            local_dir: 本地保存目录
            
        Returns:
            Optional[str]: 下载的文件路径，如果失败则返回 None
        """
        try:
            # 确保目录存在
            os.makedirs(local_dir, exist_ok=True)
            
            # 下载文件
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=self.token,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"成功下载: {filename}")
            return local_path
            
        except Exception as e:
            logger.error(f"下载文件 {filename} 失败: {str(e)}")
            return None
    
    def download_model(self, model_name: str, model_file: Optional[str] = None) -> bool:
        """
        下载模型
        
        Args:
            model_name: 模型名称
            model_file: 可选的特定文件名
            
        Returns:
            bool: 是否下载成功
        """
        try:
            # 获取模型信息
            org, model = self.get_model_info(model_name)
            repo_id = f"{org}/{model}"
            
            # 获取文件列表
            files = self.list_model_files(repo_id, model_file)
            if not files:
                logger.error(f"未找到可下载的文件")
                return False
            
            # 打印找到的文件列表
            logger.info(f"找到以下文件:")
            for f in files:
                logger.info(f"  - {f}")
            
            # 下载文件
            success = True
            for file in files:
                if not self.download_file(repo_id, file):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"下载模型失败: {str(e)}")
            return False

def download_file(url: str, local_path: str, headers: Optional[dict] = None) -> bool:
    """Download a file with progress bar."""
    try:
        if headers is None:
            headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
            
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """Main function to download models."""
    # 确保模型目录存在
    os.makedirs(LLM_MODEL_PATH, exist_ok=True)
    
    # 下载大语言模型
    if not download_llm_model():
        logging.error("Failed to download LLM model")
        sys.exit(1)
        
    # 下载嵌入模型
    if not download_embedding_model():
        logging.error("Failed to download embedding model")
        sys.exit(1)
        
    logging.info("All models downloaded successfully!")

if __name__ == "__main__":
    main() 