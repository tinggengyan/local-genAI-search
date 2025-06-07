"""
系统配置文件
集中管理所有配置参数
"""
import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# API 密钥配置
HF_TOKEN = os.getenv("HF_TOKEN", "")  # HuggingFace API Token
NVIDIA_KEY = os.getenv("NVIDIA_KEY", "")  # NVIDIA API Key (保留以保持兼容性)

# Qdrant 配置
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "MyCollection"

# 模型配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 嵌入模型
LLM_MODEL_NAME = "Qwen/Qwen-7B-Chat"  # 大语言模型
LLM_MODEL_FILE = "qwen-7B-Chat.Q4_K_M.gguf"  # 模型文件名
LLM_MODEL_PATH = str(MODELS_DIR / LLM_MODEL_FILE)

# 系统配置
MAX_RETRIES = 3  # API 调用最大重试次数
RETRY_DELAY = 1  # 重试延迟（秒）
CHUNK_SIZE = 300  # 文本分块大小
CHUNK_OVERLAP = 30  # 文本分块重叠大小
MAX_SEARCH_RESULTS = 5  # 最大搜索结果数

# 缓存配置
ENABLE_CACHE = True  # 启用结果缓存
CACHE_EXPIRY = 3600  # 缓存过期时间（秒）

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 验证必要的环境变量
if not HF_TOKEN:
    print("警告: 未设置 HF_TOKEN 环境变量，某些功能可能无法正常工作")
if not QDRANT_URL:
    print("警告: 未设置 QDRANT_URL 环境变量，将使用默认值 http://localhost:6333")

# 导出所有配置变量
__all__ = [
    'BASE_DIR', 'MODELS_DIR',
    'HF_TOKEN', 'NVIDIA_KEY',
    'QDRANT_URL', 'QDRANT_API_KEY', 'COLLECTION_NAME',
    'EMBEDDING_MODEL', 'LLM_MODEL_NAME', 'LLM_MODEL_FILE', 'LLM_MODEL_PATH',
    'MAX_RETRIES', 'RETRY_DELAY', 'CHUNK_SIZE', 'CHUNK_OVERLAP', 'MAX_SEARCH_RESULTS',
    'ENABLE_CACHE', 'CACHE_EXPIRY',
    'LOG_LEVEL', 'LOG_FORMAT'
] 