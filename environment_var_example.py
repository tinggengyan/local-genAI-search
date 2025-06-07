"""
环境变量配置文件
用于存储系统配置信息
"""

# HuggingFace API Token
# 获取地址：https://huggingface.co/settings/tokens
hf_token = ""  # 替换为你的 token

# NVIDIA API Key (不再需要，但保留变量以保持兼容性)
nvidia_key = ""

# Qdrant 配置
qdrant_url = "http://localhost:6333"  # Qdrant 服务地址
qdrant_api_key = ""  # 如果设置了认证，请填写 API key

# 模型配置，用于文本嵌入（text embedding）的模型。这个模型的主要作用是将文本转换为向量（embeddings），用于后续的相似度搜索。
# BGE-small-zh 是一个专门针对中文优化的嵌入模型，它能够将文本转换为高质量的向量表示，这对于实现语义搜索功能非常重要。
model_name = "BAAI/bge-small-zh-v1.5"  # 使用的嵌入模型
collection_name = "MyCollection"  # Qdrant 集合名称

# 本地模型配置，用于生成文本的大语言模型（LLM）。这个模型是 Mistral-7B 的量化版本，用于实际的文本生成任务，比如回答问题、生成内容等。
# GGUF 格式是经过优化的模型格式，可以在本地运行。
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = f"models/{MODEL_FILE}"

# 系统配置
MAX_RETRIES = 3  # API 调用最大重试次数
RETRY_DELAY = 1  # 重试延迟（秒）
CHUNK_SIZE = 500  # 文本分块大小
CHUNK_OVERLAP = 50  # 文本分块重叠大小
MAX_SEARCH_RESULTS = 10  # 最大搜索结果数

# 验证必要的环境变量
if not hf_token:
    print("警告: 未设置 HF_TOKEN 环境变量，某些功能可能无法正常工作")
if not qdrant_url:
    print("警告: 未设置 QDRANT_URL 环境变量，将使用默认值 http://localhost:6333") 