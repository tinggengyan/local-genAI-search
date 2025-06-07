"""
模型管理器模块
负责模型的加载、初始化和推理
"""
import logging
import psutil
from typing import Optional
from llama_cpp import Llama
from config import LLM_MODEL_PATH, LLM_MODEL_NAME

logger = logging.getLogger(__name__)

class ModelManager:
    _instance: Optional['ModelManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._model: Optional[Llama] = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            # 获取系统资源信息
            cpu_count = psutil.cpu_count(logical=False)
            memory = psutil.virtual_memory()
            
            # 根据系统资源动态调整参数
            n_threads = max(1, cpu_count - 1)  # 保留一个核心给系统
            n_ctx = min(2048, int(memory.available * 0.3 / 1024 / 1024))  # 使用30%可用内存
            
            logger.info(f"初始化模型 {LLM_MODEL_NAME}，使用 {n_threads} 个线程，上下文窗口大小 {n_ctx}")
            
            # 初始化模型
            self._model = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=512,
                n_gpu_layers=0,
                verbose=False
            )
            
            logger.info("模型初始化完成")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.95) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            
        Returns:
            str: 生成的回复文本
        """
        if not self._model:
            raise RuntimeError("模型未初始化")
            
        try:
            # 构建完整的提示词
            full_prompt = f"""<|im_start|>system
你是一个专业的 Android AI 开发助手。请使用中文回答用户的问题，并基于提供的文档内容进行回答。

重要规则：
1. 你必须引用文档编号（例如[0]，[1]）来支持你的观点，否则回答无效。
2. 每个主要观点都必须有至少一个文档引用，例如"Android 通过垃圾回收 [0] 以及内存分配优化 [1] 管理内存"。
3. 如果没有找到相关文档，请明确说明"未找到相关文档"。
4. 请确保回答准确、专业，并且完全使用中文。
5. 不要编造或推测文档中没有的信息。

<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant"""
            
            # 生成回复
            response = self._model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")
            raise
    
    @property
    def model(self) -> Llama:
        """获取模型实例"""
        if not self._model:
            raise RuntimeError("模型未初始化")
        return self._model 