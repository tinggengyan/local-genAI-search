"""
搜索管理器模块
负责文档搜索和向量存储管理
"""
import logging
import torch
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import (
    EMBEDDING_MODEL, QDRANT_URL, QDRANT_API_KEY, 
    COLLECTION_NAME, MAX_SEARCH_RESULTS
)

logger = logging.getLogger(__name__)

class SearchManager:
    _instance: Optional['SearchManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._initialize_search()
    
    def _initialize_search(self):
        """初始化搜索组件"""
        try:
            # 初始化嵌入模型
            if torch.cuda.is_available():
                model_kwargs = {'device': 'cuda'}
            elif torch.backends.mps.is_available():
                model_kwargs = {'device': 'mps'}
            else:
                model_kwargs = {'device': 'cpu'}
                
            encode_kwargs = {'normalize_embeddings': True}
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # 初始化 Qdrant 客户端
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # 确保集合存在
            if not self.client.collection_exists(COLLECTION_NAME):
                logger.info(f"创建集合: {COLLECTION_NAME}")
                self.client.create_collection(
                    COLLECTION_NAME,
                    vectors_config=VectorParams(size=512, distance=Distance.DOT)
                )
            
            # 初始化向量存储
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embeddings=self.embeddings
            )
            
            logger.info("搜索组件初始化完成")
            
        except Exception as e:
            logger.error(f"搜索组件初始化失败: {str(e)}")
            raise
    
    def search(self, query: str, k: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        try:
            search_results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            results = []
            for i, res in enumerate(search_results):
                results.append({
                    "id": i,
                    "path": res.metadata.get("path"),
                    "content": res.page_content
                })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, str]]) -> None:
        """
        添加文档到向量存储
        
        Args:
            texts: 文档文本列表
            metadatas: 文档元数据列表
        """
        try:
            self.vector_store.add_texts(texts, metadatas=metadatas)
            logger.info(f"成功添加 {len(texts)} 个文档")
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise
    
    def clear_collection(self) -> None:
        """清空集合"""
        try:
            if self.client.collection_exists(COLLECTION_NAME):
                self.client.delete_collection(COLLECTION_NAME)
                self.client.create_collection(
                    COLLECTION_NAME,
                    vectors_config=VectorParams(size=512, distance=Distance.DOT)
                )
                logger.info(f"集合 {COLLECTION_NAME} 已清空")
        except Exception as e:
            logger.error(f"清空集合失败: {str(e)}")
            raise 