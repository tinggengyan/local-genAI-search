"""
API 服务模块
提供 FastAPI 接口服务
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from config import LOG_LEVEL, LOG_FORMAT
from model_manager import ModelManager
from cache_manager import CacheManager
from search_manager import SearchManager

# 配置日志
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# 初始化管理器
model_manager = ModelManager()
cache_manager = CacheManager()
search_manager = SearchManager()

# 创建 FastAPI 应用
app = FastAPI(title="Local GenAI Search API")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    """查询请求模型"""
    query: str

@app.get("/")
async def root():
    """根路径处理"""
    return {"message": "Local GenAI Search API is running"}

@app.post("/search")
async def search(query: Query) -> List[Dict[str, Any]]:
    """
    搜索接口
    
    Args:
        query: 查询请求
        
    Returns:
        List[Dict[str, Any]]: 搜索结果列表
    """
    try:
        logger.info(f"收到搜索请求: {query.query}")
        results = search_manager.search(query.query)
        return results
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_localai")
async def ask_localai(query: Query) -> Dict[str, Any]:
    """
    问答接口
    
    Args:
        query: 查询请求
        
    Returns:
        Dict[str, Any]: 包含答案和上下文的响应
    """
    try:
        logger.info(f"收到问答请求: {query.query}")
        
        # 检查缓存
        if cache_manager.enabled:
            cached_response = cache_manager.get(query.query)
            if cached_response:
                logger.info("使用缓存响应")
                return cached_response
        
        # 搜索相关文档
        search_results = search_manager.search(query.query)
        if not search_results:
            return {"context": [], "answer": "未找到相关文档。"}
        
        # 构建上下文
        context = ""
        for i, res in enumerate(search_results):
            context += f"[{i}] {res['content']}\n\n"
        
        # 构建提示词
        prompt = f"""请基于以下文档内容回答用户的问题。回答时请引用文档编号（例如[0]，[1]）来支持你的观点。\n\n示例：\n问题：Android 如何管理内存？\n回答：Android 通过多种机制管理内存，例如垃圾回收 [0]，以及内存分配优化 [1]。\n\n文档内容：\n{context}\n用户问题：{query.query}\n\n请提供详细的回答，并适当引用文档。"""
        
        # 生成回答
        answer = model_manager.generate_response(prompt)
        
        # 构建响应
        response = {
            "context": search_results,
            "answer": answer
        }
        
        # 缓存响应
        if cache_manager.enabled:
            cache_manager.set(query.query, response)
        
        return response
        
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查接口"""
    return {
        "status": "healthy",
        "cache_enabled": cache_manager.enabled,
        "cache_size": cache_manager.size,
        "search_ready": True,  # 可以添加更多检查
        "model_ready": True
    }
