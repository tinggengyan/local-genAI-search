from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#import qdrant_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_core.messages import HumanMessage
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import environment_var
import os
from openai import OpenAI
#from langgraph.graph import END, MessageGraph
import logging
from typing import List, Dict, Any
from environment_var import hf_token, qdrant_url, qdrant_api_key, model_name, collection_name, CHUNK_SIZE, CHUNK_OVERLAP, MAX_SEARCH_RESULTS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Item(BaseModel):
    query: str
    def __init__(self, query: str) -> None:
        super().__init__(query=query)

# 初始化 embeddings
if torch.cuda.is_available():
    model_kwargs = {'device': 'cuda'}
elif torch.backends.mps.is_available():
    model_kwargs = {'device': 'mps'}
else:
    model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 初始化 Qdrant 客户端
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# 初始化向量存储（LangChain Qdrant）
qdrant = Qdrant(qdrant_client, collection_name, hf)

app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 LLM 模型
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# 初始化 tokenizer（用于估算 token 数）
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
except Exception:
    tokenizer = None

def count_tokens(text):
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text) // 2  # 粗略估算

try:
    from llama_cpp import Llama
    
    # 初始化 Llama 模型
    llm = Llama(
        model_path=f"models/{MODEL_FILE}",
        n_ctx=2048,  # 上下文窗口大小
        n_threads=8,  # 使用 8 个线程
        n_gpu_layers=0  # 在 M2 上使用 CPU 模式
    )
    
    def generate_response(prompt: str) -> str:
        """使用 Llama 模型生成回复"""
        try:
            # 构建完整的提示
            full_prompt = f"""<s>[INST] {prompt} [/INST]"""
            
            # 生成回复
            response = llm(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                stop=["</s>", "[INST]"]
            )
            
            # 重置模型状态
            llm.reset()
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")
            raise HTTPException(status_code=500, detail=f"生成回复失败: {str(e)}")
            
except ImportError:
    logger.error("请先安装 llama-cpp-python: pip install llama-cpp-python")
    raise

# 检查 Qdrant 服务是否初始化
try:
    collections = qdrant_client.get_collections()
    if not collections.collections:
        raise HTTPException(status_code=500, detail="Qdrant 服务未初始化，请先运行索引程序")
except Exception as e:
    logger.error(f"Qdrant 服务检查失败: {str(e)}")
    raise HTTPException(status_code=500, detail="Qdrant 服务未就绪")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search")
async def search(Item:Item):
    try:
        query = Item.query
        logger.info(f"Received search query: {query}")
        search_result = qdrant.similarity_search(
            query=query, k=10
        )
        i = 0
        list_res = []
        for res in search_result:
            list_res.append({"id":i,"path":res.metadata.get("path"),"content":res.page_content})
        return list_res
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_localai")
async def ask_localai(Item:Item):
    try:
        query = Item.query
        logger.info(f"Received ask_localai query: {query}")
        
        # 检查 qdrant 是否已初始化
        if not qdrant:
            raise HTTPException(status_code=500, detail="Search index not initialized")
            
        search_result = qdrant.similarity_search(
            query=query, k=10
        )
        
        if not search_result:
            return {"context": [], "answer": "No relevant documents found."}
            
        i = 0
        list_res = []
        context = ""
        mappings = {}
        max_context_tokens = 1500
        current_tokens = 0
        for res in search_result:
            chunk = str(i)+"\n"+res.page_content+"\n\n"
            chunk_tokens = count_tokens(chunk)
            if current_tokens + chunk_tokens > max_context_tokens:
                break
            context += chunk
            current_tokens += chunk_tokens
            mappings[i] = res.metadata.get("path")
            list_res.append({"id":i,"path":res.metadata.get("path"),"content":res.page_content})
            i += 1

        prompt = f"""You are a helpful AI assistant. Please answer the user's question using the provided documents as context.\nAlways reference document IDs in square brackets (e.g., [0], [1]) when making claims based on the documents.\nUse as many citations as necessary to support your answer.\n\nDocuments:\n{context}\n\nQuestion: {query}\n\nPlease provide a detailed answer with proper citations."""

        try:
            response = generate_response(prompt)
            return {"context": list_res, "answer": response}
        except Exception as e:
            logger.error(f"Error in model generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in ask_localai: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
