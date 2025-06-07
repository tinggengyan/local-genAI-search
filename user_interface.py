import streamlit as st

# 配置页面 - 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="Local GenAI Search",
    page_icon="🔍",
    layout="wide"
)

import re
import requests
import json
from requests.exceptions import RequestException
import time

st.title('_:blue[Local GenAI Search]_ :sunglasses:')

# 添加错误处理装饰器
def handle_errors(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1  # 秒
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RequestException as e:
                if attempt < max_retries - 1:
                    st.warning(f"连接服务器失败，正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                st.error(f"无法连接到服务器: {str(e)}")
                st.info("请确保后端服务正在运行 (python uvicorn_start.py)")
            except json.JSONDecodeError as e:
                st.error(f"服务器返回的数据格式错误: {str(e)}")
                st.info("请检查后端服务日志")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
            break
    return wrapper

@handle_errors
def process_query(question):
    if not question:
        return
        
    st.write("正在处理问题: \"", question+"\"")
    
    url = "http://127.0.0.1:8000/ask_localai"
    payload = json.dumps({"query": question})
    headers = {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    }

    with st.spinner('正在搜索和生成答案...'):
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()
        
        data = response.json()
        if "detail" in data:
            st.error(f"API 错误: {data['detail']}")
            return
            
        answer = data["answer"]
        documents = data['context']
        
        # 显示答案
        st.markdown("### 答案")
        st.markdown(answer)
        
        # 提取引用的文档编号（支持多种格式）
        doc_patterns = [
            r'\[Document\s*(\d+)\]',  # [Document 1]
            r'\[(\d+)\]',             # [1]
            r'文档\s*(\d+)',          # 文档 1
            r'引用\s*(\d+)'           # 引用 1
        ]
        
        num = set()  # 使用集合去重
        for pattern in doc_patterns:
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            num.update(int(m.group(1)) for m in matches)
        
        # 显示引用的文档
        if num:
            st.markdown("### 引用的文档")
            show_docs = []
            for n in sorted(num):  # 按文档编号排序
                for doc in documents:
                    if int(doc['id']) == n:
                        show_docs.append(doc)
            
            for doc in show_docs:
                with st.expander(f"文档 {doc['id']} - {doc['path']}"):
                    st.write(doc['content'])
                    try:
                        with open(doc['path'], 'rb') as f:
                            st.download_button(
                                "下载文件",
                                f,
                                file_name=doc['path'].split('/')[-1],
                                key=f"download_{doc['id']}",
                                help=f"下载文档 {doc['id']} 的原始文件"
                            )
                    except Exception as e:
                        st.warning(f"无法下载文件: {str(e)}")
        else:
            st.info("答案中没有引用任何文档")

# 主界面
st.markdown("""
### 使用说明
1. 在下方输入框中输入你的问题
2. 系统会搜索相关文档并生成答案
3. 答案中引用的文档会在下方显示
4. 可以点击展开查看文档内容，或下载原始文件
""")

question = st.text_input("请输入你的问题", "", help="输入你想问的问题，系统会搜索相关文档并生成答案")
if st.button("提问", help="点击提交问题") or question:
    process_query(question)
