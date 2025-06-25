# 主程序文件内容（简化版）
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 初始化Chroma向量存储
chroma_db = Chroma(persist_directory="./chroma", embedding_function=embeddings)

# 其他核心功能代码...