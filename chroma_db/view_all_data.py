import chromadb
from chromadb.config import Settings

# 初始化Chroma客户端
client = chromadb.Client(Settings(
    persist_directory="./chroma",
    anonymized_telemetry=False
))

# 获取指定集合
collection = client.get_collection(name="your_collection_name")

# 获取集合中的所有数据
results = collection.get()

# 打印数据统计
print(f"找到 {len(results['ids'])} 条记录\n")

# 打印每条记录的详细信息
for i in range(len(results['ids'])):
    print(f"ID: {results['ids'][i]}")
    print(f"文档: {results['documents'][i][:200]}...")  # 打印前200个字符
    print(f"元数据: {results['metadatas'][i]}")
    print(f"嵌入向量: {results['embeddings'][i][:5]}... (共 {len(results['embeddings'][i])} 维)\n")