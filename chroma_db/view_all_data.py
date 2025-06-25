import chromadb 

# 连接到 Chroma 数据库 
client = chromadb.PersistentClient(path="D:\\project\\langchain-master\\anaconda_projects\\new_langchain_project\\db")  # 替换为你的数据库路径 


# 列出所有集合
collections = client.list_collections()

# 打印集合名称和文档数量
for collection in collections:
    print(f"集合名称: {collection.name}")
    print(f"文档数量: {collection.count()}")
    print("-" * 50)

collection = client.get_collection("my_knowledge_base")  # 替换为你的集合名称 

# 获取所有存储的数据 
results = collection.get() 
print("IDs:", results["ids"]) 
print("Documents:", results["documents"]) 
print("Metadatas:", results["metadatas"]) 
print("Embeddings (部分):", results["embeddings"][:1] if results["embeddings"] else "未存储原始向量")

print("\n数据总数:", len(results["ids"]))