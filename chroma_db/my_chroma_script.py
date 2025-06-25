import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

# 初始化Chroma客户端
client = chromadb.PersistentClient(path="db")

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# 查询函数
def query_chroma(collection_name, query_text, n_results=3):
    try:
        # 获取集合
        collection = client.get_collection(collection_name)
        
        # 生成查询嵌入
        query_embedding = embeddings.embed_query(query_text)
        
        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 打印结果
        print(f"查询: '{query_text}'")
        print(f"找到 {len(results['documents'][0])} 条相关文档:")
        for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"{i+1}. 相似度: {1-dist:.2f}")
            print(f"   内容: {doc}")
            print("-" * 50)
        
    except Exception as e:
        print(f"查询出错: {e}")

# 测试查询
if __name__ == "__main__":
    while True:
        query = input("\n请输入查询内容(输入'exit'退出): ")
        if query.lower() == 'exit':
            break
        query_chroma("my_knowledge_base", query)