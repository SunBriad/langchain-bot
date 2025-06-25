from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def generate_embedding(self, text):
        """生成单个文本的嵌入向量"""
        return self.model.encode(text).tolist()
        
    def generate_embeddings(self, texts):
        """生成多个文本的嵌入向量"""
        return self.model.encode(texts).tolist()
        
    def calculate_similarity(self, embedding1, embedding2):
        """计算两个嵌入向量的余弦相似度"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# 使用示例
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    text1 = "这是第一个测试句子"
    text2 = "这是第二个测试句子"
    
    embedding1 = generator.generate_embedding(text1)
    embedding2 = generator.generate_embedding(text2)
    
    print(f"文本1嵌入向量长度: {len(embedding1)}")
    print(f"文本2嵌入向量长度: {len(embedding2)}")
    print(f"相似度: {generator.calculate_similarity(embedding1, embedding2):.4f}")