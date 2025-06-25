from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import os
from typing import List, Union
import logging

# 配置镜像源（必须在所有import之前设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYPI_MIRROR'] = 'https://pypi.tuna.tsinghua.edu.cn/simple'

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 device: str = 'cpu', 
                 cache_dir: str = './model_cache'):
        """
        初始化嵌入生成器
        
        参数:
            model_name: 模型名称或路径
            device: 计算设备 ('cpu' 或 'cuda')
            cache_dir: 模型缓存目录
        """
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 加载模型和分词器（带重试机制）
            self.model = self._load_model(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            logger.info(f"成功加载模型: {model_name}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _load_model(self, model_name: str, max_retries: int = 3):
        """带重试机制的模型加载"""
        for i in range(max_retries):
            try:
                return SentenceTransformer(
                    model_name,
                    device=self.device,
                    cache_folder=self.cache_dir
                )
            except Exception as e:
                if i == max_retries - 1:
                    raise
                logger.warning(f"尝试 {i+1}/{max_retries} 失败，重试中...")
                import time; time.sleep(5 * (i + 1))
    
    def tokenize(self, text: str) -> List[str]:
        """返回分词结果"""
        return self.tokenizer.tokenize(text)
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                          batch_size: int = 32) -> np.ndarray:
        """
        生成文本的embedding向量
        
        参数:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
        返回:
            numpy数组形状为 (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True  # 建议归一化以便相似度计算
        )
    
    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算两个embedding的余弦相似度 (0~1)"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

if __name__ == "__main__":
    # 示例用法
    try:
        # 初始化（自动使用缓存）
        generator = EmbeddingGenerator(device='cpu')
        
        # 测试分词
        text = "这是一个测试句子"
        print("分词结果:", generator.tokenize(text))
        
        # 测试嵌入生成
        embedding = generator.generate_embeddings(text)[0]
        print("Embedding shape:", embedding.shape)
        print("Embedding norm:", np.linalg.norm(embedding))  # 应该≈1（因为归一化了）
        
        # 相似度计算
        text2 = "这是另一个测试句子"
        emb2 = generator.generate_embeddings(text2)[0]
        sim = generator.similarity(embedding, emb2)
        print(f"相似度: {sim:.4f}")
        
    except Exception as e:
        logger.error(f"运行出错: {e}")
        print(f"请尝试: 1) 检查网络 2) 手动下载模型到./model_cache 3) 更换更小的模型")