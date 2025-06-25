from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from sentence_transformers import SentenceTransformer

import chromadb
import os

# 配置设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像
client = chromadb.PersistentClient(path="db")  # 数据存储在db目录

# 1. 初始化组件
def initialize_components():
    """初始化所有需要的组件"""
    # 初始化DeepSeek LLM
    llm = ChatOpenAI(
        api_key="sk-01f00364308747f1a783c455f0fc231b",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        streaming=True
    )

    # 加载文档
    loader = TextLoader("your_document.txt", encoding="utf-8")  # 替换为你的文档路径
    documents = loader.load()

    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # 加载模型和分词器（用于调试）
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = model.tokenizer

    # 打印示例分词（调试用）
    if len(documents) > 0:
        sample_text = documents[0].page_content[:100]  # 取前100字符作为示例
        tokens = tokenizer.tokenize(sample_text)
        print("\n示例分词结果:")
        print(tokenizer.convert_tokens_to_string(tokens))

    # 分割文档
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 创建向量存储
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        client=client,
        collection_name="my_knowledge_base"
    )

    return llm, vectorstore, ChatMessageHistory()

# 2. 构建处理链
def build_chain(llm, vectorstore, message_history):
    """构建RAG处理链"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 定义提示模板
    template = """基于以下上下文和对话历史，回答用户的问题:
    上下文: {context}
    对话历史: {chat_history}
    问题: {input}
    请给出专业、详细的回答:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 辅助函数
    def format_docs(docs):
        """将文档列表转换为纯文本字符串"""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_chat_history(history):
        """格式化对话历史"""
        return "\n".join(f"{msg.type}: {msg.content}" for msg in history.messages)

    # 构建处理链
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"])),
            chat_history=lambda x: get_chat_history(x["chat_history"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# 3. 主对话循环
def main_conversation(chain, message_history):
    """处理用户交互"""
    print("\n问答系统已就绪，输入 'exit' 或 'quit' 退出")
    
    while True:
        try:
            user_input = input("\n你: ")
            if user_input.lower() in ["exit", "quit"]:
                print("对话结束，再见！")
                break
            
            print("AI: ", end="", flush=True)
            full_response = ""

            # 流式处理响应
            for chunk in chain.stream({
                "input": user_input,
                "chat_history": message_history
            }):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            # 更新对话历史
            message_history.add_user_message(user_input)
            message_history.add_ai_message(full_response)
            
        except KeyboardInterrupt:
            print("\n检测到中断，退出系统...")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            continue

if __name__ == "__main__":
    # 初始化组件
    llm, vectorstore, message_history = initialize_components()
    
    # 构建处理链
    chain = build_chain(llm, vectorstore, message_history)
    
    # 启动对话
    main_conversation(chain, message_history)