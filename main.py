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
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

client = chromadb.PersistentClient(path="db")  # 数据存储在db目录

# 初始化 DeepSeek LLM
llm = ChatOpenAI(
    api_key="sk-01f00364308747f1a783c455f0fc231b",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    streaming=True
)

# 1. 加载文档并创建向量数据库
loader = TextLoader("your_document.txt",encoding="utf-8")  # 替换为你的文档路径
documents = loader.load()





embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu' },
    encode_kwargs={'normalize_embeddings': False}
)

# 加载模型和分词器
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = model.tokenizer

tokens = tokenizer.tokenize(text)

# 打印分词结果
print("分词结果:")
print(model.tokenizer.convert_tokens_to_string(tokens))

# 分割文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
# # 3. 使用 sentence-transformers 嵌入模型
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",  # 常用的小型高效模型
#     model_kwargs={'device': 'cpu'},  # 使用 'cuda' 如果有GPU
#     encode_kwargs={'normalize_embeddings': False}
# )



# 连接到本地 Chroma 服务
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    client=client,
    collection_name="my_knowledge_base"  # 指定集合名称
)


# 2. 设置检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 创建对话记忆
message_history = ChatMessageHistory()
memory = {
    
    "chat_history": message_history} 
    #ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. 定义提示模板
template = """基于以下上下文和对话历史，回答用户的问题:
上下文: {context}
对话历史: {chat_history}
问题: {input}
回答:"""
prompt = ChatPromptTemplate.from_template(template)

# 5. 构建处理链

# 构建处理链
def format_docs(docs):
    """将文档列表转换为纯文本字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_chat_history(history):
    """格式化对话历史"""
    return "\n".join(f"{msg.type}: {msg.content}" for msg in history)

chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"])),
        chat_history=lambda x: get_chat_history(x["chat_history"])
    )
    | prompt
    | llm
    | StrOutputParser()
)
# 3. 对话循环中使用
while True:
    user_input = input("\n你: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    print("AI: ", end="", flush=True)
    full_response = ""

    print(type(message_history.messages[0]))

    try:
        for chunk in chain.stream({
            "input": user_input,
            "chat_history": message_history.messages  # 传入消息列表
        }):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        # 更新记忆（新版方式）
        message_history.add_user_message(user_input)
        message_history.add_ai_message(full_response)
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        continue