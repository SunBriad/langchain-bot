# 聊天交互脚本
import os
from deepseek_integration import DeepSeekLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 初始化模型
llm = DeepSeekLLM(api_key=os.getenv("DEEPSEEK_API_KEY"))
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# 聊天循环
print("欢迎使用LangChain-Bot! 输入'退出'结束对话")
while True:
    user_input = input("你: ")
    if user_input == '退出':
        break
    response = conversation.run(input=user_input)
    print(f"Bot: {response}")