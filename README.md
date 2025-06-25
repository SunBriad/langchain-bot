# LangChain 创意广告生成器

这是一个使用LangChain生成产品创意广告文案的简单项目。

## 功能
- 根据输入的产品名称生成创意广告文案
- 使用OpenAI的LLM模型
- 可自定义提示模板

## 使用方法
1. 安装依赖: `pip install -r requirements.txt`
2. 设置OPENAI_API_KEY环境变量
3. 运行: `python main.py`

## 示例
```python
print(chain.run("智能手表"))
