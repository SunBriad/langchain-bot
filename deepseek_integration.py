from langchain_core.language_models import BaseLLM
from deepseek import DeepSeekAPI
from pydantic import Field
import json

class DeepSeekLLM(BaseLLM):
    client: DeepSeekAPI = Field(default=None, exclude=True)
    model: str = "deepseek-chat"
    temperature: float = 0.7
    
    def __init__(self, api_key, **kwargs):
        super().__init__(**kwargs)
        self.client = DeepSeekAPI(api_key=api_key)
    
    def _generate(self, prompts, **kwargs):
        responses = []
        for prompt in prompts:
            try:
                response = self.client.chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                # 处理API响应
                if isinstance(response, str):
                    response = json.loads(response)
                responses.append(response["choices"][0]["message"]["content"])
            except Exception as e:
                print(f"Error calling DeepSeek API: {e}")
                responses.append("Error generating response")
        return responses
    
    def _llm_type(self):
        return "deepseek"
    
    def _call(self, prompt, **kwargs):
        return self._generate([prompt], **kwargs)[0]