# DeepSeek API集成
import os
import requests
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

class DeepSeekLLM(LLM):
    api_key: str = os.getenv("DEEPSEEK_API_KEY")
    model_name: str = "deepseek-chat"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()['choices'][0]['message']['content']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}