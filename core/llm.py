"""LLM 调用封装。"""

import os
from openai import OpenAI


class LLMClient:
    """LLM 客户端。"""

    def __init__(self, model: str = "gpt-4o", api_key: str = None, base_url: str = None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
