"""
OpenAI LLM Judge implementation using httpx.
"""

from __future__ import annotations

import os
from typing import Any, Optional
import httpx

from rag_eval.judge.base import BaseJudge


class OpenAIJudge(BaseJudge):
    """OpenAI API judge for evaluating metrics."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Call OpenAI chat completions API."""
        if not self.api_key and "localhost" not in (self.base_url or ""):
            raise ValueError(
                "OPENAI_API_KEY environment variable or api_key parameter is required."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
