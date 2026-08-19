"""
Anthropic LLM Judge implementation using httpx.
"""

from __future__ import annotations

import os
from typing import Any, Optional
import httpx

from rag_eval.judge.base import BaseJudge


class AnthropicJudge(BaseJudge):
    """Anthropic Claude API judge for evaluating metrics."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Call Anthropic Messages API."""
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable or api_key parameter is required."
            )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            payload["system"] = system_prompt

        url = f"{self.base_url.rstrip('/')}/messages"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # Extract content text from message blocks
            blocks = data.get("content", [])
            text_blocks = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            return "".join(text_blocks)
