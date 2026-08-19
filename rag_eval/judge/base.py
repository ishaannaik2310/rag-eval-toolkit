"""
Base Judge interface for LLM-as-a-judge evaluations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class BaseJudge(ABC):
    """Abstract base class for all LLM judge implementations."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Asynchronously generate a response from the LLM judge."""
        pass

    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Synchronously generate a response from the LLM judge."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        self.generate(prompt, system_prompt, json_mode, temperature)
                    ).result()
            return loop.run_until_complete(
                self.generate(prompt, system_prompt, json_mode, temperature)
            )
        except RuntimeError:
            return asyncio.run(
                self.generate(prompt, system_prompt, json_mode, temperature)
            )
