"""The single supported local model backend; private reasoning is never returned."""
from __future__ import annotations

import asyncio
import re
from typing import Any

from .contracts import LabConfig


def public_text(value: str) -> str:
    """Remove reasoning blocks, including an unterminated block after truncation."""
    return re.sub(r"<think>.*?(?:</think>|$)", "", value or "", flags=re.S).strip()


class OllamaBackend:
    def __init__(self, config: LabConfig):
        self.config = config

    def chat(self, messages: list[dict], tools: list[dict], *,
             timeout_s: float, max_tokens: int) -> dict[str, Any]:
        """Use an overall async deadline, rather than only a socket read timeout."""
        import ollama

        async def request() -> dict:
            client = ollama.AsyncClient(host=self.config.host, timeout=timeout_s)
            try:
                response = await asyncio.wait_for(
                    client.chat(
                        model=self.config.model, messages=messages, tools=tools,
                        stream=False, think=False,
                        options={"num_ctx": self.config.num_ctx,
                                 "num_predict": max_tokens,
                                 "temperature": self.config.temperature,
                                 "seed": self.config.seed},
                    ), timeout=timeout_s,
                )
                result = response.model_dump(exclude_none=True)
                message = result.get("message", {})
                return {
                    "content": public_text(message.get("content", "")),
                    "tool_calls": message.get("tool_calls", []),
                    "prompt_tokens": result.get("prompt_eval_count"),
                    # Ollama eval_count covers generation, including reasoning.
                    "completion_tokens": result.get("eval_count"),
                    "model": result.get("model", self.config.model),
                    "done_reason": result.get("done_reason"),
                }
            finally:
                # The pinned Ollama client owns an httpx client without public close().
                await client._client.aclose()

        return asyncio.run(request())
