"""Small LLM client wrapper used by the GAPA planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .api_env import load_api_env


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    base_url: str | None
    api_key: str | None

    @property
    def is_configured(self) -> bool:
        return bool(self.model and self.api_key and not self.api_key.startswith("replace_with_"))


def get_llm_config(provider: str | None = None) -> LLMConfig:
    env = load_api_env()
    provider = (provider or env.get("GAPA_LLM_PROVIDER") or "deepseek").lower()

    if provider == "deepseek":
        return LLMConfig(
            provider=provider,
            model=env.get("GAPA_LLM_MODEL") or "deepseek-chat",
            base_url=env.get("GAPA_LLM_BASE_URL") or "https://api.deepseek.com",
            api_key=env.get("GAPA_LLM_API_KEY"),
        )
    if provider == "openai":
        return LLMConfig(
            provider=provider,
            model=env.get("GAPA_LLM_MODEL"),
            base_url=env.get("GAPA_LLM_BASE_URL") or "https://api.openai.com/v1",
            api_key=env.get("GAPA_LLM_API_KEY"),
        )

    return LLMConfig(
        provider=provider,
        model=env.get("GAPA_LLM_MODEL"),
        base_url=env.get("GAPA_LLM_BASE_URL"),
        api_key=env.get("GAPA_LLM_API_KEY"),
    )


class LLMClient:
    """OpenAI-compatible chat client with a no-key friendly configuration check."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or get_llm_config()

    @property
    def is_configured(self) -> bool:
        return self.config.is_configured

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.0) -> str:
        if not self.is_configured:
            raise RuntimeError("GAPA LLM is not configured. Set GAPA_LLM_API_KEY and GAPA_LLM_MODEL in gapa/gapa_api.env.")

        from openai import OpenAI

        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""
