"""Настройки из переменных окружения.

LLM_PROVIDER выбирает, где работают модели:
  ollama — локально через OpenAI-совместимый эндпоинт Ollama (по умолчанию, бесплатно);
  openai — любой внешний OpenAI-совместимый API (OpenAI, OpenRouter, vLLM, ...);
  fake   — детерминированная офлайн-модель для тестов и демо без моделей.
"""
import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["ollama", "openai", "fake"]

# Значения по умолчанию для каждого провайдера. Всё переопределяется переменными окружения.
_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "cheap_model": "qwen2.5:3b",
        "strong_model": "qwen2.5:7b",
        "prices": ("0", "0", "0", "0"),  # локально токены бесплатны
        "timeout": "300",  # на CPU ответ сильной модели может идти минуту и дольше
    },
    "openai": {
        "base_url": "",
        "cheap_model": "gpt-4o-mini",
        "strong_model": "gpt-4o",
        "prices": ("0.15", "0.60", "2.50", "10.00"),
        "timeout": "60",
    },
}


@dataclass(frozen=True)
class Settings:
    provider: Provider
    api_key: str | None
    base_url: str | None
    cheap_model: str
    strong_model: str
    # Цены за 1M токенов (вход / выход) — только для оценки стоимости в отчёте.
    cheap_price_in: float
    cheap_price_out: float
    strong_price_in: float
    strong_price_out: float
    review_threshold: float  # ниже этой уверенности — ручная проверка
    max_tool_steps: int  # защита от бесконечного цикла tool-calling
    timeout: float  # секунд на один запрос к модели
    fake_llm: bool  # офлайн-режим без моделей (детерминированная модель)

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if os.getenv("FAKE_LLM") == "1":
        provider = "fake"
    if provider not in ("ollama", "openai", "fake"):
        raise ValueError(f"LLM_PROVIDER='{provider}': допустимо ollama, openai или fake")

    api_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai" and not api_key:
        provider = "fake"  # без ключа внешний API недоступен — работаем офлайн

    d = _DEFAULTS["openai" if provider == "openai" else "ollama"]
    p_ci, p_co, p_si, p_so = d["prices"]
    return Settings(
        provider=provider,
        # Ollama ключ не проверяет, но клиент OpenAI требует непустое значение.
        api_key=api_key if provider == "openai" else "ollama",
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or d["base_url"] or None,
        cheap_model=os.getenv("CHEAP_MODEL", d["cheap_model"]),
        strong_model=os.getenv("STRONG_MODEL", d["strong_model"]),
        cheap_price_in=float(os.getenv("CHEAP_PRICE_IN", p_ci)),
        cheap_price_out=float(os.getenv("CHEAP_PRICE_OUT", p_co)),
        strong_price_in=float(os.getenv("STRONG_PRICE_IN", p_si)),
        strong_price_out=float(os.getenv("STRONG_PRICE_OUT", p_so)),
        review_threshold=float(os.getenv("REVIEW_THRESHOLD", "0.7")),
        max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "4")),
        timeout=float(os.getenv("LLM_TIMEOUT", d["timeout"])),
        fake_llm=provider == "fake",
    )
