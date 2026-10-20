import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["ollama", "openai", "fake"]

_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "cheap_model": "qwen2.5:3b",
        "strong_model": "qwen2.5:7b",
        "prices": ("0", "0", "0", "0"),
        "timeout": "300",
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
    cheap_price_in: float
    cheap_price_out: float
    strong_price_in: float
    strong_price_out: float
    review_threshold: float
    max_tool_steps: int
    timeout: float
    fake_llm: bool

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
        provider = "fake"

    d = _DEFAULTS["openai" if provider == "openai" else "ollama"]
    p_ci, p_co, p_si, p_so = d["prices"]
    return Settings(
        provider=provider,
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
