"""Строгие схемы данных. Всё, что возвращает LLM, проходит через эти модели."""
from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]
Severity = Literal["low", "medium", "high"]
Owner = Literal["support", "pr", "marketing", "product", "legal"]
Stage = Literal["prefilter_dropped", "triaged_low", "planned"]


class Mention(BaseModel):
    id: str
    source: str
    author: str
    text: str
    url: str | None = None
    published_at: str | None = None


class Triage(BaseModel):
    """Ответ дешёвой модели: быстрая сортировка упоминания."""

    relevant: bool = Field(description="Касается ли упоминание бренда")
    sentiment: Sentiment
    severity: Severity = Field(description="Насколько срочно нужна реакция")
    topic: str = Field(description="Одно из: delivery, payment, quality, data_leak, legal, praise, other")
    confidence: float = Field(ge=0, le=1)


class ActionStep(BaseModel):
    action: str = Field(min_length=5)
    owner: Owner
    deadline_hours: int = Field(gt=0, le=168)


class ActionPlan(BaseModel):
    """Ответ сильной модели: план реакции с опорой на плейбук."""

    summary: str = Field(min_length=10, description="Суть проблемы в 1–2 предложениях")
    playbook_section: str = Field(description="Ключ раздела плейбука, на который опирается план")
    steps: list[ActionStep] = Field(min_length=1, max_length=6)
    reply_draft: str | None = Field(default=None, description="Черновик публичного ответа")
    evidence: list[str] = Field(default_factory=list, description="Факты из инструментов")
    confidence: float = Field(ge=0, le=1)


class MentionResult(BaseModel):
    mention_id: str
    stage: Stage
    drop_reason: str | None = None
    triage: Triage | None = None
    plan: ActionPlan | None = None
    needs_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
