"""Инструменты агента.

У каждого инструмента: строгая схема аргументов (Pydantic), описание для модели
в формате OpenAI tools и функция-исполнитель. Аргументы от модели валидируются
до выполнения — модель не может вызвать инструмент с мусором.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from .schemas import Mention

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_json(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@dataclass
class ToolContext:
    """Данные, к которым у инструментов есть доступ (только чтение)."""

    mentions: list[Mention]
    company: dict
    playbook: dict


# ---------- аргументы ----------

class SearchArgs(BaseModel):
    query: str = Field(min_length=2, description="Ключевые слова: район, продукт, тип проблемы")
    limit: int = Field(default=5, ge=1, le=10)


class PlaybookArgs(BaseModel):
    section: str = Field(description="Ключ раздела: delivery, payment, quality, data_leak, legal, praise, other")


class NoArgs(BaseModel):
    pass


# ---------- реализации ----------

def search_related_mentions(ctx: ToolContext, args: SearchArgs) -> dict:
    """Ищет похожие упоминания — помогает понять, единичный это случай или волна."""
    words = [w.lower() for w in args.query.split() if len(w) > 2]
    hits = [m for m in ctx.mentions if any(w in m.text.lower() for w in words)]
    return {
        "query": args.query,
        "total_found": len(hits),
        "samples": [{"id": m.id, "source": m.source, "text": m.text} for m in hits[: args.limit]],
    }


def get_playbook(ctx: ToolContext, args: PlaybookArgs) -> dict:
    section = ctx.playbook.get(args.section)
    if section is None:
        return {"error": f"Раздел '{args.section}' не найден", "available": list(ctx.playbook)}
    return {"section": args.section, **section}


def get_company_facts(ctx: ToolContext, args: NoArgs) -> dict:
    return ctx.company


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[ToolContext, BaseModel], dict]

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("search_related_mentions",
             "Найти другие упоминания по ключевым словам, чтобы понять масштаб проблемы.",
             SearchArgs, search_related_mentions),
        Tool("get_playbook",
             "Получить правила реагирования из корпоративного плейбука по разделу.",
             PlaybookArgs, get_playbook),
        Tool("get_company_facts",
             "Получить факты о компании: продукты, SLA поддержки, политику возвратов.",
             NoArgs, get_company_facts),
    ]
}


def execute_tool(ctx: ToolContext, name: str, raw_args: dict) -> dict:
    """Безопасное выполнение: неизвестный инструмент или кривые аргументы
    возвращаются модели как ошибка, а не роняют граф."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Неизвестный инструмент '{name}'", "available": list(TOOLS)}
    try:
        args = tool.args_model.model_validate(raw_args or {})
    except ValidationError as e:
        return {"error": "Некорректные аргументы", "details": e.errors(include_url=False)}
    return tool.fn(ctx, args)


def openai_tools() -> list[dict]:
    return [t.openai_schema() for t in TOOLS.values()]
