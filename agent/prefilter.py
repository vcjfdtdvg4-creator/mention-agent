import hashlib
import re

from .schemas import Mention

SPAM_MARKERS = ("подпишись", "промокод http", "переходи по ссылке", "заработок без вложений")
MIN_LENGTH = 15


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def prefilter(mentions: list[Mention], aliases: list[str]) -> tuple[list[Mention], dict[str, str]]:
    kept: list[Mention] = []
    dropped: dict[str, str] = {}
    seen: set[str] = set()
    aliases = [a.lower() for a in aliases]

    for m in mentions:
        norm = _normalize(m.text)
        digest = hashlib.sha1(norm.encode()).hexdigest()
        if digest in seen:
            dropped[m.id] = "duplicate"
        elif len(norm) < MIN_LENGTH:
            dropped[m.id] = "too_short"
        elif any(s in m.text.lower() for s in SPAM_MARKERS):
            dropped[m.id] = "spam"
        elif not any(a in m.text.lower() for a in aliases):
            dropped[m.id] = "not_about_brand"
        else:
            kept.append(m)
        seen.add(digest)
    return kept, dropped
