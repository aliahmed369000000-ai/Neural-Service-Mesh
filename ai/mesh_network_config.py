"""إعدادات واكتشاف عقد NSM بدون اتصالات شبكية."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SeedEndpoint:
    url: str

    @property
    def health_url(self) -> str:
        return self.url.rstrip("/") + "/health"


def parse_seed_urls(value: str | None) -> tuple[SeedEndpoint, ...]:
    """حلّل قائمة بذور مفصولة بفواصل/أسطر، وارفض العناوين غير الآمنة."""
    candidates = (value or "").replace("\n", ",").split(",")
    result: list[SeedEndpoint] = []
    seen: set[str] = set()
    for raw in candidates:
        url = raw.strip().rstrip("/")
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/') }"
        if normalized not in seen:
            seen.add(normalized)
            result.append(SeedEndpoint(normalized))
    return tuple(result)


def configured_seed_urls() -> tuple[SeedEndpoint, ...]:
    """اقرأ SEED_NODE_URLS مع دعم التوافق مع SEED_NODE_URL المفرد."""
    return parse_seed_urls(os.getenv("SEED_NODE_URLS") or os.getenv("SEED_NODE_URL"))


def seed_retry_order(seeds: tuple[SeedEndpoint, ...], failed_url: str = "") -> tuple[SeedEndpoint, ...]:
    """رتّب البذور بحيث تُجرّب البدائل قبل البذرة الفاشلة."""
    return tuple(seed for seed in seeds if seed.url != failed_url) + tuple(
        seed for seed in seeds if seed.url == failed_url
    )


def reconnect_delay(attempt: int, base: float = 1.0, maximum: float = 60.0) -> float:
    """تأخير exponential backoff محدود لإعادة الاتصال."""
    return min(maximum, base * (2 ** max(0, attempt)))


def public_url(name: str, fallback: str = "") -> str:
    """لا تعلن bind address داخلياً كرابط عام."""
    value = os.getenv(name, fallback).strip().rstrip("/")
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        return ""
    if parsed.hostname in {"0.0.0.0", "127.0.0.1", "localhost"}:
        return ""
    return value


__all__ = [
    "SeedEndpoint",
    "configured_seed_urls",
    "parse_seed_urls",
    "public_url",
    "reconnect_delay",
    "seed_retry_order",
]
