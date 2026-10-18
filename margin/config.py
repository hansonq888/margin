"""Runtime configuration, loaded from environment / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    model_vision: str
    model_fast: str
    model_smart: str
    reasoning_off: str  # "kwargs" | "system" | "none"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def load() -> Config:
    return Config(
        api_key=os.getenv("NEBIUS_API_KEY", ""),
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
        model_vision=os.getenv("MARGIN_MODEL_VISION", "nvidia/nemotron-3-nano-omni-30b-a3b"),
        model_fast=os.getenv("MARGIN_MODEL_FAST", "nvidia/nemotron-3-nano-30b-a3b"),
        model_smart=os.getenv("MARGIN_MODEL_SMART", "nvidia/nemotron-3-super-120b-a12b"),
        reasoning_off=os.getenv("MARGIN_REASONING_OFF", "kwargs"),
    )
