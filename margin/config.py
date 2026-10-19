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
    model_fast: str  # conversation: speed matters
    model_smart: str  # confirming mistakes, help hints
    model_screen: str  # background first-pass check of every problem
    reasoning_off: str  # "kwargs" | "system" | "none"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def load() -> Config:
    return Config(
        api_key=os.getenv("NEBIUS_API_KEY", ""),
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
        model_vision=os.getenv("MARGIN_MODEL_VISION", "nvidia/Nemotron-3.5-Super-VL"),
        model_fast=os.getenv("MARGIN_MODEL_FAST", "nvidia/Nemotron-3_5-Lightning"),
        model_smart=os.getenv("MARGIN_MODEL_SMART", "nvidia/nemotron-3-super-120b-a12b"),
        # Super, not Lightning: on university-level work Lightning's screen missed real mistakes (6/8 vs 7/8 caught,
        # 0 false alarms either way), and screening runs in the background so the extra speed isn't felt.
        model_screen=os.getenv("MARGIN_MODEL_SCREEN", "nvidia/nemotron-3-super-120b-a12b"),
        reasoning_off=os.getenv("MARGIN_REASONING_OFF", "kwargs"),
    )
