from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ConditionInput(BaseModel):
    asset_id: str = Field(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")


class GenerateRequest(BaseModel):
    """t2i: テキストのみ。edit: 参照画像 1〜16 枚(プロンプト中で <Picture 1> … と参照)。"""

    mode: Literal["t2i", "edit"] = "t2i"
    prompt: str = Field(..., min_length=1, max_length=8000)
    width: int = Field(2048, ge=256, le=8192)
    height: int = Field(2048, ge=256, le=8192)

    @field_validator("width", "height", mode="before")
    @classmethod
    def _none_to_default(cls, v):
        return 2048 if v is None else v
    seed: int | None = None
    # 2 / 3 = pDMD LoRA 蒸留(高速)、6 = チェックポイント既定。None = サーバ既定(H3IMG_STEPS)
    steps: Literal[2, 3, 6] | None = None
    # native = 最低 4MP で生成して縮小(モデル推奨)、requested = 要求サイズで生成(高速)
    generation_area: Literal["native", "requested"] | None = None
    conditions: list[ConditionInput] = Field(default_factory=list, max_length=16)

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        return (v // 32) * 32


class AssetResponse(BaseModel):
    id: str
    kind: Literal["image"]
    filename: str
    size: int


class JobResponse(BaseModel):
    id: str
    status: str                 # queued | running | completed | failed | interrupted
    progress: float = 0
    error: str | None = None
    image_url: str | None = None
    video_url: str | None = None  # ltx25 互換のため常に None
    request: dict[str, Any]
    created_at: str
    updated_at: str
    generation_seconds: float | None = None
    peak_vram_gb: float | None = None
    steps_used: int | None = None
