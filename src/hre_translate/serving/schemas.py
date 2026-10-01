from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ModelChoice = Literal["auto", "hybrid", "dictionary", "translation_memory", "nllb"]


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    source: Literal["hre"] = "hre"
    target: Literal["vi"] = "vi"
    model: ModelChoice = "auto"

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class TranslateResponse(BaseModel):
    translation: str
    model: str
    latency_ms: float = Field(ge=0)
    retrieved_terms: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_examples: list[dict[str, Any]] = Field(default_factory=list)
    similarity: float | None = None


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=16)
    source: Literal["hre"] = "hre"
    target: Literal["vi"] = "vi"
    model: ModelChoice = "auto"

    @field_validator("texts")
    @classmethod
    def valid_texts(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 2000 for value in cleaned):
            raise ValueError("each text must contain 1–2000 nonblank characters")
        return cleaned


class BatchResponse(BaseModel):
    translations: list[TranslateResponse]
    latency_ms_total: float = Field(ge=0)


class FeedbackRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2000)
    prediction: str = Field(min_length=1, max_length=4000)
    correction: str = Field(min_length=1, max_length=4000)
    model: str = Field(min_length=1, max_length=100)
    rating: int | None = Field(default=None, ge=1, le=5)

    @field_validator("source", "prediction", "correction", "model")
    @classmethod
    def nonblank_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be blank")
        return value


class FeedbackResponse(BaseModel):
    id: int
    timestamp: str
