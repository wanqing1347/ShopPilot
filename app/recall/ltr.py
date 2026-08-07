from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from app.agent.settings import (
    retrieval_embedding_model,
    retrieval_reranker_model_file,
)
from app.models import Candidate
from app.recall.tokenizer import normalize_text, tokenize
from app.utils.runtime import PROJECT_ROOT

RerankerMode = Literal["auto", "none", "rules", "learned"]
FEATURE_VERSION = 1
FEATURE_NAMES = (
    "normalized_rrf",
    "bm25_score",
    "vector_score",
    "bm25_reciprocal_rank",
    "vector_reciprocal_rank",
    "material_match",
    "style_match",
    "feature_match_ratio",
    "preference_overlap",
    "budget_fit",
    "rating",
    "sales",
    "delivery",
    "quality",
    "rule_score",
    "train_ctr",
    "train_favorite_rate",
    "train_purchase_rate",
    "train_dislike_rate",
)


@dataclass(frozen=True)
class RerankSignals:
    normalized_rrf: float
    bm25_score: float
    vector_score: float
    bm25_rank: int | None
    vector_rank: int | None
    rule_score: float


@dataclass(frozen=True)
class LearnedReranker:
    model_version: int
    feature_version: int
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    scales: tuple[float, ...]
    embedding_provider: str
    embedding_model: str | None
    group_priors: dict[str, tuple[float, float, float, float]]
    training: dict[str, Any]

    def compatible(self, provider_name: str, model_name: str | None) -> bool:
        if self.feature_version != FEATURE_VERSION:
            return False
        if self.feature_names != FEATURE_NAMES:
            return False
        if self.embedding_provider != provider_name:
            return False
        if provider_name == "sentence_transformers":
            return normalize_text(self.embedding_model or "") == normalize_text(model_name or "")
        return True

    def feature_vector(
        self,
        *,
        candidate: Candidate,
        query: str,
        user_preferences: list[str],
        budget_cny: float | None,
        signals: RerankSignals,
        candidate_text: str,
    ) -> list[float]:
        query_blob = normalize_text(" ".join([query, *user_preferences]))
        normalized_candidate = normalize_text(candidate_text)
        attributes = candidate.attributes

        material = normalize_text(str(attributes.get("material", "")))
        style = normalize_text(str(attributes.get("style", "")))
        features = [
            normalize_text(str(value))
            for value in attributes.get("features", [])
            if normalize_text(str(value))
        ]
        material_match = 1.0 if material and material in query_blob else 0.0
        style_match = 1.0 if style and style in query_blob else 0.0
        feature_match_ratio = (
            sum(feature in query_blob for feature in features) / len(features)
            if features
            else 0.0
        )

        preference_tokens = set(tokenize(" ".join(user_preferences)))
        candidate_tokens = set(tokenize(normalized_candidate))
        preference_overlap = (
            len(preference_tokens & candidate_tokens) / len(preference_tokens)
            if preference_tokens
            else 0.0
        )

        price = candidate.landed_price_cny or candidate.price_cny
        budget_fit = 0.5
        if budget_cny and price is not None:
            budget_fit = max(0.0, min(1.0, 1.0 - price / budget_cny))

        rating = max(0.0, min(float(candidate.rating or 0.0) / 5.0, 1.0))
        sales = min(math.log1p(float(candidate.sales or 0.0)) / math.log1p(5000), 1.0)
        delivery = max(0.0, 1.0 - max(0, (candidate.delivery_days or 30) - 5) / 30)
        quality = 1.0 if candidate.quality_grade == "A" else 0.4
        bm25_score = max(0.0, signals.bm25_score)
        bm25_score = bm25_score / (bm25_score + 10.0)
        vector_score = max(0.0, min(1.0, (signals.vector_score + 1.0) / 2.0))
        prior = self.group_priors.get(candidate.same_group_id, (0.0, 0.0, 0.0, 0.0))

        return [
            max(0.0, min(1.0, signals.normalized_rrf)),
            bm25_score,
            vector_score,
            1.0 / signals.bm25_rank if signals.bm25_rank else 0.0,
            1.0 / signals.vector_rank if signals.vector_rank else 0.0,
            material_match,
            style_match,
            feature_match_ratio,
            preference_overlap,
            budget_fit,
            rating,
            sales,
            delivery,
            quality,
            max(0.0, min(1.0, signals.rule_score)),
            *prior,
        ]

    def score_features(self, features: list[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError(
                f"LTR 特征维度错误: expected={len(self.weights)}, actual={len(features)}"
            )
        return sum(
            weight * (value / max(scale, 1e-8))
            for value, weight, scale in zip(features, self.weights, self.scales, strict=True)
        )

    def score(
        self,
        *,
        candidate: Candidate,
        query: str,
        user_preferences: list[str],
        budget_cny: float | None,
        signals: RerankSignals,
        candidate_text: str,
    ) -> float:
        return self.score_features(
            self.feature_vector(
                candidate=candidate,
                query=query,
                user_preferences=user_preferences,
                budget_cny=budget_cny,
                signals=signals,
                candidate_text=candidate_text,
            )
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "scales": list(self.scales),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "group_priors": {
                group_id: list(values)
                for group_id, values in sorted(self.group_priors.items())
            },
            "training": self.training,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_document(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def from_document(cls, raw: dict[str, Any]) -> "LearnedReranker":
        feature_names = tuple(str(value) for value in raw["feature_names"])
        weights = tuple(float(value) for value in raw["weights"])
        scales = tuple(float(value) for value in raw["scales"])
        if len(feature_names) != len(weights) or len(weights) != len(scales):
            raise ValueError("LTR 模型 feature_names/weights/scales 长度不一致")
        priors = {
            str(group_id): tuple(float(value) for value in values)
            for group_id, values in (raw.get("group_priors") or {}).items()
        }
        if any(len(values) != 4 for values in priors.values()):
            raise ValueError("LTR group_priors 必须包含 ctr/favorite/purchase/dislike 四个值")
        return cls(
            model_version=int(raw.get("model_version") or 1),
            feature_version=int(raw["feature_version"]),
            feature_names=feature_names,
            weights=weights,
            scales=scales,
            embedding_provider=str(raw["embedding_provider"]),
            embedding_model=(
                str(raw["embedding_model"])
                if raw.get("embedding_model") is not None
                else None
            ),
            group_priors=priors,
            training=dict(raw.get("training") or {}),
        )


def reranker_model_path() -> Path:
    configured = Path(retrieval_reranker_model_file()).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (PROJECT_ROOT / configured).resolve()


@lru_cache(maxsize=8)
def _load_model(path_text: str, modified_ns: int) -> LearnedReranker:
    del modified_ns
    path = Path(path_text)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"LTR 模型必须是 JSON 对象: {path}")
    return LearnedReranker.from_document(raw)


def load_compatible_reranker(
    *,
    provider_name: str,
    model_name: str | None = None,
) -> tuple[LearnedReranker | None, str | None]:
    path = reranker_model_path()
    if not path.exists():
        return None, f"模型文件不存在: {path}"
    try:
        model = _load_model(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, f"模型加载失败: {exc}"
    resolved_model = model_name or retrieval_embedding_model()
    if not model.compatible(provider_name, resolved_model):
        return (
            None,
            "模型与当前 embedding 不兼容: "
            f"model={model.embedding_provider}/{model.embedding_model}, "
            f"runtime={provider_name}/{resolved_model}",
        )
    return model, None


def clear_ltr_cache() -> None:
    _load_model.cache_clear()
