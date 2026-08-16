from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.agent.settings import dataset_dir, dataset_schema_version
from app.models import Candidate, Platform
from app.utils.runtime import PROJECT_ROOT


def resolve_dataset_root(configured: str | Path) -> Path:
    configured_path = Path(configured).expanduser()
    return (
        configured_path.resolve()
        if configured_path.is_absolute()
        else (PROJECT_ROOT / configured_path).resolve()
    )


def dataset_root() -> Path:
    configured = Path(dataset_dir()).expanduser()
    return configured.resolve() if configured.is_absolute() else (PROJECT_ROOT / configured).resolve()


def products_file() -> Path:
    return dataset_root() / "products.jsonl"


def dataset_summary_file() -> Path:
    return dataset_root() / "dataset_summary.json"


@lru_cache(maxsize=8)
def _load_catalog(
    path_text: str,
    modified_ns: int,
    expected_schema: int,
) -> tuple[Candidate, ...]:
    del modified_ns  # Included in the cache key so regenerated files reload automatically.
    path = Path(path_text)
    rows: list[Candidate] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                candidate = Candidate.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"商品数据不符合 Candidate Schema: {path}:{line_no}: {exc}"
                ) from exc
            if candidate.schema_version != expected_schema:
                raise ValueError(
                    f"商品数据 Schema 不兼容: {candidate.item_id} 使用 "
                    f"v{candidate.schema_version}，项目要求 v{expected_schema}"
                )
            if candidate.item_id in seen_ids:
                raise ValueError(f"商品数据 item_id 重复: {candidate.item_id}")
            seen_ids.add(candidate.item_id)
            rows.append(candidate)
    if not rows:
        raise ValueError(f"商品数据为空: {path}")
    return tuple(rows)


def load_catalog_from_dir(directory: str | Path) -> tuple[Candidate, ...]:
    root = resolve_dataset_root(directory)
    path = root / "products.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "未找到 ShopPilot 商品数据集。请生成数据后设置 "
            f"SHOPPILOT_DATASET_DIR；当前路径: {path}"
        )

    summary_path = root / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 dataset_summary.json: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"dataset_summary.json 解析失败: {summary_path}: {exc}") from exc
    expected_schema = dataset_schema_version()
    actual_schema = int(summary.get("schema_version") or 0)
    if actual_schema != expected_schema:
        raise ValueError(
            f"数据集 Schema 不兼容: dataset=v{actual_schema}, project=v{expected_schema}"
        )

    return _load_catalog(str(path), path.stat().st_mtime_ns, expected_schema)


def load_catalog() -> tuple[Candidate, ...]:
    return load_catalog_from_dir(dataset_root())


def catalog_for_platform(
    platform: Platform | str,
    *,
    include_unavailable: bool = False,
) -> list[Candidate]:
    return [
        item
        for item in load_catalog()
        if item.platform == platform and (include_unavailable or item.is_available)
    ]


def clear_catalog_cache() -> None:
    _load_catalog.cache_clear()
