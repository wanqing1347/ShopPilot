from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.knowledge.models import KnowledgeDocument
from app.recall.catalog import dataset_root


def knowledge_file() -> Path:
    return dataset_root() / "category_knowledge.jsonl"


@lru_cache(maxsize=4)
def _load_documents(path_text: str, modified_ns: int) -> tuple[KnowledgeDocument, ...]:
    del modified_ns
    path = Path(path_text)
    documents: list[KnowledgeDocument] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"知识文档 JSON 解析失败: {path}:{line_no}: {exc}") from exc
            required = {"doc_id", "category_key", "category_path", "title", "content", "source"}
            missing = sorted(required - raw.keys())
            if missing:
                raise ValueError(f"知识文档缺少字段 {missing}: {path}:{line_no}")
            doc_id = str(raw["doc_id"]).strip()
            if not doc_id:
                raise ValueError(f"知识文档 doc_id 为空: {path}:{line_no}")
            if doc_id in seen_ids:
                raise ValueError(f"知识文档 doc_id 重复: {doc_id}")
            seen_ids.add(doc_id)
            category_path = tuple(str(value).strip() for value in raw["category_path"] if str(value).strip())
            if not category_path:
                raise ValueError(f"知识文档 category_path 为空: {doc_id}")
            title = str(raw["title"]).strip()
            content = str(raw["content"]).strip()
            if not title or not content:
                raise ValueError(f"知识文档标题或内容为空: {doc_id}")
            documents.append(
                KnowledgeDocument(
                    doc_id=doc_id,
                    category_key=str(raw["category_key"]).strip(),
                    category_path=category_path,
                    title=title,
                    content=content,
                    source=str(raw["source"]).strip(),
                    updated_at=(str(raw["updated_at"]).strip() if raw.get("updated_at") else None),
                )
            )
    if not documents:
        raise ValueError(f"知识文档为空: {path}")
    return tuple(documents)


def load_knowledge_documents() -> tuple[KnowledgeDocument, ...]:
    path = knowledge_file()
    if not path.exists():
        raise FileNotFoundError(f"缺少品类知识数据: {path}")
    return _load_documents(str(path), path.stat().st_mtime_ns)


def clear_knowledge_cache() -> None:
    _load_documents.cache_clear()
