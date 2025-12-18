#!/usr/bin/env python3
"""
LLM-based reranking for tag predictions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from openai import OpenAI


@dataclass
class LLMRerankerConfig:
    """Configuration for LLM reranker."""
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_candidates: int = 20
    max_return: int = 5
    max_input_chars: int = 2000
    temperature: float = 0.0


class LLMReranker:
    """Uses an LLM to re-rank candidate tags."""

    def __init__(self, config: LLMRerankerConfig):
        self.config = config
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )

    @classmethod
    def from_env(cls, **overrides) -> "LLMReranker":
        """
        Build reranker using OPENAI_* env vars.
        Expected env vars:
            OPENAI_TOKEN / OPENAI_API_KEY
            OPENAI_MODEL
            OPENAI_BASE_URL (optional)
        """
        api_key = overrides.get("api_key") or os.getenv("OPENAI_TOKEN") or os.getenv("OPENAI_API_KEY")
        model = overrides.get("model") or os.getenv("OPENAI_MODEL")
        base_url = overrides.get("base_url") or os.getenv("OPENAI_BASE_URL")
        if not api_key or not model:
            raise ValueError("OPENAI_TOKEN (or OPENAI_API_KEY) and OPENAI_MODEL must be set for LLM reranker.")
        cfg = LLMRerankerConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_candidates=int(overrides.get("max_candidates") or 20),
            max_return=int(overrides.get("max_return") or 5),
            max_input_chars=int(overrides.get("max_input_chars") or 2000),
            temperature=float(overrides.get("temperature") or 0.0),
        )
        return cls(cfg)

    def rerank(
        self,
        text: str,
        good_type: Optional[str],
        candidates: Sequence[Tuple[str, float]],
        max_return: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Run reranker on candidate tags.
        Args:
            text: Product text (name + desc + reviews).
            good_type: Category (helps LLM reason).
            candidates: List of (tag, score) from base model.
            max_return: Optional cap for number of tags to keep.
        Returns:
            List of (tag, llm_score) sorted by llm_score desc.
        """
        if not candidates:
            return []
        max_return = int(max_return or self.config.max_return)

        trimmed_text = (text or "").strip()
        if len(trimmed_text) > self.config.max_input_chars:
            trimmed_text = trimmed_text[: self.config.max_input_chars]

        limited = list(candidates[: self.config.max_candidates])
        candidate_lines = "\n".join(
            f"{i+1}. {tag} (model_prob={score:.3f})" for i, (tag, score) in enumerate(limited)
        )

        system_prompt = (
            "You help decide which product tags apply to an item. "
            "Only choose tags from the provided list. "
            "Respond with strict JSON: {\"tags\": [{\"tag\": \"...\", \"score\": 0-1}, ...]} "
            "sorted by score desc and limited to the requested number. "
            "Return an empty list if no tags fit."
        )

        user_prompt = f"""Product category: {good_type or "unknown"}
Product text:
{trimmed_text}

Candidate tags (choose only from this list):
{candidate_lines}

Return at most {max_return} tags that are strongly supported by the text.
"""

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
        )
        content = response.choices[0].message.content if response.choices else ""
        return self._parse_response(content, limited, max_return)

    def _parse_response(
        self,
        content: Optional[str],
        candidates: Sequence[Tuple[str, float]],
        max_return: int,
    ) -> List[Tuple[str, float]]:
        if not content:
            return []
        raw = content.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                return []
            try:
                data = json.loads(raw[start : end + 1])
            except Exception:
                return []

        tags_data = []
        if isinstance(data, dict):
            tags_data = data.get("tags", [])
        elif isinstance(data, list):
            tags_data = data

        allowed = {tag: score for tag, score in candidates}
        result = []
        for item in tags_data:
            if not isinstance(item, dict):
                continue
            tag = item.get("tag")
            if not tag or tag not in allowed:
                continue
            score = item.get("score")
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = allowed[tag]
            result.append((tag, float(score)))
            if len(result) >= max_return:
                break

        return result


__all__ = ["LLMReranker", "LLMRerankerConfig"]
