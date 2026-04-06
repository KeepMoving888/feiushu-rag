"""
RAG 问答模块（体验优化版）
实现能力：
1) 多知识库自动路由检索（跨 collection）
2) 轻量意图分流（问候/闲聊优先）
3) 严格防幻觉 + 更可读回答模板（结论先行、依据后置）
4) 来源展示增强（filename/source_name/table_name/collection）
5) Redis 缓存高频问答
6) LLM 失败时降级为“检索证据直出”
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_openai import ChatOpenAI

from config import CONFIG
from vector_store import VectorStoreError, VectorStoreManager

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


class RAGChainError(Exception):
    """RAG 链路异常。"""


class RAGChain:
    """企业知识库 RAG 问答链。"""

    def __init__(self, vector_store: VectorStoreManager | None = None) -> None:
        self.vector_store = vector_store or VectorStoreManager()
        self.llm = self._init_llm()
        self.redis_client = self._init_redis_cache()

    @staticmethod
    def _init_llm() -> ChatOpenAI:
        try:
            llm_cfg = CONFIG.llm
            mode = llm_cfg.mode

            if mode == "api":
                if not llm_cfg.api.api_key:
                    raise RAGChainError("在线 API 模式缺少 API Key，请设置 LLM_API_KEY 或 OPENAI_API_KEY")

                return ChatOpenAI(
                    model=llm_cfg.api.model_name,
                    temperature=llm_cfg.temperature,
                    max_tokens=llm_cfg.max_tokens,
                    base_url=llm_cfg.api.base_url,
                    openai_api_key=llm_cfg.api.api_key,
                )

            if mode == "ollama":
                return ChatOpenAI(
                    model=llm_cfg.ollama.model_name,
                    temperature=llm_cfg.temperature,
                    max_tokens=llm_cfg.max_tokens,
                    base_url=llm_cfg.ollama.base_url,
                    openai_api_key="ollama-local",
                )

            raise RAGChainError(f"不支持的 LLM_MODE: {mode}，仅支持 api / ollama")
        except RAGChainError:
            raise
        except Exception as exc:
            raise RAGChainError(f"初始化 LLM 失败: {exc}") from exc

    @staticmethod
    def _init_redis_cache() -> Any:
        cache_cfg = CONFIG.rag_cache
        if not cache_cfg.enabled or not cache_cfg.redis_url or redis is None:
            return None

        try:
            client = redis.Redis.from_url(cache_cfg.redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:
            return None

    @staticmethod
    def _is_smalltalk(question: str) -> bool:
        q = question.strip().lower()
        smalltalk_keywords = ["你好", "在吗", "嗨", "hello", "hi", "你是谁", "你能做什么"]
        return any(k in q for k in smalltalk_keywords) and len(q) <= 20

    @staticmethod
    def _build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
        context_blocks: list[str] = []
        for idx, item in enumerate(contexts, start=1):
            meta = item.get("metadata", {})
            source_name = str(
                meta.get("source_name")
                or meta.get("filename")
                or meta.get("table_name")
                or "unknown"
            )
            collection_name = str(meta.get("collection_name") or "default")
            chunk_index = item.get("metadata", {}).get("chunk_index", idx - 1)
            content = item.get("text", "")
            context_blocks.append(
                f"[片段{idx} | 来源: {source_name} | 库: {collection_name} | chunk_index: {chunk_index}]\n{content}"
            )

        joined_context = "\n\n".join(context_blocks) if context_blocks else "无可用上下文"

        return (
            "你是企业知识库问答助手。\n"
            "必须遵守：\n"
            "1) 只能依据知识库片段作答，不得使用片段外信息。\n"
            "2) 若证据不足，明确回答：根据当前知识库内容无法确定。\n"
            "3) 不得编造事实。\n"
            "4) 回答必须为编号要点（1. 2. 3.），最多 3 条，每条 12~25 字。\n"
            "5) 只输出要点，不要额外解释或标题。\n\n"
            f"用户问题：{question}\n\n"
            "知识库检索片段：\n"
            f"{joined_context}\n\n"
            "请输出中文答案。"
        )

    @staticmethod
    def _collect_sources(contexts: list[dict[str, Any]]) -> list[str]:
        sources: list[str] = []
        for item in contexts:
            meta = item.get("metadata", {})
            source_name = str(meta.get("source_name") or meta.get("filename") or meta.get("table_name") or "")
            if source_name and source_name not in sources:
                sources.append(source_name)
        return sources

    @staticmethod
    def _append_source_footer(answer: str, contexts: list[dict[str, Any]]) -> str:
        # 按你的要求：答案区只保留核心编号要点，不拼接额外大段来源文案
        return answer

    @staticmethod
    def _format_to_numbered_points(answer: str) -> str:
        """将回答规整为 1/2/3 编号要点，避免长段大白话。"""

        if not answer or not answer.strip():
            return "1. 根据当前知识库内容无法确定。"

        text = answer.strip().replace("\r", "")
        raw_lines = [ln.strip(" -•\t") for ln in text.split("\n") if ln.strip()]

        # 去掉可能的标题行
        filtered: list[str] = []
        for ln in raw_lines:
            if ln in {"答案", "回答", "结论", "依据", "引用来源"}:
                continue
            filtered.append(ln)

        if not filtered:
            return "1. 根据当前知识库内容无法确定。"

        points: list[str] = []
        for ln in filtered:
            if ln[:2] in {"1.", "2.", "3.", "1、", "2、", "3、"}:
                content = ln[2:].strip()
            else:
                content = ln

            if content:
                points.append(content)
            if len(points) >= 3:
                break

        if not points:
            points = ["根据当前知识库内容无法确定。"]

        # 单行超长时按中文句号切分，确保简洁
        if len(points) == 1 and len(points[0]) > 40:
            sentence_parts = [p.strip() for p in points[0].replace("；", "。").split("。") if p.strip()]
            if sentence_parts:
                points = sentence_parts[:3]

        return "\n".join([f"{i}. {p}" for i, p in enumerate(points[:3], start=1)])

    def _build_cache_key(self, question: str, top_k: int) -> str:
        raw = {
            "mode": CONFIG.llm.mode,
            "model": CONFIG.llm.active_model_name,
            "q": question.strip(),
            "k": top_k,
            "routing": "multi_collection",
        }
        digest = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return f"{CONFIG.rag_cache.key_prefix}{digest}"

    def _get_cached_answer(self, cache_key: str) -> dict[str, Any] | None:
        if self.redis_client is None:
            return None

        try:
            value = self.redis_client.get(cache_key)
            if not value:
                return None
            data = json.loads(value)
            if isinstance(data, dict):
                data["cached"] = True
                return data
            return None
        except Exception:
            return None

    def _set_cached_answer(self, cache_key: str, result: dict[str, Any]) -> None:
        if self.redis_client is None:
            return

        try:
            payload = dict(result)
            payload.pop("cached", None)
            self.redis_client.setex(cache_key, CONFIG.rag_cache.ttl_seconds, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return

    def _route_and_retrieve(self, question: str, k: int = 3) -> list[dict[str, Any]]:
        """多知识库路由检索：先探测后深检索。"""

        try:
            collections = self.vector_store.list_collections()
        except Exception:
            collections = []

        if not collections:
            collections = [self.vector_store.collection_name]

        probe_results: list[tuple[str, float]] = []

        # 第一阶段：每个库 probe 1 条，估计相关性
        for col in collections:
            try:
                vs = VectorStoreManager(collection_name=col)
                one = vs.similarity_search(query=question, k=1)
                if not one:
                    continue
                score = one[0].get("score")
                score_val = float(score) if isinstance(score, (int, float)) else 999.0
                probe_results.append((col, score_val))
            except Exception:
                continue

        if not probe_results:
            return []

        probe_results.sort(key=lambda x: x[1])
        top_collections = [x[0] for x in probe_results[:3]]

        # 第二阶段：在候选库做深检索并合并重排
        merged: list[dict[str, Any]] = []
        for col in top_collections:
            try:
                vs = VectorStoreManager(collection_name=col)
                docs = vs.similarity_search(query=question, k=k)
                for d in docs:
                    md = d.get("metadata", {}) if isinstance(d.get("metadata"), dict) else {}
                    md = {**md, "collection_name": col}
                    if not md.get("source_name"):
                        md["source_name"] = md.get("filename") or md.get("table_name") or "unknown"
                    merged.append({"text": d.get("text", ""), "metadata": md, "score": d.get("score")})
            except Exception:
                continue

        if not merged:
            return []

        merged.sort(key=lambda x: float(x.get("score")) if isinstance(x.get("score"), (int, float)) else 999.0)
        return merged[: max(1, min(k * 2, 6))]

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        if not question or not question.strip():
            raise RAGChainError("问题不能为空")

        # C) 轻量意图分流
        if self._is_smalltalk(question):
            return {
                "answer": "已收到。请直接输入您的问题，我将基于知识库进行回答。",
                "sources": [],
                "contexts": [],
                "anti_hallucination": "smalltalk",
                "cached": False,
            }

        k = max(1, min(int(top_k), 3))
        cache_key = self._build_cache_key(question=question, top_k=k)

        fallback_result = {
            "answer": "抱歉，当前服务繁忙或知识库暂不可用。基于现有信息我不知道。",
            "sources": [],
            "contexts": [],
            "anti_hallucination": "fallback",
            "cached": False,
            "error": "",
        }

        cached = self._get_cached_answer(cache_key)
        if cached:
            return cached

        try:
            # A) 多知识库路由检索
            contexts = self._route_and_retrieve(question=question, k=k)
            if not contexts:
                result = {
                    "answer": "根据当前知识库内容暂无法确定答案。请补充制度名称、部门或时间范围后重试。",
                    "sources": [],
                    "contexts": [],
                    "anti_hallucination": "no_context_found",
                    "cached": False,
                }
                self._set_cached_answer(cache_key, result)
                return result

            prompt = self._build_prompt(question, contexts)
            try:
                llm_resp = self.llm.invoke(prompt)
                raw_answer = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)
                raw_answer = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)
            except Exception as exc:
                evidence_lines: list[str] = []
                for idx, c in enumerate(contexts[:3], start=1):
                    text = str(c.get("text", "")).strip().replace("\n", " ")
                    brief = text[:220] + ("..." if len(text) > 220 else "")
                    evidence_lines.append(f"{idx}. {brief}")

                sources = self._collect_sources(contexts)
                answer = (
                    "当前大模型服务暂时不可用（连接失败），先返回检索到的相关内容供参考：\n"
                    + "\n".join(evidence_lines)
                )
                final_answer = self._append_source_footer(answer, contexts)
                return {
                    "answer": final_answer,
                    "sources": sources,
                    "contexts": contexts,
                    "anti_hallucination": "llm_connection_error_with_context",
                    "cached": False,
                    "error": f"LLM invoke failed: {exc}",
                }

            if not raw_answer:
                raw_answer = "根据当前知识库内容无法确定。"

            # D) 体验优化：不足证据时更简洁引导
            if "无法确定" in raw_answer or "不知道" in raw_answer:
                raw_answer = "根据当前知识库内容无法确定。"

            sources = self._collect_sources(contexts)
            compact_answer = self._format_to_numbered_points(raw_answer)
            final_answer = self._append_source_footer(compact_answer, contexts)

            anti_status = "insufficient_context" if not sources else "grounded_with_context"

            result = {
                "answer": final_answer,
                "sources": sources,
                "contexts": contexts,
                "anti_hallucination": anti_status,
                "cached": False,
            }
            self._set_cached_answer(cache_key, result)
            return result

        except (VectorStoreError, RAGChainError) as exc:
            fallback_result["error"] = str(exc)
            return fallback_result
        except Exception as exc:
            fallback_result["error"] = str(exc)
            return fallback_result
