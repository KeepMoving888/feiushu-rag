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
import re
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
            "2) 不得编造事实。\n"
            "3) 回答必须为编号要点（1. 2. 3.），最多 3 条。\n"
            "4) 若问题较泛化，请先概括可确认的制度要点，再补充不确定项。\n"
            "5) 若仅部分可确认，不要整段回答‘无法确定’，应给出可确认部分。\n"
            "6) 仅在完全无有效证据时，才回答：根据当前知识库内容无法确定。\n"
            "7) 只输出要点，不要额外解释或标题。\n\n"
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

    @staticmethod
    def _expand_query_candidates(question: str) -> list[str]:
        """泛化问题扩展：在保证语义相关前提下补充精确子查询。"""

        q = (question or "").strip()
        if not q:
            return []

        candidates: list[str] = [q]
        normalized = q.lower()

        # 请假/休假类
        if any(k in q for k in ["请假", "放假", "休假", "假期", "调休"]) or "leave" in normalized:
            candidates.extend(
                [
                    f"{q} 年假",
                    f"{q} 病假",
                    f"{q} 事假",
                    f"{q} 调休",
                    f"{q} 审批流程",
                    f"{q} 申请材料",
                ]
            )

        # 报销/财务类
        if any(k in q for k in ["报销", "费用", "发票", "差旅", "财务"]) or any(k in normalized for k in ["expense", "invoice"]):
            candidates.extend([f"{q} 报销流程", f"{q} 发票要求", f"{q} 审批", f"{q} 额度"])

        # IT/安全类
        if any(k in q for k in ["密码", "账号", "权限", "安全", "it"]) or any(k in normalized for k in ["password", "security", "account"]):
            candidates.extend([f"{q} 权限申请", f"{q} 安全规范", f"{q} 审批流程"])

        # 去重保序 + 长度保护
        seen: set[str] = set()
        deduped: list[str] = []
        for c in candidates:
            c = re.sub(r"\s+", " ", c).strip()
            if not c or c in seen or len(c) > 120:
                continue
            seen.add(c)
            deduped.append(c)

        return deduped[:7]

    def _retrieve_from_collection(self, question: str, collection_name: str, k: int = 3) -> list[dict[str, Any]]:
        """从指定知识库检索（支持泛化扩展 + 精排）。"""

        if not collection_name or not collection_name.strip():
            return []

        queries = self._expand_query_candidates(question)
        if not queries:
            return []

        try:
            vs = VectorStoreManager(collection_name=collection_name.strip())
        except Exception:
            return []

        # doc 唯一键：优先 id，其次 文本+来源
        merged_by_key: dict[str, dict[str, Any]] = {}

        for qi, q in enumerate(queries):
            try:
                docs = vs.similarity_search(query=q, k=max(k, 3))
            except Exception:
                continue

            for d in docs:
                md = d.get("metadata", {}) if isinstance(d.get("metadata"), dict) else {}
                md = {**md, "collection_name": collection_name.strip()}
                if not md.get("source_name"):
                    md["source_name"] = md.get("filename") or md.get("table_name") or "unknown"

                text = str(d.get("text", "") or "")
                score_raw = d.get("score")
                score = float(score_raw) if isinstance(score_raw, (int, float)) else 999.0

                # 精排加权：原问题命中优先，其次扩展查询
                # 值越小越好，给原问题命中一个负偏置。
                rerank_score = score + (0.0 if qi == 0 else 0.08 * qi)

                doc_id = str(md.get("id") or "")
                key = doc_id or f"{md.get('source_name','')}-{md.get('chunk_index','')}-{hash(text[:120])}"

                current = {
                    "text": text,
                    "metadata": md,
                    "score": score_raw,
                    "_rerank_score": rerank_score,
                }
                old = merged_by_key.get(key)
                if old is None or float(current["_rerank_score"]) < float(old.get("_rerank_score", 999.0)):
                    merged_by_key[key] = current

        merged = list(merged_by_key.values())
        merged.sort(key=lambda x: float(x.get("_rerank_score", 999.0)))

        final = []
        for item in merged[: max(1, min(k * 2, 8))]:
            item.pop("_rerank_score", None)
            final.append(item)
        return final

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

    def ask(self, question: str, top_k: int = 3, collection_name: str | None = None) -> dict[str, Any]:
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
            # A) 指定知识库检索（优先）或多知识库路由检索
            if collection_name and collection_name.strip():
                contexts = self._retrieve_from_collection(question=question, collection_name=collection_name, k=k)
            else:
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

            # D) 优化：仅在确实无上下文时才强制“无法确定”
            # 有检索上下文时，允许回答可确认的部分，避免泛化问题被一刀切。
            normalized = raw_answer.replace(" ", "")
            if (
                ("无法确定" in normalized or "不知道" in normalized)
                and contexts
                and any(str(c.get("text", "")).strip() for c in contexts)
            ):
                evidence_hints: list[str] = []
                for c in contexts[:3]:
                    meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
                    src = str(meta.get("source_name") or meta.get("filename") or meta.get("table_name") or "该制度")
                    snippet = str(c.get("text", "")).strip().replace("\n", " ")
                    if snippet:
                        evidence_hints.append(f"可参考{src}：{snippet[:70]}{'...' if len(snippet) > 70 else ''}")

                if evidence_hints:
                    raw_answer = "\n".join([f"{i}. {txt}" for i, txt in enumerate(evidence_hints[:3], start=1)])
                else:
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
