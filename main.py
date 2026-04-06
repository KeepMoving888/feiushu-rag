
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import CONFIG
from document_processor import DocumentProcessor, DocumentProcessorError
from feishu_client import FeishuClient, FeishuClientError
from rag_chain import RAGChain, RAGChainError
from vector_store import VectorStoreError
"""
FastAPI 后端入口
接口：
1) /health 健康检查
2) /upload 文件上传并入库
3) /ask RAG 问答
4) /feishu/webhook 飞书回调接收（文本问答 + 文件入库闭环）

优化策略（保持整体框架不变）：
- webhook 幂等处理（优先 Redis，回退内存缓存）
- 回消息分段发送（避免超长消息发送失败）
- 同一事件并发锁（并发到达时仅处理一次）
- 消息发送重试（指数退避）
- trace_id 贯穿日志，便于线上定位
- APScheduler 定时任务：知识库增量同步 + 权限映射同步
- 同步状态持久化（JSON）：服务重启后不丢状态
"""

try:
    import redis
except Exception:  # pragma: no cover
    redis = None

app = FastAPI(title="Feishu RAG Knowledge Base", version="0.1.0")

# 全局实例（原型阶段）
feishu_client = FeishuClient()
doc_processor = DocumentProcessor()
rag_chain = RAGChain()

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feishu-rag")

# APScheduler
scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

# 持久化状态文件
_STATE_DIR = Path("./data/state")
_SYNC_DOC_STATE_FILE = _STATE_DIR / "sync_doc_state.json"
_PERMISSION_MAPPING_FILE = _STATE_DIR / "permission_mapping.json"
_BITABLE_BINDING_FILE = _STATE_DIR / "bitable_binding.json"
_BITABLE_BINDINGS_ADMIN_FILE = _STATE_DIR / "bitable_bindings_admin.json"
_BITABLE_RECORD_STATE_FILE = _STATE_DIR / "bitable_record_state.json"
_BITABLE_SYNC_RESULT_FILE = _STATE_DIR / "bitable_sync_result.json"

# 知识库同步状态（内存 + JSON持久化）
# key: doc_token/file_token, value: modified_time
_SYNC_DOC_STATE: dict[str, str] = {}
_SYNC_DOC_LOCK = threading.Lock()

# 权限映射状态（内存 + JSON持久化）
# key: user_id/open_id, value: role/permission data
_PERMISSION_MAPPING: dict[str, dict[str, Any]] = {}
_PERMISSION_LOCK = threading.Lock()

# 最近一次同步结果（便于管理接口查看）
_LAST_SYNC_RESULT: dict[str, Any] = {}
_LAST_PERMISSION_SYNC_RESULT: dict[str, Any] = {}
_LAST_BITABLE_SYNC_RESULT: dict[str, Any] = {}

# 多维表格记录级同步状态（用于增量对比）
# key: "app_token::table_id" -> {"record_id": "last_modified_time"}
_BITABLE_RECORD_STATE: dict[str, dict[str, str]] = {}
_BITABLE_RECORD_LOCK = threading.Lock()

# 幂等缓存（内存版回退）
_EVENT_CACHE: dict[str, float] = {}
_EVENT_CACHE_TTL_SECONDS = 10 * 60  # 10 分钟
_EVENT_LOCK = threading.Lock()

# Redis 幂等（可选）
_REDIS_URL = os.getenv("REDIS_URL", "").strip()
_REDIS_PREFIX = "feishu_rag:event:"
_REDIS_TTL_SECONDS = _EVENT_CACHE_TTL_SECONDS
_REDIS_CLIENT = None
if redis and _REDIS_URL:
    try:
        _REDIS_CLIENT = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        _REDIS_CLIENT.ping()
        logger.info("Redis idempotency enabled")
    except Exception:
        _REDIS_CLIENT = None
        logger.warning("Redis init failed, fallback to memory cache")

# 飞书单条文本消息长度保护阈值（保守值）
_FEISHU_TEXT_CHUNK_SIZE = 1500

# 回消息重试参数
_REPLY_MAX_RETRIES = 3
_REPLY_BACKOFF_SECONDS = 0.5

# 飞书知识库同步配置
FEISHU_SYNC_SPACE_ID = os.getenv("FEISHU_SYNC_SPACE_ID", "").strip()

# 飞书多维表格增量同步配置（支持 .env + 持久化绑定）
_DEFAULT_BITABLE_APP_TOKEN = os.getenv("BITABLE_APP_TOKEN", "").strip()
_DEFAULT_BITABLE_TABLE_ID = os.getenv("BITABLE_TABLE_ID", "").strip()
try:
    _DEFAULT_BITABLE_MAX_RECORDS = int(os.getenv("BITABLE_MAX_RECORDS", "1000") or "1000")
except (TypeError, ValueError):
    _DEFAULT_BITABLE_MAX_RECORDS = 1000

BITABLE_BINDING: dict[str, Any] = {
    "app_token": _DEFAULT_BITABLE_APP_TOKEN,
    "table_id": _DEFAULT_BITABLE_TABLE_ID,
    "max_records": _DEFAULT_BITABLE_MAX_RECORDS,
}

# 管理接口鉴权（可选，建议生产开启）
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "").strip()


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


def _ensure_state_dir() -> None:
    """确保状态目录存在。"""

    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def _load_json_file(path: Path, default: Any) -> Any:
    """读取 JSON 文件，失败时返回默认值。"""

    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json_file(path: Path, data: Any) -> None:
    """保存 JSON 文件。"""

    _ensure_state_dir()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_persistent_states() -> None:
    """启动时加载持久化状态到内存。"""

    sync_state = _load_json_file(_SYNC_DOC_STATE_FILE, {})
    perm_state = _load_json_file(_PERMISSION_MAPPING_FILE, {})

    if isinstance(sync_state, dict):
        with _SYNC_DOC_LOCK:
            _SYNC_DOC_STATE.clear()
            _SYNC_DOC_STATE.update({str(k): str(v) for k, v in sync_state.items()})

    if isinstance(perm_state, dict):
        with _PERMISSION_LOCK:
            _PERMISSION_MAPPING.clear()
            _PERMISSION_MAPPING.update(perm_state)


def _persist_sync_doc_state() -> None:
    """落盘文档同步状态。"""

    with _SYNC_DOC_LOCK:
        snapshot = dict(_SYNC_DOC_STATE)
    _save_json_file(_SYNC_DOC_STATE_FILE, snapshot)


def _persist_permission_mapping() -> None:
    """落盘权限映射。"""

    with _PERMISSION_LOCK:
        snapshot = dict(_PERMISSION_MAPPING)
    _save_json_file(_PERMISSION_MAPPING_FILE, snapshot)


def _load_bitable_binding() -> None:
    """加载多维表格绑定配置（若存在）。"""

    data = _load_json_file(_BITABLE_BINDING_FILE, {})
    if not isinstance(data, dict):
        return

    app_token = str(data.get("app_token") or "").strip()
    table_id = str(data.get("table_id") or "").strip()

    try:
        max_records = int(data.get("max_records") or BITABLE_BINDING.get("max_records", 1000))
    except (TypeError, ValueError):
        max_records = int(BITABLE_BINDING.get("max_records", 1000))

    if app_token:
        BITABLE_BINDING["app_token"] = app_token
    if table_id:
        BITABLE_BINDING["table_id"] = table_id
    BITABLE_BINDING["max_records"] = max_records


def _persist_bitable_binding() -> None:
    """落盘多维表格绑定配置。"""

    _save_json_file(_BITABLE_BINDING_FILE, dict(BITABLE_BINDING))


def _load_bitable_record_state() -> None:
    """加载多维表格记录级同步状态。"""

    data = _load_json_file(_BITABLE_RECORD_STATE_FILE, {})
    if not isinstance(data, dict):
        return

    cleaned: dict[str, dict[str, str]] = {}
    for table_key, rec_map in data.items():
        if not isinstance(rec_map, dict):
            continue
        cleaned[str(table_key)] = {str(k): str(v) for k, v in rec_map.items()}

    with _BITABLE_RECORD_LOCK:
        _BITABLE_RECORD_STATE.clear()
        _BITABLE_RECORD_STATE.update(cleaned)


def _persist_bitable_record_state() -> None:
    """落盘多维表格记录级同步状态。"""

    with _BITABLE_RECORD_LOCK:
        snapshot = dict(_BITABLE_RECORD_STATE)
    _save_json_file(_BITABLE_RECORD_STATE_FILE, snapshot)


def _load_bitable_bindings() -> list[dict[str, Any]]:
    """加载 admin 维护的多维表格绑定列表（统一配置源）。"""

    data = _load_json_file(_BITABLE_BINDINGS_ADMIN_FILE, [])
    if not isinstance(data, list):
        return []

    result: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        app_token = str(item.get("app_token") or "").strip()
        table_id = str(item.get("table_id") or "").strip()
        if not app_token or not table_id:
            continue

        try:
            max_records = int(item.get("max_records") or 1000)
        except (TypeError, ValueError):
            max_records = 1000

        normalized = {
            "app_token": app_token,
            "table_id": table_id,
            "table_name": str(item.get("table_name") or table_id),
            "max_records": max(1, min(max_records, 50000)),
            "auto_sync": bool(item.get("auto_sync", False)),
            "last_sync_time": str(item.get("last_sync_time") or ""),
            "last_status": str(item.get("last_status") or ""),
            "last_records": int(item.get("last_records") or 0),
            "last_message": str(item.get("last_message") or ""),
        }
        result.append(normalized)

    return result


def _save_bitable_bindings(bindings: list[dict[str, Any]]) -> None:
    """保存 admin 维护的多维表格绑定列表。"""

    _save_json_file(_BITABLE_BINDINGS_ADMIN_FILE, bindings)


def _persist_bitable_sync_result() -> None:
    """落盘最近一次多维表格同步结果。"""

    _save_json_file(_BITABLE_SYNC_RESULT_FILE, dict(_LAST_BITABLE_SYNC_RESULT))


def _load_bitable_sync_result() -> None:
    """启动时加载最近一次多维表格同步结果。"""

    data = _load_json_file(_BITABLE_SYNC_RESULT_FILE, {})
    if isinstance(data, dict):
        _LAST_BITABLE_SYNC_RESULT.clear()
        _LAST_BITABLE_SYNC_RESULT.update(data)


def _ingest_local_file(file_path: Path) -> dict[str, Any]:
    """将本地文件解析后写入向量库。"""

    parsed = doc_processor.parse_file(str(file_path))
    chunks = doc_processor.split_text(parsed.text, parsed.metadata)
    ids = rag_chain.vector_store.add_documents(chunks)
    return {"chunk_count": len(chunks), "vector_ids": ids}


def _clean_event_cache() -> None:
    """清理过期 event_id，避免内存持续增长。"""

    now = time.time()
    expired_keys = [eid for eid, ts in _EVENT_CACHE.items() if now - ts > _EVENT_CACHE_TTL_SECONDS]
    for eid in expired_keys:
        _EVENT_CACHE.pop(eid, None)


def _is_duplicate_event(event_id: str) -> bool:
    """判断事件是否重复（优先 Redis，回退内存缓存）。"""

    if not event_id:
        return False

    if _REDIS_CLIENT is not None:
        try:
            ok = _REDIS_CLIENT.set(f"{_REDIS_PREFIX}{event_id}", "1", nx=True, ex=_REDIS_TTL_SECONDS)
            return not bool(ok)
        except Exception:
            pass

    with _EVENT_LOCK:
        _clean_event_cache()
        if event_id in _EVENT_CACHE:
            return True
        _EVENT_CACHE[event_id] = time.time()
        return False


def _split_text_message(text: str, chunk_size: int = _FEISHU_TEXT_CHUNK_SIZE) -> list[str]:
    """按长度切分文本消息。"""

    if not text:
        return [""]

    normalized = text.strip()
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        chunks.append(normalized[start : start + chunk_size])
        start += chunk_size
    return chunks


def _send_with_retry(open_id: str, text: str, trace_id: str, retries: int = _REPLY_MAX_RETRIES) -> bool:
    """发送单条文本，失败后按指数退避重试。"""

    if not open_id:
        return False

    for attempt in range(retries):
        try:
            feishu_client.send_text_message(open_id, text)
            return True
        except FeishuClientError as exc:
            logger.warning("[%s] send retry %s/%s failed: %s", trace_id, attempt + 1, retries, exc)
            if attempt == retries - 1:
                return False
            time.sleep(_REPLY_BACKOFF_SECONDS * (2**attempt))
    return False


def _safe_reply_to_user(open_id: str, text: str, trace_id: str) -> None:
    """安全回复用户：自动分段发送 + 单段重试 + 不影响主流程。"""

    if not open_id:
        return

    segments = _split_text_message(text)
    for idx, segment in enumerate(segments, start=1):
        content = f"[{idx}/{len(segments)}] {segment}" if len(segments) > 1 else segment
        _send_with_retry(open_id, content, trace_id=trace_id)


def _extract_doc_item_fields(item: dict[str, Any]) -> tuple[str, str, str]:
    """兼容不同返回结构提取文档 token / 名称 / 更新时间。"""

    token = str(item.get("token") or item.get("file_token") or item.get("obj_token") or "")
    name = str(item.get("title") or item.get("name") or token)
    modified = str(item.get("modified_time") or item.get("update_time") or item.get("edited_time") or "")
    return token, name, modified


def _extract_record_modified_time(record: dict[str, Any]) -> str:
    """提取多维表格记录更新时间（兼容不同字段）。"""

    return str(
        record.get("last_modified_time")
        or record.get("updated_time")
        or record.get("created_time")
        or ""
    )


def _sync_single_bitable_table(app_token: str, table_id: str, table_name: str = "", max_records: int = 1000) -> dict[str, Any]:
    """同步单张多维表格（增量 + 删除对齐）。"""

    table_key = f"{app_token}::{table_id}"
    if not table_name:
        table_name = table_id

    # 1) 拉取结构与记录
    structure = feishu_client.get_bitable_structure(app_token=app_token, table_id=table_id)
    headers = structure.get("field_name_to_id", {})
    records = feishu_client.get_bitable_records(app_token=app_token, table_id=table_id, page_size=200)
    records = records[: max(1, int(max_records))]

    current_map: dict[str, str] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("record_id") or rec.get("id") or "").strip()
        if not rid:
            continue
        current_map[rid] = _extract_record_modified_time(rec)

    # 2) 读取本地旧状态
    with _BITABLE_RECORD_LOCK:
        old_map = dict(_BITABLE_RECORD_STATE.get(table_key, {}))

    old_ids = set(old_map.keys())
    current_ids = set(current_map.keys())

    deleted_ids = old_ids - current_ids
    candidate_ids = [rid for rid in current_ids if old_map.get(rid, "") != current_map.get(rid, "")]

    # 3) 删除已被飞书删除的记录
    deleted_count = 0
    for rid in deleted_ids:
        try:
            feishu_client.delete_bitable_record_from_vector(app_token=app_token, table_id=table_id, record_id=rid)
            deleted_count += 1
        except FeishuClientError:
            # 删除失败不阻断整体流程
            continue

    # 4) 仅同步新增/修改记录
    record_lookup: dict[str, dict[str, Any]] = {}
    for rec in records:
        if isinstance(rec, dict):
            rid = str(rec.get("record_id") or rec.get("id") or "").strip()
            if rid:
                record_lookup[rid] = rec

    changed_records: list[dict[str, Any]] = [record_lookup[rid] for rid in candidate_ids if rid in record_lookup]

    texts, metadatas = doc_processor.process_bitable_data(
        headers=headers,
        records=changed_records,
        app_token=app_token,
        table_id=table_id,
        table_name=table_name,
    )

    synced_count = 0
    for idx, text in enumerate(texts):
        md = metadatas[idx] if idx < len(metadatas) else {}
        rid = str(md.get("record_id") or "").strip()
        if not rid:
            continue
        try:
            rag_chain.vector_store.update_bitable_record(
                app_token=app_token,
                table_id=table_id,
                record_id=rid,
                new_data={"text": text, "metadata": md},
            )
            synced_count += 1
        except VectorStoreError:
            continue

    # 5) 更新本地记录状态（相当于更新最后同步时间）
    with _BITABLE_RECORD_LOCK:
        _BITABLE_RECORD_STATE[table_key] = current_map
    _persist_bitable_record_state()

    return {
        "table_key": table_key,
        "app_token": app_token,
        "table_id": table_id,
        "table_name": table_name,
        "total_records": len(current_map),
        "synced_records": synced_count,
        "deleted_records": deleted_count,
        "timestamp": int(time.time()),
    }


def run_bitable_incremental_sync(trigger: str = "scheduler", trace_id: str = "") -> dict[str, Any]:
    """定时同步所有开启自动同步的多维表格（每10分钟）。"""

    if not trace_id:
        trace_id = str(uuid.uuid4())

    bindings = _load_bitable_bindings()
    enabled_bindings = [b for b in bindings if bool(b.get("auto_sync", False))]

    # 兼容单一绑定配置（老逻辑）
    fallback_app = str(BITABLE_BINDING.get("app_token") or "").strip()
    fallback_table = str(BITABLE_BINDING.get("table_id") or "").strip()
    if fallback_app and fallback_table and not enabled_bindings:
        enabled_bindings = [
            {
                "app_token": fallback_app,
                "table_id": fallback_table,
                "table_name": str(BITABLE_BINDING.get("table_name") or fallback_table),
                "max_records": int(BITABLE_BINDING.get("max_records") or 1000),
                "auto_sync": True,
            }
        ]

    table_results: list[dict[str, Any]] = []
    failed = 0

    for b in enabled_bindings:
        app_token = str(b.get("app_token") or "").strip()
        table_id = str(b.get("table_id") or "").strip()
        table_name = str(b.get("table_name") or table_id)
        try:
            max_records = int(b.get("max_records") or 1000)
        except (TypeError, ValueError):
            max_records = 1000

        if not app_token or not table_id:
            continue

        try:
            one = _sync_single_bitable_table(
                app_token=app_token,
                table_id=table_id,
                table_name=table_name,
                max_records=max_records,
            )
            table_results.append(one)
        except Exception as exc:
            failed += 1
            table_results.append(
                {
                    "app_token": app_token,
                    "table_id": table_id,
                    "table_name": table_name,
                    "error": str(exc),
                }
            )

    # 回写每个绑定项的最近同步状态
    bindings_mut = _load_bitable_bindings()
    for b in bindings_mut:
        app = str(b.get("app_token") or "").strip()
        tid = str(b.get("table_id") or "").strip()
        hit = next(
            (
                x
                for x in table_results
                if str(x.get("app_token") or "").strip() == app and str(x.get("table_id") or "").strip() == tid
            ),
            None,
        )
        if not hit:
            continue

        b["last_sync_time"] = str(int(time.time()))
        if hit.get("error"):
            b["last_status"] = "failed"
            b["last_message"] = str(hit.get("error"))
            b["last_records"] = 0
        else:
            b["last_status"] = "success"
            b["last_message"] = "同步成功"
            b["last_records"] = int(hit.get("synced_records") or 0)

    _save_bitable_bindings(bindings_mut)

    result = {
        "success": failed == 0,
        "trigger": trigger,
        "tables_total": len(enabled_bindings),
        "tables_failed": failed,
        "tables": table_results,
        "trace_id": trace_id,
        "timestamp": int(time.time()),
    }
    _LAST_BITABLE_SYNC_RESULT.clear()
    _LAST_BITABLE_SYNC_RESULT.update(result)
    _persist_bitable_sync_result()
    return result


def run_incremental_knowledge_sync(trigger: str = "scheduler", trace_id: str = "") -> dict[str, Any]:
    """飞书知识库增量同步。

    同步范围：
    1) Wiki/Drive 文档（按 modified_time 增量）
    2) 绑定的多维表格（按记录更新时间聚合出的版本号增量）
    """

    if not trace_id:
        trace_id = str(uuid.uuid4())

    logger.info("[%s] knowledge sync start, trigger=%s", trace_id, trigger)

    changed_count = 0
    failed_count = 0
    scanned_count = 0
    has_state_changed = False

    try:
        # -----------------------------
        # A. 文档增量同步
        # -----------------------------
        if FEISHU_SYNC_SPACE_ID:
            resp = feishu_client.list_knowledge_docs(space_id=FEISHU_SYNC_SPACE_ID, page_size=200)
            items = resp.get("data", {}).get("items", [])
        else:
            resp = feishu_client.list_drive_files(page_size=200)
            items = resp.get("data", {}).get("files", [])

        if not isinstance(items, list):
            items = []

        scanned_count += len(items)

        for item in items:
            token, name, modified = _extract_doc_item_fields(item)
            if not token:
                continue

            with _SYNC_DOC_LOCK:
                last_modified = _SYNC_DOC_STATE.get(token, "")

            if last_modified == modified and modified:
                continue

            try:
                file_bytes = feishu_client.download_file(token)
                upload_dir = Path(CONFIG.upload_dir)
                upload_dir.mkdir(parents=True, exist_ok=True)

                suffix = Path(name).suffix or ".txt"
                stem = Path(name).stem or "feishu_doc"
                local_path = upload_dir / f"sync_{stem}{suffix}"
                local_path.write_bytes(file_bytes)

                _ingest_local_file(local_path)

                with _SYNC_DOC_LOCK:
                    _SYNC_DOC_STATE[token] = modified or str(int(time.time()))
                has_state_changed = True
                changed_count += 1
            except (FeishuClientError, DocumentProcessorError, VectorStoreError, OSError) as exc:
                failed_count += 1
                logger.warning("[%s] sync changed doc failed: token=%s err=%s", trace_id, token, exc)

        if has_state_changed:
            _persist_sync_doc_state()

        logger.info(
            "[%s] knowledge sync done, scanned=%s changed=%s failed=%s",
            trace_id,
            scanned_count,
            changed_count,
            failed_count,
        )
        result = {
            "success": True,
            "trigger": trigger,
            "scanned": scanned_count,
            "changed": changed_count,
            "failed": failed_count,
            "trace_id": trace_id,
            "timestamp": int(time.time()),
        }
        _LAST_SYNC_RESULT.clear()
        _LAST_SYNC_RESULT.update(result)
        return result
    except Exception as exc:
        logger.exception("[%s] knowledge sync failed: %s", trace_id, exc)
        result = {
            "success": False,
            "trigger": trigger,
            "scanned": scanned_count,
            "changed": changed_count,
            "failed": failed_count,
            "error": str(exc),
            "trace_id": trace_id,
            "timestamp": int(time.time()),
        }
        _LAST_SYNC_RESULT.clear()
        _LAST_SYNC_RESULT.update(result)
        return result


def run_permission_sync(trigger: str = "scheduler", trace_id: str = "") -> dict[str, Any]:
    """飞书权限同步（原型）。"""

    if not trace_id:
        trace_id = str(uuid.uuid4())

    logger.info("[%s] permission sync start, trigger=%s", trace_id, trigger)

    try:
        resp = feishu_client.list_drive_files(page_size=200)
        items = resp.get("data", {}).get("files", [])
        if not isinstance(items, list):
            items = []

        new_mapping: dict[str, dict[str, Any]] = {}

        for item in items:
            owner_id = str(item.get("owner_id") or item.get("owner", {}).get("id") or "")
            token = str(item.get("token") or item.get("file_token") or "")
            title = str(item.get("name") or item.get("title") or token)
            if not owner_id:
                continue

            if owner_id not in new_mapping:
                new_mapping[owner_id] = {"role": "owner", "docs": []}
            new_mapping[owner_id]["docs"].append({"token": token, "title": title})

        with _PERMISSION_LOCK:
            _PERMISSION_MAPPING.clear()
            _PERMISSION_MAPPING.update(new_mapping)

        _persist_permission_mapping()

        logger.info("[%s] permission sync done, users=%s", trace_id, len(new_mapping))
        result = {
            "success": True,
            "trigger": trigger,
            "users": len(new_mapping),
            "trace_id": trace_id,
            "timestamp": int(time.time()),
        }
        _LAST_PERMISSION_SYNC_RESULT.clear()
        _LAST_PERMISSION_SYNC_RESULT.update(result)
        return result
    except Exception as exc:
        logger.exception("[%s] permission sync failed: %s", trace_id, exc)
        result = {
            "success": False,
            "trigger": trigger,
            "users": 0,
            "error": str(exc),
            "trace_id": trace_id,
            "timestamp": int(time.time()),
        }
        _LAST_PERMISSION_SYNC_RESULT.clear()
        _LAST_PERMISSION_SYNC_RESULT.update(result)
        return result


def _is_knowledge_change_event(event_type: str) -> bool:
    """判断是否为飞书知识库变更事件。"""

    if not event_type:
        return False

    candidate_keywords = ["wiki", "doc", "document", "knowledge", "drive.file"]
    candidate_actions = ["created", "updated", "changed", "deleted", "edited"]

    et = event_type.lower()
    return any(k in et for k in candidate_keywords) and any(a in et for a in candidate_actions)


def _is_helpdesk_event(event_type: str) -> bool:
    """判断是否为飞书服务台事件。"""

    if not event_type:
        return False
    et = event_type.lower()
    return "helpdesk" in et or "service_desk" in et


def _estimate_answer_confidence(rag_result: dict[str, Any]) -> float:
    """估算 RAG 回答置信度（0~1，原型规则）。"""

    answer = str(rag_result.get("answer", ""))
    sources = rag_result.get("sources", [])
    anti = str(rag_result.get("anti_hallucination", ""))

    if not answer.strip():
        return 0.0
    if "不知道" in answer or "无法确定" in answer:
        return 0.3
    if anti in {"no_context_found", "fallback", "insufficient_context"}:
        return 0.45

    source_count = len(sources) if isinstance(sources, list) else 0
    if source_count >= 2:
        return 0.92
    if source_count == 1:
        return 0.82
    return 0.65


def _verify_admin_token(x_admin_token: str | None) -> None:
    """校验管理接口 token。"""

    if not ADMIN_API_TOKEN:
        return

    if not x_admin_token or x_admin_token.strip() != ADMIN_API_TOKEN:
        raise HTTPException(status_code=401, detail="管理接口鉴权失败")


@app.on_event("startup")
def _startup_scheduler() -> None:
    """应用启动时注册定时任务并加载状态。"""

    _load_persistent_states()
    _load_bitable_binding()
    _load_bitable_record_state()
    _load_bitable_sync_result()

    if scheduler.running:
        return

    scheduler.add_job(
        func=run_incremental_knowledge_sync,
        trigger="interval",
        minutes=10,
        kwargs={"trigger": "scheduler"},
        id="knowledge_incremental_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    scheduler.add_job(
        func=run_permission_sync,
        trigger="interval",
        hours=1,
        kwargs={"trigger": "scheduler"},
        id="permission_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    # 任务3：每10分钟同步所有开启自动同步的多维表格
    scheduler.add_job(
        func=run_bitable_incremental_sync,
        trigger="interval",
        minutes=10,
        kwargs={"trigger": "scheduler"},
        id="bitable_incremental_sync",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    scheduler.start()
    logger.info("APScheduler started with 3 jobs")


@app.on_event("shutdown")
def _shutdown_scheduler() -> None:
    """应用关闭时停止定时任务并落盘状态。"""

    _persist_sync_doc_state()
    _persist_permission_mapping()
    _persist_bitable_binding()
    _persist_bitable_record_state()
    _persist_bitable_sync_result()
    _persist_bitable_record_state()

    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin/sync/status")
def get_sync_status(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> dict[str, Any]:
    """查看最近一次知识库同步与权限同步状态。"""

    _verify_admin_token(x_admin_token)

    with _SYNC_DOC_LOCK:
        doc_state_size = len(_SYNC_DOC_STATE)
    with _PERMISSION_LOCK:
        permission_size = len(_PERMISSION_MAPPING)

    return {
        "knowledge_sync": dict(_LAST_SYNC_RESULT),
        "permission_sync": dict(_LAST_PERMISSION_SYNC_RESULT),
        "bitable_sync": dict(_LAST_BITABLE_SYNC_RESULT),
        "bitable_binding": dict(BITABLE_BINDING),
        "sync_doc_state_size": doc_state_size,
        "permission_mapping_size": permission_size,
        "scheduler_running": scheduler.running,
    }


@app.post("/admin/sync/run")
def run_sync_now(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> dict[str, Any]:
    """手动触发一次知识库增量同步与权限同步。"""

    _verify_admin_token(x_admin_token)

    trace_id = str(uuid.uuid4())
    knowledge_result = run_incremental_knowledge_sync(trigger="manual_api", trace_id=trace_id)
    permission_result = run_permission_sync(trigger="manual_api", trace_id=trace_id)

    return {
        "message": "手动同步已执行",
        "knowledge_sync": knowledge_result,
        "permission_sync": permission_result,
        "trace_id": trace_id,
    }


class BitableBindingRequest(BaseModel):
    app_token: str
    table_id: str
    max_records: int = 1000


@app.post("/admin/bitable/bind")
def bind_bitable_config(
    req: BitableBindingRequest,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict[str, Any]:
    """绑定并持久化多维表格增量同步配置。"""

    _verify_admin_token(x_admin_token)

    app_token = req.app_token.strip()
    table_id = req.table_id.strip()
    max_records = max(1, min(int(req.max_records), 50000))

    if not app_token:
        raise HTTPException(status_code=400, detail="app_token 不能为空")
    if not table_id:
        raise HTTPException(status_code=400, detail="table_id 不能为空")

    BITABLE_BINDING["app_token"] = app_token
    BITABLE_BINDING["table_id"] = table_id
    BITABLE_BINDING["max_records"] = max_records
    _persist_bitable_binding()

    return {
        "message": "多维表格绑定配置已保存",
        "bitable_binding": dict(BITABLE_BINDING),
    }


@app.post("/api/bitable/sync/{app_token}/{table_id}")
def sync_single_bitable_api(
    app_token: str,
    table_id: str,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict[str, Any]:
    """手动触发单表增量同步。"""

    _verify_admin_token(x_admin_token)

    trace_id = str(uuid.uuid4())
    try:
        res = _sync_single_bitable_table(app_token=app_token, table_id=table_id, table_name=table_id)
        return {
            "success": True,
            "trace_id": trace_id,
            "result": res,
        }
    except Exception as exc:
        return {
            "success": False,
            "trace_id": trace_id,
            "error": str(exc),
        }


class BitableAutoSyncRequest(BaseModel):
    app_token: str
    table_id: str
    auto_sync: bool


@app.post("/admin/bitable/auto-sync")
def update_bitable_auto_sync(
    req: BitableAutoSyncRequest,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict[str, Any]:
    """启停某张多维表格的自动同步。"""

    _verify_admin_token(x_admin_token)

    app_token = req.app_token.strip()
    table_id = req.table_id.strip()
    if not app_token or not table_id:
        raise HTTPException(status_code=400, detail="app_token/table_id 不能为空")

    bindings = _load_bitable_bindings()
    matched = False
    for b in bindings:
        if str(b.get("app_token") or "").strip() == app_token and str(b.get("table_id") or "").strip() == table_id:
            b["auto_sync"] = bool(req.auto_sync)
            matched = True
            break

    if not matched:
        raise HTTPException(status_code=404, detail="未找到对应绑定配置")

    _save_bitable_bindings(bindings)
    return {
        "message": "自动同步开关已更新",
        "app_token": app_token,
        "table_id": table_id,
        "auto_sync": bool(req.auto_sync),
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传文档并写入向量库。"""

    try:
        upload_dir = Path(CONFIG.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        filename = file.filename or "unknown.txt"
        target_path = upload_dir / filename

        content = await file.read()
        target_path.write_bytes(content)

        ingest_result = _ingest_local_file(target_path)
        return {
            "message": "文件上传并入库成功",
            "filename": filename,
            **ingest_result,
        }
    except (DocumentProcessorError, VectorStoreError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"未知错误: {exc}") from exc


@app.post("/ask")
def ask_question(req: AskRequest) -> dict[str, Any]:
    """RAG 问答接口。"""

    try:
        return rag_chain.ask(question=req.question, top_k=req.top_k)
    except RAGChainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"未知错误: {exc}") from exc


@app.post("/feishu/webhook")
async def feishu_webhook(request: Request) -> dict[str, Any]:
    """飞书事件回调接收。"""

    trace_id = request.headers.get("X-Trace-Id", "") or str(uuid.uuid4())

    raw_body = await request.body()
    body_text = raw_body.decode("utf-8", errors="ignore")

    timestamp = request.headers.get("X-Lark-Request-Timestamp", "")
    nonce = request.headers.get("X-Lark-Request-Nonce", "")
    signature = request.headers.get("X-Lark-Signature", "")

    logger.info("[%s] webhook received", trace_id)

    try:
        payload = json.loads(body_text) if body_text else {}

        # 飞书首次配置回调会先发 url_verification。
        # 为保证平台校验稳定通过，这里直接原样回传 challenge。
        if payload.get("type") == "url_verification":
            challenge = str(payload.get("challenge", ""))
            logger.info("[%s] url_verification challenge=%s", trace_id, challenge)
            return JSONResponse(content={"challenge": challenge}, media_type="application/json")
        is_valid = feishu_client.verify_event_signature(
            timestamp=timestamp,
            nonce=nonce,
            body=body_text,
            signature=signature,
        )
        if not is_valid:
            logger.warning("[%s] signature invalid", trace_id)
            raise HTTPException(status_code=401, detail="签名校验失败")

        header_event_type = str(payload.get("header", {}).get("event_type", ""))

        # 服务台事件优先处理：用户提问 -> RAG -> 自动回复，低置信度提示转人工
        if _is_helpdesk_event(header_event_type):
            helpdesk = feishu_client.parse_helpdesk_event(payload)
            event_id = str(helpdesk.get("event_id") or "")
            if _is_duplicate_event(event_id):
                logger.info("[%s] duplicated helpdesk event ignored: %s", trace_id, event_id)
                return {"message": "重复服务台事件已忽略", "event_id": event_id, "trace_id": trace_id}

            question = str(helpdesk.get("question") or "").strip()
            open_id = str(helpdesk.get("open_id") or "")

            if not question:
                feishu_client.send_helpdesk_text_reply(open_id, "您好，请描述您要咨询的问题。")
                return {"message": "服务台空问题已处理", "trace_id": trace_id}

            rag_result = rag_chain.ask(question=question, top_k=3)
            confidence = _estimate_answer_confidence(rag_result)
            answer = str(rag_result.get("answer", "根据当前知识库内容无法确定。"))

            escalate_result = {"success": False, "message": "not_needed"}
            if confidence < 0.8:
                answer = (
                    f"{answer}\n\n"
                    "当前答案置信度较低（<80%），建议转人工客服进一步处理。"
                )
                escalate_result = feishu_client.escalate_helpdesk_to_human(
                    ticket_id=str(helpdesk.get("ticket_id") or ""),
                    reason="RAG answer confidence lower than 80%",
                )

            feishu_client.send_helpdesk_text_reply(open_id, answer)
            return {
                "message": "服务台消息已自动回复",
                "confidence": confidence,
                "escalate": escalate_result,
                "trace_id": trace_id,
            }

        parsed = feishu_client.handle_message_or_file_event(payload)

        event_id = parsed.get("event_id", "")
        if _is_duplicate_event(event_id):
            logger.info("[%s] duplicated event ignored: %s", trace_id, event_id)
            return {"message": "重复事件已忽略", "event_id": event_id, "trace_id": trace_id}

        if _is_knowledge_change_event(header_event_type):
            sync_result = run_incremental_knowledge_sync(trigger="webhook_event", trace_id=trace_id)
            return {
                "message": "知识库变更事件已处理并触发增量同步",
                "event_type": header_event_type,
                "sync_result": sync_result,
                "trace_id": trace_id,
            }

        event_type = parsed.get("type")
        open_id = parsed.get("open_id", "")

        if event_type == "user_text_message":
            text = (parsed.get("text") or "").strip()
            if not text:
                _safe_reply_to_user(open_id, "我收到了空消息，请发送你的问题。", trace_id)
                return {"message": "收到空文本消息", "trace_id": trace_id}

            rag_result = rag_chain.ask(question=text, top_k=3)
            answer = rag_result.get("answer", "根据当前知识库内容无法确定。")
            _safe_reply_to_user(open_id, answer, trace_id)

            logger.info("[%s] text event processed", trace_id)
            return {
                "message": "文本消息已处理",
                "event_type": event_type,
                "question": text,
                "answer": answer,
                "trace_id": trace_id,
            }

        if event_type == "user_file_message":
            file_key = parsed.get("file_key", "")
            file_name = parsed.get("file_name", "uploaded_from_feishu")
            message_id = parsed.get("message_id", "")

            if not file_key or not message_id:
                _safe_reply_to_user(open_id, "未获取到 message_id 或 file_key，无法处理该文件。", trace_id)
                return {"message": "文件事件缺少必要字段", "trace_id": trace_id}

            upload_dir = Path(CONFIG.upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)

            suffix = Path(file_name).suffix or ".txt"
            safe_name = Path(file_name).stem or "feishu_file"
            local_path = upload_dir / f"feishu_{safe_name}{suffix}"

            try:
                file_bytes = feishu_client.download_user_file(message_id=message_id, file_key=file_key)
                local_path.write_bytes(file_bytes)
                ingest_result = _ingest_local_file(local_path)
                reply = f"文件已入库成功：{file_name}，切片数 {ingest_result['chunk_count']}。"
            except (FeishuClientError, DocumentProcessorError, VectorStoreError, OSError) as exc:
                reply = f"文件处理失败：{exc}"

            _safe_reply_to_user(open_id, reply, trace_id)
            logger.info("[%s] file event processed: %s", trace_id, file_name)

            return {
                "message": "文件消息已处理",
                "event_type": event_type,
                "file_name": file_name,
                "local_path": str(local_path),
                "trace_id": trace_id,
            }

        logger.info("[%s] unsupported event type: %s", trace_id, event_type)
        return {"message": "暂未处理该事件类型", "event": parsed, "trace_id": trace_id}

    except FeishuClientError as exc:
        logger.exception("[%s] feishu error: %s", trace_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        logger.exception("[%s] json decode error: %s", trace_id, exc)
        raise HTTPException(status_code=400, detail=f"非法 JSON: {exc}") from exc
    except RAGChainError as exc:
        logger.exception("[%s] rag error: %s", trace_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[%s] unknown error: %s", trace_id, exc)
        raise HTTPException(status_code=500, detail=f"未知错误: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
