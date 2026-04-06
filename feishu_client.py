"""
飞书 API 对接模块（机器人核心能力）

实现功能：
1) 飞书事件签名校验（Webhook 合法性验证）
2) 接收并解析用户消息事件/文件事件
3) 下载用户上传到飞书机器人的文件
4) 给用户发送文本回复消息
5) 获取飞书知识库文档列表与文档内容

说明：
- 本模块为“可直接调用”的基础实现，适合 FastAPI / Streamlit 原型快速接入。
- 已包含异常处理与字段兼容逻辑；生产环境可补充重试、幂等、审计日志。
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import requests

from config import CONFIG


class FeishuClientError(Exception):
    """飞书客户端异常。"""


class FeishuClient:
    """飞书开放平台基础客户端。"""

    def __init__(self) -> None:
        self.base_url = CONFIG.feishu.base_url.rstrip("/")
        self.app_id = CONFIG.feishu.app_id
        self.app_secret = CONFIG.feishu.app_secret
        self.verification_token = CONFIG.feishu.verification_token
        self.encrypt_key = CONFIG.feishu.encrypt_key
        self._tenant_access_token: str | None = None

    # =========================
    # 基础请求能力
    # =========================
    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """统一请求封装。

        - 自动拼接 base_url
        - 自动处理 HTTP 错误 / JSON 解析错误 / 飞书业务错误
        """

        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = requests.request(method=method, url=url, timeout=30, **kwargs)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise FeishuClientError(f"请求飞书接口失败: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise FeishuClientError("飞书接口返回非 JSON 数据") from exc

        if data.get("code", 0) != 0:
            raise FeishuClientError(f"飞书接口业务报错: {data}")
        return data

    def get_tenant_access_token(self, force_refresh: bool = False) -> str:
        """获取并缓存 tenant_access_token。"""

        if self._tenant_access_token and not force_refresh:
            return self._tenant_access_token

        payload = {"app_id": self.app_id, "app_secret": self.app_secret}
        data = self._request(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            json=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

        token = data.get("tenant_access_token")
        if not token:
            raise FeishuClientError("未获取到 tenant_access_token")

        self._tenant_access_token = token
        return token

    def _auth_headers(self, content_type_json: bool = False) -> dict[str, str]:
        """生成鉴权 Header。"""

        token = self.get_tenant_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        if content_type_json:
            headers["Content-Type"] = "application/json; charset=utf-8"
        return headers

    # =========================
    # 1) Webhook 事件合法性校验
    # =========================
    def verify_event_signature(self, timestamp: str, nonce: str, body: str, signature: str) -> bool:
        """校验飞书事件签名（HMAC-SHA256）。

        参数：
        - timestamp: 请求头 X-Lark-Request-Timestamp
        - nonce: 请求头 X-Lark-Request-Nonce
        - body: 请求原始字符串
        - signature: 请求头 X-Lark-Signature

        说明：
        - 若未配置 encrypt_key，开发环境默认放行。
        - 生产环境建议必须配置 encrypt_key 并启用校验。
        """

        if not self.encrypt_key:
            return True

        if not all([timestamp, nonce, body, signature]):
            return False

        # 轻量签名串规则：timestamp + nonce + encrypt_key + body
        raw = f"{timestamp}{nonce}{self.encrypt_key}{body}".encode("utf-8")
        digest = hmac.new(self.encrypt_key.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        return hmac.compare_digest(digest, signature)

    def verify_event_token(self, payload: dict[str, Any]) -> bool:
        """校验事件体 token（verification_token）。"""

        if not self.verification_token:
            return True
        return payload.get("token", "") == self.verification_token

    # =========================
    # 2) 接收消息/文件事件
    # =========================
    def parse_incoming_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """解析飞书事件回调。

        返回统一结构：
        - url_verification: {type, challenge}
        - event_callback:   {type, event_id, event_type, event, raw}
        """

        if not isinstance(payload, dict):
            raise FeishuClientError("payload 必须是字典")

        if not self.verify_event_token(payload):
            raise FeishuClientError("事件 token 校验失败")

        if payload.get("type") == "url_verification":
            return {
                "type": "url_verification",
                "challenge": payload.get("challenge", ""),
            }

        header = payload.get("header", {})
        event = payload.get("event", {})

        return {
            "type": "event_callback",
            "event_id": header.get("event_id"),
            "event_type": header.get("event_type"),
            "create_time": header.get("create_time"),
            "event": event,
            "raw": payload,
        }

    def handle_message_or_file_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """处理用户消息事件，统一抽取文本/文件关键字段。"""

        parsed = self.parse_incoming_event(payload)
        if parsed.get("type") != "event_callback":
            return parsed

        event = parsed.get("event", {}) if isinstance(parsed.get("event"), dict) else {}
        message = event.get("message", {}) if isinstance(event.get("message"), dict) else {}

        message_type = message.get("message_type", "")
        message_id = message.get("message_id", "")
        chat_id = message.get("chat_id", "")
        content_raw = message.get("content", "{}")

        try:
            content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
            if not isinstance(content, dict):
                content = {}
        except json.JSONDecodeError:
            content = {}

        sender = event.get("sender", {}) if isinstance(event.get("sender"), dict) else {}
        sender_id = sender.get("sender_id", {}) if isinstance(sender.get("sender_id"), dict) else {}

        base_info = {
            "event_id": parsed.get("event_id"),
            "message_id": message_id,
            "chat_id": chat_id,
            "open_id": sender_id.get("open_id", ""),
            "user_id": sender_id.get("user_id", ""),
            "union_id": sender_id.get("union_id", ""),
            "message_type": message_type,
            "raw_content": content,
        }

        if message_type == "text":
            return {
                "type": "user_text_message",
                **base_info,
                "text": content.get("text", ""),
            }

        if message_type == "file":
            # 飞书 file 消息常见字段
            return {
                "type": "user_file_message",
                **base_info,
                "file_key": content.get("file_key", ""),
                "file_name": content.get("file_name", ""),
            }

        return {
            "type": "unsupported_message_type",
            **base_info,
        }

    # =========================
    # 3) 下载用户上传文件
    # =========================
    def download_user_file(self, message_id: str, file_key: str, file_type: str = "file") -> bytes:
        """下载用户发送给机器人的文件。

        参数：
        - message_id: 消息事件中的 message_id
        - file_key:   消息 content 中的 file_key
        - file_type:  资源类型，默认 file（常见还有 image/audio/video）

        飞书接口：GET /im/v1/messages/:message_id/resources/:file_key?type=file
        """

        if not message_id:
            raise FeishuClientError("message_id 不能为空")
        if not file_key:
            raise FeishuClientError("file_key 不能为空")

        path = f"/im/v1/messages/{message_id}/resources/{file_key}?type={file_type}"
        url = f"{self.base_url}/{path.lstrip('/')}"

        try:
            resp = requests.get(url, headers=self._auth_headers(), timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            raise FeishuClientError(f"下载用户文件失败: {exc}") from exc

    # 兼容你现有后台里的导入调用（drive 文件 token 下载）
    def download_file(self, file_token: str) -> bytes:
        """下载飞书 Drive 文件（二进制）。"""

        if not file_token:
            raise FeishuClientError("file_token 不能为空")

        url = f"{self.base_url}/drive/v1/medias/{file_token}/download"
        try:
            resp = requests.get(url, headers=self._auth_headers(), timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            raise FeishuClientError(f"下载飞书文件失败: {exc}") from exc

    # =========================
    # 4) 发送文本回复
    # =========================
    def send_text_message(self, receive_id: str, text: str, receive_id_type: str = "open_id") -> dict[str, Any]:
        """发送文本消息给用户/群聊。"""

        if not receive_id:
            raise FeishuClientError("receive_id 不能为空")
        if not text or not text.strip():
            raise FeishuClientError("text 不能为空")

        payload = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }

        return self._request(
            "POST",
            f"/im/v1/messages?receive_id_type={receive_id_type}",
            headers=self._auth_headers(content_type_json=True),
            json=payload,
        )

    # =========================
    # 5) 飞书知识库文档列表与内容
    # =========================
    def list_knowledge_docs(self, space_id: str, page_size: int = 50, page_token: str = "") -> dict[str, Any]:
        """获取飞书知识库（Wiki）文档列表。"""

        if not space_id:
            raise FeishuClientError("space_id 不能为空")

        params = [f"page_size={max(1, min(page_size, 200))}"]
        if page_token:
            params.append(f"page_token={page_token}")

        path = f"/wiki/v2/spaces/{space_id}/nodes?{'&'.join(params)}"
        return self._request("GET", path, headers=self._auth_headers())

    def get_knowledge_doc_content(self, doc_token: str) -> dict[str, Any]:
        """获取飞书知识库单文档内容（Wiki 节点信息）。"""

        if not doc_token:
            raise FeishuClientError("doc_token 不能为空")

        return self._request(
            "GET",
            f"/wiki/v2/spaces/get_node?token={doc_token}",
            headers=self._auth_headers(),
        )

    # 兼容后台“飞书文档导入”场景：网盘文件列表
    def list_drive_files(self, page_size: int = 50, page_token: str = "") -> dict[str, Any]:
        """获取飞书网盘文件列表。"""

        params = [f"page_size={max(1, min(page_size, 200))}"]
        if page_token:
            params.append(f"page_token={page_token}")

        path = f"/drive/v1/files?{'&'.join(params)}"
        return self._request("GET", path, headers=self._auth_headers())

    # =========================
    # 多维表格（Bitable）
    # =========================
    def list_bitable_tables(self, app_token: str, page_size: int = 100, page_token: str = "") -> dict[str, Any]:
        """获取多维表格中的表列表。"""

        if not app_token:
            raise FeishuClientError("app_token 不能为空")

        params = [f"page_size={max(1, min(page_size, 500))}"]
        if page_token:
            params.append(f"page_token={page_token}")

        path = f"/bitable/v1/apps/{app_token}/tables?{'&'.join(params)}"
        return self._request("GET", path, headers=self._auth_headers())

    def get_bitable_structure(self, app_token: str, table_id: str) -> dict[str, Any]:
        """获取多维表格结构（表头字段）。

        返回：
        - field_name_to_id: 字段名 -> 字段 ID
        - field_id_to_name: 字段 ID -> 字段名
        - fields: 原始字段列表（便于上层保留更多属性）
        """

        fields_resp = self.get_bitable_table_fields(app_token=app_token, table_id=table_id, page_size=500)
        items = fields_resp.get("data", {}).get("items", [])
        if not isinstance(items, list):
            items = []

        field_name_to_id: dict[str, str] = {}
        field_id_to_name: dict[str, str] = {}

        for f in items:
            if not isinstance(f, dict):
                continue
            field_id = str(f.get("field_id") or "").strip()
            field_name = str(f.get("field_name") or field_id).strip()
            if not field_id:
                continue

            field_name_to_id[field_name] = field_id
            field_id_to_name[field_id] = field_name

        return {
            "field_name_to_id": field_name_to_id,
            "field_id_to_name": field_id_to_name,
            "fields": items,
        }

    def get_bitable_table_fields(self, app_token: str, table_id: str, page_size: int = 200) -> dict[str, Any]:
        """获取多维表格字段结构。"""

        if not app_token:
            raise FeishuClientError("app_token 不能为空")
        if not table_id:
            raise FeishuClientError("table_id 不能为空")

        path = f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields?page_size={max(1, min(page_size, 500))}"
        return self._request("GET", path, headers=self._auth_headers())

    def list_bitable_records(
        self,
        app_token: str,
        table_id: str,
        page_size: int = 200,
        page_token: str = "",
        view_id: str = "",
    ) -> dict[str, Any]:
        """获取多维表格数据记录。"""

        if not app_token:
            raise FeishuClientError("app_token 不能为空")
        if not table_id:
            raise FeishuClientError("table_id 不能为空")

        params = [f"page_size={max(1, min(page_size, 500))}"]
        if page_token:
            params.append(f"page_token={page_token}")
        if view_id:
            params.append(f"view_id={view_id}")

        path = f"/bitable/v1/apps/{app_token}/tables/{table_id}/records?{'&'.join(params)}"
        return self._request("GET", path, headers=self._auth_headers())

    def get_bitable_records(self, app_token: str, table_id: str, page_size: int = 200, view_id: str = "") -> list[dict[str, Any]]:
        """分页获取多维表格全部记录。

        参数：
        - app_token: 多维表格应用 token
        - table_id:  表 ID
        - page_size: 单页大小（1~500）
        - view_id:   可选，指定视图拉取

        返回：
        - 完整记录列表（聚合所有分页）
        """

        if not app_token:
            raise FeishuClientError("app_token 不能为空")
        if not table_id:
            raise FeishuClientError("table_id 不能为空")

        all_records: list[dict[str, Any]] = []
        next_page_token = ""

        while True:
            resp = self.list_bitable_records(
                app_token=app_token,
                table_id=table_id,
                page_size=page_size,
                page_token=next_page_token,
                view_id=view_id,
            )
            data = resp.get("data", {})
            items = data.get("items", [])
            if isinstance(items, list):
                all_records.extend([x for x in items if isinstance(x, dict)])

            has_more = bool(data.get("has_more", False))
            next_page_token = str(data.get("page_token", ""))
            if not has_more or not next_page_token:
                break

        return all_records

    def get_single_record(self, app_token: str, table_id: str, record_id: str) -> dict[str, Any]:
        """获取多维表格单条记录详情。"""

        if not app_token:
            raise FeishuClientError("app_token 不能为空")
        if not table_id:
            raise FeishuClientError("table_id 不能为空")
        if not record_id:
            raise FeishuClientError("record_id 不能为空")

        path = f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}"
        resp = self._request("GET", path, headers=self._auth_headers())
        return resp.get("data", {}).get("record", {})

    def bitable_to_structured_text(
        self,
        app_token: str,
        table_id: str,
        table_name: str = "",
        max_records: int = 1000,
    ) -> dict[str, Any]:
        """将多维表格结构和数据转换为可入库的结构化文本。"""

        structure = self.get_bitable_structure(app_token=app_token, table_id=table_id)
        field_names = list(structure.get("field_name_to_id", {}).keys())

        all_records = self.get_bitable_records(app_token=app_token, table_id=table_id, page_size=200)
        all_records = all_records[:max_records]

        lines: list[str] = []
        title = table_name or table_id
        lines.append(f"[飞书多维表格] {title}")
        lines.append(f"字段: {', '.join(field_names) if field_names else '无'}")
        lines.append("数据:")

        for idx, rec in enumerate(all_records, start=1):
            fields_map = rec.get("fields", {}) if isinstance(rec, dict) else {}
            if not isinstance(fields_map, dict):
                fields_map = {}

            kv_parts: list[str] = []
            for key, value in fields_map.items():
                if isinstance(value, (dict, list)):
                    val_text = json.dumps(value, ensure_ascii=False)
                else:
                    val_text = str(value)
                kv_parts.append(f"{key}={val_text}")

            lines.append(f"- 记录{idx}: " + " | ".join(kv_parts))

        return {
            "text": "\n".join(lines),
            "metadata": {
                "source": "feishu_bitable",
                "app_token": app_token,
                "table_id": table_id,
                "table_name": table_name or table_id,
                "record_count": len(all_records),
            },
        }

    def delete_bitable_record_from_vector(self, app_token: str, table_id: str, record_id: str) -> dict[str, Any]:
        """根据记录 ID 删除向量库中对应条目。

        说明：
        - 该函数会尝试按 metadata 条件删除：source/app_token/table_id/record_id。
        - 要求入库时在 metadata 中保留上述字段，才能精准删除。
        - 复用现有 tenant_access_token 认证体系（本函数会先校验该记录在飞书侧是否存在/可访问）。
        """

        if not app_token:
            raise FeishuClientError("app_token 不能为空")
        if not table_id:
            raise FeishuClientError("table_id 不能为空")
        if not record_id:
            raise FeishuClientError("record_id 不能为空")

        # 先访问飞书记录详情，复用鉴权能力并验证参数有效性
        _ = self.get_single_record(app_token=app_token, table_id=table_id, record_id=record_id)

        try:
            from vector_store import VectorStoreManager

            vs = VectorStoreManager()
            where = {
                "$and": [
                    {"app_token": app_token},
                    {"table_id": table_id},
                    {"record_id": record_id},
                ]
            }
            vs.vs.delete(where=where)  # noqa: SLF001
            return {
                "success": True,
                "message": "向量库记录删除完成",
                "app_token": app_token,
                "table_id": table_id,
                "record_id": record_id,
            }
        except Exception as exc:
            raise FeishuClientError(f"删除向量库记录失败: {exc}") from exc

    # =========================
    # 飞书服务台（Helpdesk）
    # =========================
    def parse_helpdesk_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """解析服务台消息事件。"""

        if not isinstance(payload, dict):
            raise FeishuClientError("payload 必须是字典")

        if not self.verify_event_token(payload):
            raise FeishuClientError("事件 token 校验失败")

        if payload.get("type") == "url_verification":
            return {"type": "url_verification", "challenge": payload.get("challenge", "")}

        header = payload.get("header", {}) if isinstance(payload.get("header"), dict) else {}
        event = payload.get("event", {}) if isinstance(payload.get("event"), dict) else {}

        # 兼容常见服务台消息字段
        text = str(event.get("text") or event.get("question") or "").strip()
        if not text:
            message = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
            content_raw = message.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content_obj = json.loads(content_raw)
                    text = str(content_obj.get("text") or "").strip()
                except json.JSONDecodeError:
                    text = content_raw.strip()

        open_id = str(event.get("open_id") or event.get("user_open_id") or "")
        ticket_id = str(event.get("ticket_id") or event.get("conversation_id") or "")

        return {
            "type": "helpdesk_message",
            "event_id": header.get("event_id"),
            "event_type": header.get("event_type"),
            "open_id": open_id,
            "ticket_id": ticket_id,
            "question": text,
            "raw": payload,
        }

    def send_helpdesk_text_reply(self, open_id: str, text: str) -> dict[str, Any]:
        """服务台场景文本回复（原型复用 IM 文本消息）。"""

        return self.send_text_message(receive_id=open_id, text=text, receive_id_type="open_id")

    def escalate_helpdesk_to_human(self, ticket_id: str, reason: str = "RAG 低置信度，建议人工介入") -> dict[str, Any]:
        """尝试将服务台会话升级/转人工。

        说明：
        - 不同企业租户的服务台接口能力与路径可能有差异。
        - 这里提供“可直接调用”的标准化封装：成功返回 success=True；失败返回 success=False 且不抛异常，
          以保证 webhook 主流程稳定。
        """

        if not ticket_id:
            return {"success": False, "message": "missing ticket_id"}

        # 常见转人工接口路径（若租户未开通该能力会返回业务错误）
        path = f"/helpdesk/v1/tickets/{ticket_id}/escalate"
        payload = {"reason": reason}

        try:
            resp = self._request(
                "POST",
                path,
                headers=self._auth_headers(content_type_json=True),
                json=payload,
            )
            return {"success": True, "message": "escalated", "data": resp}
        except Exception as exc:
            return {"success": False, "message": str(exc)}
