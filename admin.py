"""
Streamlit 管理后台（RAG 知识库主入口）

功能：
1) 简易管理员登录（固定密码，原型）
2) 知识库管理：创建 / 删除 / 切换（独立 Chroma collection）
3) 批量文件上传：PDF / Word / Excel / TXT，支持拖拽，单次最多 100 个
4) 处理进度展示：解析 + 分块 + 向量入库全过程
5) 切片管理：查看、手动修改、合并、删除
6) 飞书文档导入：拉取列表，多选导入
"""

from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from config import CONFIG
from document_processor import DocumentProcessor, DocumentProcessorError
from feishu_client import FeishuClient, FeishuClientError
from rag_chain import RAGChain
from vector_store import VectorStoreError, VectorStoreManager

# =========================
# 基础配置
# =========================
st.set_page_config(page_title="飞书RAG知识库管理后台", layout="wide")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ENABLE_ADMIN_LOGIN = os.getenv("ENABLE_ADMIN_LOGIN", "false").strip().lower() in {"1", "true", "yes", "y"}


# =========================
# 会话状态初始化
# =========================
def init_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_kb" not in st.session_state:
        st.session_state.current_kb = CONFIG.vector_store.collection_name
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if "feishu_client" not in st.session_state:
        st.session_state.feishu_client = FeishuClient()

    # 向量库初始化（模型路径缺失时给出可读提示，不直接崩溃）
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager(collection_name=st.session_state.current_kb)

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = RAGChain(vector_store=st.session_state.vector_store)


try:
    init_state()
except VectorStoreError as exc:
    st.error(f"向量库初始化失败：{exc}")
    st.info(
        "请先在项目目录放置本地嵌入模型，并在 .env 中设置 EMBEDDING_MODEL 为相对路径，"
        "例如：EMBEDDING_MODEL=./models/bge-m3"
    )
    st.stop()


# =========================
# 工具函数
# =========================
def refresh_chain() -> None:
    """切换知识库后，刷新 RAGChain 实例。"""

    st.session_state.rag_chain = RAGChain(vector_store=st.session_state.vector_store)


def save_uploaded_file(uploaded_file: Any) -> Path:
    """将上传文件落地到 upload_dir。"""

    upload_dir = Path(CONFIG.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def process_single_file(file_path: Path, vector_store: VectorStoreManager, doc_processor: DocumentProcessor) -> tuple[int, list[str]]:
    """处理单文件：解析 -> 分块 -> 入库。"""

    parsed = doc_processor.parse_file(str(file_path))
    chunks = doc_processor.split_text(parsed.text, parsed.metadata)
    ids = vector_store.add_documents(chunks)
    return len(chunks), ids


def _auto_ingest_state_path() -> Path:
    p = Path("./data/state")
    p.mkdir(parents=True, exist_ok=True)
    return p / "auto_ingest_state.json"


def _calc_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_local_data_files(data_dir: Path) -> list[Path]:
    supported = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt"}
    files: list[Path] = []
    if not data_dir.exists():
        return files

    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in supported:
            continue
        # 跳过向量库和状态目录下文件
        if "chroma_db" in p.parts or "state" in p.parts:
            continue
        files.append(p)
    return files


def auto_ingest_data_files(vector_store: VectorStoreManager, doc_processor: DocumentProcessor) -> dict[str, Any]:
    """自动扫描 data 目录并增量入库（基于文件哈希缓存）。"""

    data_dir = Path("./data")
    state_path = _auto_ingest_state_path()

    old_state: dict[str, str] = {}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                old_state = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            old_state = {}

    files = _list_local_data_files(data_dir)
    processed = 0
    skipped = 0
    failed = 0
    details: list[str] = []
    new_state = dict(old_state)

    for f in files:
        rel = f.as_posix()
        try:
            digest = _calc_file_sha256(f)
            if old_state.get(rel) == digest:
                skipped += 1
                continue

            chunk_count, _ = process_single_file(f, vector_store, doc_processor)
            new_state[rel] = digest
            processed += 1
            details.append(f"✅ {f.name} -> 切片 {chunk_count}")
        except Exception as exc:
            failed += 1
            details.append(f"❌ {f.name} -> {exc}")

    # 删除已不存在文件的状态
    current_keys = {p.as_posix() for p in files}
    for k in list(new_state.keys()):
        if k not in current_keys:
            new_state.pop(k, None)

    state_path.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "scanned": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }


def _bindings_path() -> Path:
    p = Path("./data/state")
    p.mkdir(parents=True, exist_ok=True)
    return p / "bitable_bindings_admin.json"


def load_bitable_bindings() -> list[dict[str, Any]]:
    """读取多维表格绑定列表。"""

    path = _bindings_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_bitable_bindings(bindings: list[dict[str, Any]]) -> None:
    """保存多维表格绑定列表。"""

    path = _bindings_path()
    path.write_text(json.dumps(bindings, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_bitable_binding(
    binding: dict[str, Any],
    feishu_client: FeishuClient,
    doc_processor: DocumentProcessor,
    vector_store: VectorStoreManager,
) -> tuple[bool, int, str]:
    """对单个绑定执行全量同步（行级分块）。"""

    app_token = str(binding.get("app_token") or "").strip()
    table_id = str(binding.get("table_id") or "").strip()
    table_name = str(binding.get("table_name") or table_id).strip()

    if not app_token or not table_id:
        return False, 0, "app_token/table_id 不能为空"

    try:
        structure = feishu_client.get_bitable_structure(app_token=app_token, table_id=table_id)
        headers = structure.get("field_name_to_id", {})
        records = feishu_client.get_bitable_records(app_token=app_token, table_id=table_id)

        chunks, metadatas = doc_processor.process_bitable_data(
            headers=headers,
            records=records,
            app_token=app_token,
            table_id=table_id,
            table_name=table_name,
        )

        vector_store.delete_bitable_table(app_token=app_token, table_id=table_id, collection_name=st.session_state.current_kb)

        if not chunks:
            return True, 0, "无有效记录可同步"

        vector_store.add_bitable_to_vectorstore(chunks, metadatas, collection_name=st.session_state.current_kb)
        return True, len(chunks), "同步成功"
    except (FeishuClientError, DocumentProcessorError, VectorStoreError) as exc:
        return False, 0, str(exc)


# =========================
# 登录（默认关闭，便于 Streamlit Cloud 直接使用）
# =========================
st.title("飞书 RAG 企业知识库管理后台")

if ENABLE_ADMIN_LOGIN:
    if not st.session_state.logged_in:
        st.subheader("管理员登录")
        with st.form("login_form"):
            pwd = st.text_input("请输入管理员密码", type="password")
            login_btn = st.form_submit_button("登录")

            if login_btn:
                if pwd == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.success("登录成功")
                    st.rerun()
                else:
                    st.error("密码错误")

        st.stop()

    # 已登录后显示退出按钮
    with st.sidebar:
        st.success("已登录")
        if st.button("退出登录"):
            st.session_state.logged_in = False
            st.rerun()
else:
    # 免登录模式
    st.session_state.logged_in = True
    with st.sidebar:
        st.info("当前为免登录模式（ENABLE_ADMIN_LOGIN=false）")

vector_store: VectorStoreManager = st.session_state.vector_store
doc_processor: DocumentProcessor = st.session_state.doc_processor
rag_chain: RAGChain = st.session_state.rag_chain
feishu_client: FeishuClient = st.session_state.feishu_client

# 首次进入页面自动加载 data 目录文档（增量+缓存）
if "auto_ingest_done" not in st.session_state:
    with st.spinner("正在自动加载 data 目录文档（首次可能稍慢）..."):
        auto_result = auto_ingest_data_files(vector_store, doc_processor)
    st.session_state.auto_ingest_done = True
    st.session_state.auto_ingest_result = auto_result


# =========================
# 1) 知识库管理
# =========================
st.subheader("1) 知识库管理")

auto_result = st.session_state.get("auto_ingest_result", {})
if auto_result:
    st.caption(
        f"自动加载状态：扫描 {auto_result.get('scanned', 0)} 个文件，"
        f"新增/更新 {auto_result.get('processed', 0)}，跳过 {auto_result.get('skipped', 0)}，失败 {auto_result.get('failed', 0)}"
    )
    details = auto_result.get("details", []) if isinstance(auto_result.get("details"), list) else []
    if details:
        with st.expander("查看自动导入详情"):
            st.text("\n".join(details[:200]))

reingest_col1, reingest_col2 = st.columns([1, 5])
with reingest_col1:
    if st.button("重新自动导入 data", help="清空自动导入缓存并重新扫描 ./data 文档"):
        try:
            state_path = _auto_ingest_state_path()
            if state_path.exists():
                state_path.unlink(missing_ok=True)

            with st.spinner("正在重新扫描并导入 data 目录..."):
                auto_result = auto_ingest_data_files(vector_store, doc_processor)

            st.session_state.auto_ingest_done = True
            st.session_state.auto_ingest_result = auto_result
            st.success(
                f"重导入完成：扫描 {auto_result.get('scanned', 0)}，"
                f"新增/更新 {auto_result.get('processed', 0)}，"
                f"跳过 {auto_result.get('skipped', 0)}，失败 {auto_result.get('failed', 0)}"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"重新自动导入失败：{exc}")
with reingest_col2:
    st.caption("如已清空知识库，请点击“重新自动导入 data”重新把 ./data 文件入库。")

col1, col2, col3 = st.columns([2, 2, 2])

try:
    kb_list = vector_store.list_collections()
except VectorStoreError as exc:
    kb_list = [st.session_state.current_kb]
    st.warning(f"获取知识库列表失败：{exc}")

if st.session_state.current_kb not in kb_list:
    kb_list.append(st.session_state.current_kb)

with col1:
    selected_kb = st.selectbox(
        "切换知识库",
        options=kb_list,
        index=kb_list.index(st.session_state.current_kb),
        key="switch_kb_name",
    )
    if st.button("应用切换"):
        try:
            vector_store.switch_collection(selected_kb)
            st.session_state.current_kb = selected_kb
            refresh_chain()
            st.success(f"已切换到知识库：{selected_kb}")
        except VectorStoreError as exc:
            st.error(f"切换失败：{exc}")

with col2:
    new_kb_name = st.text_input("新建知识库名称")
    if st.button("创建知识库"):
        try:
            vector_store.create_collection(new_kb_name)
            st.session_state.current_kb = new_kb_name.strip()
            refresh_chain()
            st.success(f"创建成功：{new_kb_name}")
            st.rerun()
        except VectorStoreError as exc:
            st.error(f"创建失败：{exc}")

with col3:
    del_kb_name = st.selectbox("删除知识库", options=kb_list, key="del_kb_name")
    if st.button("删除知识库"):
        try:
            vector_store.delete_collection(del_kb_name)
            st.session_state.current_kb = CONFIG.vector_store.collection_name
            vector_store.switch_collection(st.session_state.current_kb)
            refresh_chain()
            st.success(f"已删除知识库：{del_kb_name}")
            st.rerun()
        except VectorStoreError as exc:
            st.error(f"删除失败：{exc}")

st.caption(f"当前知识库：{st.session_state.current_kb}")

st.divider()

# =========================
# 2) 文件上传与处理进度
# =========================
st.subheader("2) 文件上传（支持拖拽，单次最多 100 个）")
uploaded_files = st.file_uploader(
    "支持 PDF / DOCX / DOC / XLSX / XLS / TXT",
    type=["pdf", "docx", "doc", "xlsx", "xls", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 100:
        st.error("单次最多上传 100 个文件，请减少后重试。")
    else:
        if st.button("开始处理并入库"):
            progress = st.progress(0)
            status = st.empty()
            result_box = st.empty()

            success_count = 0
            fail_count = 0
            details: list[str] = []

            total = len(uploaded_files)
            for idx, up in enumerate(uploaded_files, start=1):
                status.info(f"处理中：{up.name} ({idx}/{total})")
                try:
                    file_path = save_uploaded_file(up)
                    chunk_count, ids = process_single_file(file_path, vector_store, doc_processor)
                    details.append(f"✅ {up.name} -> 切片 {chunk_count}，入库 {len(ids)}")
                    success_count += 1
                except (DocumentProcessorError, VectorStoreError, OSError) as exc:
                    details.append(f"❌ {up.name} -> 失败：{exc}")
                    fail_count += 1
                finally:
                    progress.progress(int(idx / total * 100))

            status.success("处理完成")
            result_box.text("\n".join(details))
            st.success(f"完成：成功 {success_count} 个，失败 {fail_count} 个")

st.divider()

# =========================
# 3) 切片管理
# =========================
st.subheader("3) 切片管理")

filter_filename = st.text_input("按文件名过滤（可选）")
if st.button("刷新切片列表"):
    st.rerun()

try:
    chunks = vector_store.list_chunks(filename=filter_filename or None, limit=300)
except VectorStoreError as exc:
    chunks = []
    st.error(f"读取切片失败：{exc}")

st.caption(f"当前展示切片数量：{len(chunks)}")

# 一键清空当前知识库
if st.button("一键清空当前知识库切片"):
    try:
        deleted_count = vector_store.clear_current_collection()
        st.success(f"已清空完成，共删除 {deleted_count} 个切片")
        st.rerun()
    except VectorStoreError as exc:
        st.error(f"清空失败：{exc}")

if st.button("一键清空并重导入 data", help="清空当前知识库后，删除自动导入缓存并重新扫描 ./data"):
    try:
        deleted_count = vector_store.clear_current_collection()
        state_path = _auto_ingest_state_path()
        if state_path.exists():
            state_path.unlink(missing_ok=True)

        with st.spinner("正在重导入 data 目录..."):
            auto_result = auto_ingest_data_files(vector_store, doc_processor)

        st.session_state.auto_ingest_done = True
        st.session_state.auto_ingest_result = auto_result
        st.success(
            f"重建完成：已清空 {deleted_count} 个切片；"
            f"扫描 {auto_result.get('scanned', 0)}，"
            f"新增/更新 {auto_result.get('processed', 0)}，"
            f"跳过 {auto_result.get('skipped', 0)}，失败 {auto_result.get('failed', 0)}"
        )
        st.rerun()
    except Exception as exc:
        st.error(f"一键重建失败：{exc}")

if chunks:
    chunk_options = [f"{c['id']} | {c.get('metadata', {}).get('filename', 'unknown')}" for c in chunks]

    # 3.1 修改切片
    st.markdown("#### 3.1 手动修改切片")
    selected_edit = st.selectbox("选择要修改的切片", options=chunk_options, key="edit_chunk")
    edit_id = selected_edit.split(" | ")[0]
    edit_chunk = next((c for c in chunks if c["id"] == edit_id), None)
    default_text = edit_chunk.get("text", "") if edit_chunk else ""
    new_text = st.text_area("编辑切片文本", value=default_text, height=180)

    if st.button("保存修改"):
        try:
            vector_store.update_chunk(chunk_id=edit_id, new_text=new_text, metadata=edit_chunk.get("metadata", {}))
            st.success("切片修改成功")
            st.rerun()
        except VectorStoreError as exc:
            st.error(f"修改失败：{exc}")

    # 3.2 合并切片
    st.markdown("#### 3.2 合并切片")
    merge_selected = st.multiselect("选择要合并的切片（至少2个）", options=chunk_options)
    merge_text = st.text_area("合并后文本", height=180)

    if st.button("执行合并"):
        try:
            merge_ids = [x.split(" | ")[0] for x in merge_selected]
            if not merge_text.strip() and len(merge_ids) >= 2:
                # 如果未填写合并文本，自动拼接
                selected_chunks = [c for c in chunks if c["id"] in merge_ids]
                merge_text = "\n\n".join([c.get("text", "") for c in selected_chunks])

            new_id = vector_store.merge_chunks(chunk_ids=merge_ids, merged_text=merge_text)
            st.success(f"合并成功，新切片 ID：{new_id}")
            st.rerun()
        except VectorStoreError as exc:
            st.error(f"合并失败：{exc}")

    # 3.3 删除切片
    st.markdown("#### 3.3 删除切片")
    select_all_delete = st.checkbox("全选当前列表切片", key="select_all_delete")

    default_delete = chunk_options if select_all_delete else []
    delete_selected = st.multiselect("选择要删除的切片", options=chunk_options, default=default_delete, key="delete_chunks")

    if st.button("删除选中切片"):
        try:
            del_ids = [x.split(" | ")[0] for x in delete_selected]
            if not del_ids:
                st.warning("请先选择要删除的切片")
            else:
                vector_store.delete_by_ids(del_ids)
                st.success(f"已删除 {len(del_ids)} 个切片")
                st.rerun()
        except VectorStoreError as exc:
            st.error(f"删除失败：{exc}")

    with st.expander("查看切片详情（前20条）"):
        for item in chunks[:20]:
            st.markdown(f"**ID:** {item['id']}")
            st.caption(str(item.get("metadata", {})))
            st.write(item.get("text", ""))
            st.divider()

st.divider()

# =========================
# 4) 飞书文档导入
# =========================
st.subheader("4) 飞书文档导入")

st.info("此区域支持两种方式：① 拉取网盘文档列表后多选导入；② 直接输入 file_token 手动导入。")

col_doc1, col_doc2 = st.columns([2, 1])
with col_doc1:
    if st.button("拉取飞书文档列表"):
        try:
            files_resp = feishu_client.list_drive_files(page_size=100)
            data = files_resp.get("data", {}) if isinstance(files_resp.get("data"), dict) else {}
            items = data.get("files", []) or data.get("items", [])
            if not isinstance(items, list):
                items = []
            st.session_state.feishu_files = items
            if not items:
                st.warning("未获取到飞书文档列表。请检查：应用权限、文档可见范围、云文档是否在当前租户。")
            else:
                st.success(f"已获取 {len(items)} 个文档")
        except FeishuClientError as exc:
            st.error(f"拉取飞书文档失败：{exc}")

with col_doc2:
    manual_token = st.text_input("手动导入 file_token", key="manual_file_token")
    if st.button("按 token 导入文档"):
        if not manual_token.strip():
            st.warning("请先输入 file_token")
        else:
            try:
                binary = feishu_client.download_file(manual_token.strip())
                target = Path(CONFIG.upload_dir)
                target.mkdir(parents=True, exist_ok=True)
                file_path = target / f"feishu_manual_{manual_token.strip()}.txt"
                file_path.write_bytes(binary)
                chunk_count, _ = process_single_file(file_path, vector_store, doc_processor)
                st.success(f"手动导入成功，切片 {chunk_count}")
            except (FeishuClientError, DocumentProcessorError, VectorStoreError, OSError) as exc:
                st.error(f"手动导入失败：{exc}")

feishu_items = st.session_state.get("feishu_files", [])
if feishu_items:
    options_map: dict[str, dict[str, str]] = {}
    for f in feishu_items:
        token = str(f.get("token") or f.get("file_token") or f.get("obj_token") or "")
        name = str(f.get("name") or f.get("title") or token)
        if token:
            options_map[f"{name} ({token})"] = {"token": token, "name": name}

    selected_docs = st.multiselect("选择要导入的飞书文档", options=list(options_map.keys()))

    if st.button("导入选中文档"):
        if not selected_docs:
            st.warning("请先选择至少一个文档")
        else:
            progress = st.progress(0)
            logs: list[str] = []
            total = len(selected_docs)
            for idx, label in enumerate(selected_docs, start=1):
                info = options_map[label]
                token = info["token"]
                name = info["name"]
                try:
                    binary = feishu_client.download_file(token)
                    target = Path(CONFIG.upload_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    file_path = target / f"feishu_{name}.txt"
                    file_path.write_bytes(binary)
                    chunk_count, _ = process_single_file(file_path, vector_store, doc_processor)
                    logs.append(f"✅ {name} 导入成功，切片 {chunk_count}")
                except (FeishuClientError, DocumentProcessorError, VectorStoreError, OSError) as exc:
                    logs.append(f"❌ {name} 导入失败：{exc}")
                finally:
                    progress.progress(int(idx / total * 100))
            st.text("\n".join(logs))
            st.success("飞书文档导入流程完成")

st.divider()

# =========================
# 5) 飞书多维表格绑定（含管理能力）
# =========================
st.subheader("5) 飞书多维表格绑定")

bindings = load_bitable_bindings()

col_b1, col_b2, col_b3 = st.columns([2, 2, 2])
with col_b1:
    bitable_app_token = st.text_input("多维表格 App Token", key="bitable_app_token")
with col_b2:
    bind_table_id = st.text_input("Table ID（可选，手动绑定用）", key="bind_table_id_main")
with col_b3:
    bind_table_name = st.text_input("表格名称（可选）", key="bind_table_name_main")

max_records = st.number_input("最多导入记录数", min_value=1, max_value=5000, value=1000, step=50)

c51, c52, c53 = st.columns([1, 1, 1])
with c51:
    if st.button("拉取多维表格列表"):
        try:
            if not bitable_app_token.strip():
                st.warning("请先输入 App Token")
            else:
                resp = feishu_client.list_bitable_tables(app_token=bitable_app_token.strip(), page_size=200)
                tables = resp.get("data", {}).get("items", [])
                st.session_state.bitable_tables = tables if isinstance(tables, list) else []
                st.success(f"已拉取 {len(st.session_state.bitable_tables)} 个表")
        except FeishuClientError as exc:
            st.error(f"拉取多维表格失败：{exc}")

with c52:
    if st.button("查看后台同步状态"):
        try:
            import requests

            status_url = os.getenv("ADMIN_API_BASE_URL", "http://127.0.0.1:8000") + "/admin/sync/status"
            token = os.getenv("ADMIN_API_TOKEN", "")
            headers = {"X-Admin-Token": token} if token else {}
            resp = requests.get(status_url, headers=headers, timeout=15)
            if resp.ok:
                data = resp.json()
                st.json({
                    "bitable_sync": data.get("bitable_sync", {}),
                    "knowledge_sync": data.get("knowledge_sync", {}),
                })
            else:
                st.warning(f"读取状态失败：{resp.status_code} {resp.text}")
        except Exception as exc:
            st.warning(f"读取状态异常：{exc}")

with c53:
    if st.button("按 Table ID 绑定并同步"):
        app_token = bitable_app_token.strip()
        table_id = bind_table_id.strip()
        table_name = bind_table_name.strip() or table_id
        if not app_token or not table_id:
            st.warning("请填写 App Token 和 Table ID")
        else:
            existed = next((b for b in bindings if b.get("app_token") == app_token and b.get("table_id") == table_id), None)
            if existed:
                existed["table_name"] = table_name
                existed["max_records"] = int(max_records)
                existed["auto_sync"] = True
            else:
                bindings.append({
                    "app_token": app_token,
                    "table_id": table_id,
                    "table_name": table_name,
                    "max_records": int(max_records),
                    "auto_sync": True,
                    "last_sync_time": "",
                    "last_status": "pending",
                    "last_records": 0,
                    "last_message": "",
                })

            ok, count, msg = sync_bitable_binding(
                {"app_token": app_token, "table_id": table_id, "table_name": table_name},
                feishu_client=feishu_client,
                doc_processor=doc_processor,
                vector_store=vector_store,
            )
            target = next((b for b in bindings if b.get("app_token") == app_token and b.get("table_id") == table_id), None)
            if target is not None:
                target["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                target["last_status"] = "success" if ok else "failed"
                target["last_records"] = count
                target["last_message"] = msg
            save_bitable_bindings(bindings)
            st.success(f"完成：{msg}") if ok else st.error(f"失败：{msg}")

bitable_tables = st.session_state.get("bitable_tables", [])
if bitable_tables:
    table_options: dict[str, dict[str, str]] = {}
    for t in bitable_tables:
        table_id = str(t.get("table_id") or "")
        table_name = str(t.get("name") or table_id)
        if table_id:
            table_options[f"{table_name} ({table_id})"] = {"table_id": table_id, "table_name": table_name}

    selected_tables = st.multiselect("选择要绑定并导入的多维表", options=list(table_options.keys()))

    if st.button("批量绑定并导入多维表"):
        if not selected_tables:
            st.warning("请至少选择一个多维表")
        elif not bitable_app_token.strip():
            st.warning("请先输入 App Token")
        else:
            progress = st.progress(0)
            logs: list[str] = []
            total = len(selected_tables)
            for idx, label in enumerate(selected_tables, start=1):
                info = table_options[label]
                table_id = info["table_id"]
                table_name = info["table_name"]
                existed = next((b for b in bindings if b.get("app_token") == bitable_app_token.strip() and b.get("table_id") == table_id), None)
                if existed:
                    existed["table_name"] = table_name
                    existed["max_records"] = int(max_records)
                    existed["auto_sync"] = True
                else:
                    bindings.append({
                        "app_token": bitable_app_token.strip(),
                        "table_id": table_id,
                        "table_name": table_name,
                        "max_records": int(max_records),
                        "auto_sync": True,
                        "last_sync_time": "",
                        "last_status": "pending",
                        "last_records": 0,
                        "last_message": "",
                    })

                ok, count, msg = sync_bitable_binding(
                    {"app_token": bitable_app_token.strip(), "table_id": table_id, "table_name": table_name},
                    feishu_client=feishu_client,
                    doc_processor=doc_processor,
                    vector_store=vector_store,
                )
                target = next((b for b in bindings if b.get("app_token") == bitable_app_token.strip() and b.get("table_id") == table_id), None)
                if target is not None:
                    target["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    target["last_status"] = "success" if ok else "failed"
                    target["last_records"] = count
                    target["last_message"] = msg

                logs.append(("✅" if ok else "❌") + f" {table_name}: {msg}")
                progress.progress(int(idx / total * 100))

            save_bitable_bindings(bindings)
            st.text("\n".join(logs))
            st.success("多维表格批量绑定处理完成")

st.markdown("#### 已绑定多维表格列表")
if not bindings:
    st.info("暂无已绑定多维表格")
else:
    for i, b in enumerate(bindings):
        with st.container(border=True):
            st.markdown(f"**{b.get('table_name', '-') }**")
            st.caption(
                f"app_token: {b.get('app_token','-')} | table_id: {b.get('table_id','-')} | "
                f"最后同步: {b.get('last_sync_time','-')} | 状态: {b.get('last_status','-')} | 记录数: {b.get('last_records',0)}"
            )

            auto_key = f"auto_sync_main_{i}"
            current_auto = bool(b.get("auto_sync", True))
            new_auto = st.toggle("自动同步（10分钟）", value=current_auto, key=auto_key)
            if new_auto != current_auto:
                b["auto_sync"] = new_auto
                save_bitable_bindings(bindings)

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("手动同步", key=f"sync_main_{i}"):
                    ok, count, msg = sync_bitable_binding(b, feishu_client, doc_processor, vector_store)
                    b["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    b["last_status"] = "success" if ok else "failed"
                    b["last_records"] = count
                    b["last_message"] = msg
                    save_bitable_bindings(bindings)
                    st.success(f"同步成功，记录数：{count}") if ok else st.error(f"同步失败：{msg}")

            with c2:
                if st.button("解绑", key=f"unbind_main_{i}"):
                    try:
                        vector_store.delete_bitable_table(
                            app_token=str(b.get("app_token") or ""),
                            table_id=str(b.get("table_id") or ""),
                            collection_name=st.session_state.current_kb,
                        )
                    except Exception:
                        pass
                    bindings = [x for x in bindings if not (x.get("app_token") == b.get("app_token") and x.get("table_id") == b.get("table_id"))]
                    save_bitable_bindings(bindings)
                    st.success("解绑完成")
                    st.rerun()

            with c3:
                if st.button("查看数据", key=f"view_main_{i}"):
                    try:
                        records = feishu_client.get_bitable_records(
                            app_token=str(b.get("app_token") or ""),
                            table_id=str(b.get("table_id") or ""),
                        )
                        st.json(records[:10])
                    except FeishuClientError as exc:
                        st.error(f"查看失败：{exc}")

            if b.get("last_message"):
                st.caption(f"同步信息：{b.get('last_message')}")

st.divider()

# =========================
# 6) 问答调试（主入口常用）
# =========================
st.subheader("6) 知识库问答调试")
question = st.text_input("请输入问题")

if st.button("开始问答"):
    try:
        result = rag_chain.ask(
            question=question,
            top_k=5,
            collection_name=st.session_state.current_kb,
        )

        # 1) 答案区（贴近你给的样式）
        st.markdown("## 💡 答案")
        answer_text = str(result.get("answer", "")).strip() or "未生成答案"
        st.markdown(answer_text)

        if result.get("error"):
            st.warning(f"调试信息：{result.get('error')}")

        # 2) 引用来源区
        st.markdown("## 📖 引用来源")
        contexts = result.get("contexts", []) if isinstance(result.get("contexts"), list) else []

        if not contexts:
            sources = result.get("sources", []) if isinstance(result.get("sources"), list) else []
            if not sources:
                st.info("暂无引用来源")
            else:
                for s in sources[:5]:
                    st.markdown(
                        f"<div style='background:#f8efd9;border-radius:10px;padding:10px 14px;margin:8px 0;'>"
                        f"📄 {s}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            for ctx in contexts[:5]:
                metadata = ctx.get("metadata", {}) if isinstance(ctx, dict) else {}
                filename = str(metadata.get("filename", "unknown"))
                raw_score = ctx.get("score") if isinstance(ctx, dict) else None

                if isinstance(raw_score, (int, float)):
                    score_text = f"{float(raw_score):.3f}"
                else:
                    score_text = "-"

                st.markdown(
                    f"<div style='background:#f8efd9;border-radius:10px;padding:10px 14px;margin:8px 0;'>"
                    f"📄 {filename} (相似度：{score_text})"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # 3) 可折叠调试片段（保留）
        with st.expander("查看检索片段详情（调试）"):
            if not contexts:
                st.write("未返回检索片段")
            else:
                for idx, ctx in enumerate(contexts[:5], start=1):
                    metadata = ctx.get("metadata", {}) if isinstance(ctx, dict) else {}
                    filename = metadata.get("filename", "unknown")
                    chunk_index = metadata.get("chunk_index", "-")
                    score = ctx.get("score") if isinstance(ctx, dict) else None
                    st.markdown(f"**片段 {idx} | 文件: {filename} | chunk: {chunk_index} | score: {score}**")
                    st.caption(str(metadata))
                    st.write(ctx.get("text", "") if isinstance(ctx, dict) else "")
                    st.divider()
    except Exception as exc:
        st.error(f"问答失败：{exc}")
