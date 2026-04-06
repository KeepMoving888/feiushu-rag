"""
向量库模块（Chroma）
实现：
1) 向量存储
2) 检索
3) 删除
4) 多知识库（独立 collection）切换
5) 切片管理（查询 / 更新 / 删除 / 合并）
"""

from __future__ import annotations
from typing import Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import CONFIG


class VectorStoreError(Exception):
    """向量库异常。"""


class VectorStoreManager:
    """基于 Chroma 的向量库管理器。"""

    def __init__(self, collection_name: str | None = None) -> None:
        """初始化嵌入模型与 Chroma。"""

        try:
            self.persist_directory = CONFIG.vector_store.persist_directory
            self.collection_name = collection_name or CONFIG.vector_store.collection_name

            requested_device = CONFIG.embedding.device
            model_kwargs = {"device": requested_device}

            # 优先按配置设备初始化，若 GPU 不可用自动降级到 CPU
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=CONFIG.embedding.model_name_or_path,
                    model_kwargs=model_kwargs,
                    encode_kwargs={"normalize_embeddings": CONFIG.embedding.normalize_embeddings},
                )
            except Exception as emb_exc:
                if requested_device == "cuda":
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=CONFIG.embedding.model_name_or_path,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": CONFIG.embedding.normalize_embeddings},
                    )
                else:
                    raise emb_exc

            self.vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        except Exception as exc:
            raise VectorStoreError(f"初始化向量库失败: {exc}") from exc

    def switch_collection(self, collection_name: str) -> None:
        """切换知识库 collection。"""

        if not collection_name or not collection_name.strip():
            raise VectorStoreError("collection_name 不能为空")

        self.collection_name = collection_name.strip()
        try:
            self.vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        except Exception as exc:
            raise VectorStoreError(f"切换知识库失败: {exc}") from exc

    def add_documents(self, chunks: list[dict[str, Any]]) -> list[str]:
        """批量写入分块文档。"""

        if not chunks:
            raise VectorStoreError("chunks 不能为空")

        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        ids = [f"doc_{m.get('filename', 'unknown')}_{m.get('chunk_index', i)}" for i, m in enumerate(metadatas)]

        try:
            self.vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            return ids
        except Exception as exc:
            raise VectorStoreError(f"写入向量库失败: {exc}") from exc

    def similarity_search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """相似度检索。"""

        if not query or not query.strip():
            raise VectorStoreError("query 不能为空")

        try:
            # 使用带分数检索，便于后续置信度判断
            docs_with_score = self.vs.similarity_search_with_score(query=query, k=k)
            result: list[dict[str, Any]] = []
            for item in docs_with_score:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                doc, score = item
                result.append(
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                    }
                )
            return result
        except Exception:
            # 部分向量库实现不支持 score，回退普通检索
            try:
                docs = self.vs.similarity_search(query=query, k=k)
                result: list[dict[str, Any]] = []
                for d in docs:
                    result.append({"text": d.page_content, "metadata": d.metadata, "score": None})
                return result
            except Exception as exc:
                raise VectorStoreError(f"检索失败: {exc}") from exc

    def list_collections(self) -> list[str]:
        """列出当前持久化目录下的所有知识库集合。"""

        try:
            client = self.vs._client  # noqa: SLF001
            collections = client.list_collections()
            return [c.name for c in collections]
        except Exception as exc:
            raise VectorStoreError(f"获取知识库列表失败: {exc}") from exc

    def create_collection(self, name: str) -> None:
        """创建知识库（collection）。"""

        if not name or not name.strip():
            raise VectorStoreError("知识库名称不能为空")
        self.switch_collection(name.strip())

    def delete_collection(self, name: str) -> None:
        """删除知识库（collection）。"""

        if not name or not name.strip():
            raise VectorStoreError("知识库名称不能为空")

        try:
            client = self.vs._client  # noqa: SLF001
            client.delete_collection(name=name.strip())
            # 删除后切回默认库，避免悬空句柄
            self.switch_collection(CONFIG.vector_store.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"删除知识库失败: {exc}") from exc

    def list_chunks(self, filename: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """查看切片详情。"""

        try:
            where = {"filename": filename} if filename else None
            data = self.vs.get(where=where, include=["documents", "metadatas"], limit=limit)
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])

            result: list[dict[str, Any]] = []
            for i, cid in enumerate(ids):
                result.append(
                    {
                        "id": cid,
                        "text": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                    }
                )
            return result
        except Exception as exc:
            raise VectorStoreError(f"查询切片失败: {exc}") from exc

    def update_chunk(self, chunk_id: str, new_text: str, metadata: dict[str, Any] | None = None) -> None:
        """修改指定切片文本（重新计算 embedding 后覆盖）。"""

        if not chunk_id:
            raise VectorStoreError("chunk_id 不能为空")
        if not new_text or not new_text.strip():
            raise VectorStoreError("new_text 不能为空")

        try:
            if metadata is None:
                old = self.vs.get(ids=[chunk_id], include=["metadatas"])
                mds = old.get("metadatas", [])
                metadata = mds[0] if mds else {}

            embedding = self.embeddings.embed_documents([new_text])[0]
            self.vs._collection.upsert(  # noqa: SLF001
                ids=[chunk_id],
                documents=[new_text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
        except Exception as exc:
            raise VectorStoreError(f"修改切片失败: {exc}") from exc

    def merge_chunks(self, chunk_ids: list[str], merged_text: str, merged_metadata: dict[str, Any] | None = None) -> str:
        """合并多个切片为一个新切片，并删除原切片。"""

        if len(chunk_ids) < 2:
            raise VectorStoreError("至少选择两个切片进行合并")
        if not merged_text or not merged_text.strip():
            raise VectorStoreError("merged_text 不能为空")

        new_id = f"merged_{abs(hash('|'.join(chunk_ids)))}"
        try:
            if merged_metadata is None:
                old = self.vs.get(ids=[chunk_ids[0]], include=["metadatas"])
                mds = old.get("metadatas", [])
                merged_metadata = mds[0] if mds else {}

            # 使用 add_documents + 指定 id，确保通过 embedding_function 正确计算向量
            doc = Document(page_content=merged_text, metadata=merged_metadata)
            self.vs.add_documents(documents=[doc], ids=[new_id])

            self.delete_by_ids(chunk_ids)
            return new_id
        except Exception as exc:
            raise VectorStoreError(f"合并切片失败: {exc}") from exc

    def delete_by_ids(self, ids: list[str]) -> None:
        """按向量 ID 删除。"""

        if not ids:
            raise VectorStoreError("ids 不能为空")

        try:
            self.vs.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"删除失败: {exc}") from exc

    def clear_current_collection(self) -> int:
        """一键清空当前知识库的全部切片。"""

        try:
            data = self.vs.get(include=[])
            ids = data.get("ids", []) or []
            if not ids:
                return 0
            self.vs.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"清空知识库失败: {exc}") from exc

    def add_bitable_to_vectorstore(
        self,
        chunks: list[str],
        metadatas: list[dict[str, Any]],
        collection_name: str = "default",
    ) -> list[str]:
        """批量写入飞书多维表格行级数据到向量库。

        参数：
        - chunks: 每行记录对应的文本块列表
        - metadatas: 与 chunks 一一对应的 metadata（需包含 app_token/table_id/record_id）
        - collection_name: 目标知识库；默认 "default"（若项目未创建该库请先创建）

        返回：
        - 写入成功的向量 ID 列表
        """

        if not chunks:
            raise VectorStoreError("chunks 不能为空")
        if not metadatas:
            raise VectorStoreError("metadatas 不能为空")
        if len(chunks) != len(metadatas):
            raise VectorStoreError("chunks 与 metadatas 长度必须一致")

        old_collection = self.collection_name
        try:
            if collection_name and collection_name != self.collection_name:
                self.switch_collection(collection_name)

            ids: list[str] = []
            docs: list[Document] = []
            for idx, text in enumerate(chunks):
                meta = metadatas[idx] if isinstance(metadatas[idx], dict) else {}
                app_token = str(meta.get("app_token") or "")
                table_id = str(meta.get("table_id") or "")
                record_id = str(meta.get("record_id") or idx)
                doc_id = f"bitable_{app_token}_{table_id}_{record_id}"

                merged_meta = {"source_type": "bitable", **meta}
                ids.append(doc_id)
                docs.append(Document(page_content=text, metadata=merged_meta))

            self.vs.add_documents(documents=docs, ids=ids)
            return ids
        except Exception as exc:
            raise VectorStoreError(f"多维表格数据入库失败: {exc}") from exc
        finally:
            if old_collection != self.collection_name:
                self.switch_collection(old_collection)

    def update_bitable_record(
        self,
        app_token: str,
        table_id: str,
        record_id: str,
        new_data: dict[str, Any],
    ) -> str:
        """更新单条多维表格记录向量数据（先删后加，失败回滚）。

        原子性策略：
        1) 先备份旧记录
        2) 删除旧记录
        3) 写入新记录
        4) 若写入失败，自动回滚旧记录，避免新旧并存或数据丢失
        """

        if not app_token:
            raise VectorStoreError("app_token 不能为空")
        if not table_id:
            raise VectorStoreError("table_id 不能为空")
        if not record_id:
            raise VectorStoreError("record_id 不能为空")
        if not isinstance(new_data, dict):
            raise VectorStoreError("new_data 必须是字典")

        text = str(new_data.get("text") or "").strip()
        metadata = new_data.get("metadata", {})
        if not text:
            raise VectorStoreError("new_data.text 不能为空")
        if not isinstance(metadata, dict):
            metadata = {}

        where = {
            "$and": [
                {"app_token": app_token},
                {"table_id": table_id},
                {"record_id": record_id},
            ]
        }

        try:
            old = self.vs.get(where=where, include=["documents", "metadatas"])
            old_ids = old.get("ids", []) or []
            old_docs = old.get("documents", []) or []
            old_metas = old.get("metadatas", []) or []

            # 先删旧
            self.vs.delete(where=where)

            # 再加新（固定使用同一 id，避免并存）
            new_id = f"bitable_{app_token}_{table_id}_{record_id}"
            merged_meta = {
                "source_type": "bitable",
                "app_token": app_token,
                "table_id": table_id,
                "record_id": record_id,
                **metadata,
            }
            self.vs.add_documents(documents=[Document(page_content=text, metadata=merged_meta)], ids=[new_id])
            return new_id
        except Exception as exc:
            # 回滚旧数据
            try:
                if old_ids:
                    rollback_docs: list[Document] = []
                    for i, old_id in enumerate(old_ids):
                        old_text = old_docs[i] if i < len(old_docs) else ""
                        old_meta = old_metas[i] if i < len(old_metas) else {}
                        rollback_docs.append(Document(page_content=old_text, metadata=old_meta))
                    self.vs.add_documents(documents=rollback_docs, ids=old_ids)
            except Exception:
                pass
            raise VectorStoreError(f"更新多维表格记录失败: {exc}") from exc

    def delete_bitable_table(self, app_token: str, table_id: str, collection_name: str = "default") -> int:
        """删除整张多维表格在向量库中的数据。"""

        if not app_token:
            raise VectorStoreError("app_token 不能为空")
        if not table_id:
            raise VectorStoreError("table_id 不能为空")

        old_collection = self.collection_name
        try:
            if collection_name and collection_name != self.collection_name:
                self.switch_collection(collection_name)

            where = {
                "$and": [
                    {"app_token": app_token},
                    {"table_id": table_id},
                ]
            }
            existed = self.vs.get(where=where, include=[])
            ids = existed.get("ids", []) or []
            if not ids:
                return 0

            self.vs.delete(where=where)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"删除多维表格数据失败: {exc}") from exc
        finally:
            if old_collection != self.collection_name:
                self.switch_collection(old_collection)
