"""
配置模块
- 统一管理飞书 API、向量库、嵌入模型、LLM 等配置
- 支持环境变量读取
- 支持两种大模型模式可切换：
  1) 在线 API 模式（豆包 / 通义千问，均可走 OpenAI 兼容接口）
  2) 本地 Ollama 模式

使用说明（核心开关）：
- LLM_MODE=api      -> 使用在线 API 模式
- LLM_MODE=ollama   -> 使用本地 Ollama 模式
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeishuConfig:
    """飞书相关配置。"""

    app_id: str
    app_secret: str
    verification_token: str
    encrypt_key: str
    base_url: str = "https://open.feishu.cn/open-apis"


@dataclass
class EmbeddingConfig:
    """嵌入模型配置（BGE-M3，本地 Sentence-Transformers 加载）。

    说明：
    - model_name_or_path 支持两类值：
      1) 本地路径（推荐，内网可用，启动更稳定）
      2) HuggingFace / ModelScope 模型名
    - 这里默认给出你本机的本地路径示例。
    """

    model_name_or_path: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    """向量库配置（本地 Chroma）。"""

    persist_directory: str = "./data/chroma_db"
    collection_name: str = "feishu_knowledge_base"


@dataclass
class APIModelConfig:
    """在线 API 模式配置（豆包 / 通义千问）。

    统一按 OpenAI 兼容参数组织，便于和 LangChain ChatOpenAI 直接对接。

    provider 可选：
    - doubao
    - qwen
    - other_openai_compatible
    """

    provider: str = "doubao"
    api_key: str = ""
    model_name: str = ""
    base_url: str = ""


@dataclass
class OllamaConfig:
    """本地 Ollama 模式配置。"""

    base_url: str = "http://127.0.0.1:11434/v1"
    model_name: str = "qwen2.5:7b"


@dataclass
class LLMConfig:
    """大模型总配置。"""

    mode: str = "api"  # api | ollama
    temperature: float = 0.2
    max_tokens: int = 1024
    api: APIModelConfig = field(default_factory=APIModelConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)

    @property
    def active_provider(self) -> str:
        """返回当前生效模式标识。"""

        return self.mode

    @property
    def active_model_name(self) -> str:
        """返回当前模式下应使用的模型名。"""

        return self.api.model_name if self.mode == "api" else self.ollama.model_name

    @property
    def active_base_url(self) -> str:
        """返回当前模式下应使用的 base_url。"""

        return self.api.base_url if self.mode == "api" else self.ollama.base_url

    @property
    def active_api_key(self) -> str:
        """返回当前模式下应使用的 API Key。

        - api 模式：返回在线 API key
        - ollama 模式：返回空字符串（通常不需要 key）
        """

        return self.api.api_key if self.mode == "api" else ""


@dataclass
class RAGCacheConfig:
    """RAG 问答缓存配置（Redis）。"""

    enabled: bool = True
    redis_url: str = ""
    ttl_seconds: int = 600
    key_prefix: str = "feishu_rag:qa:"


@dataclass
class AppConfig:
    """应用总配置。"""

    feishu: FeishuConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    llm: LLMConfig
    rag_cache: RAGCacheConfig
    upload_dir: str = "./data/uploads"


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """轻量加载 .env 文件到进程环境变量。

    说明：
    - 不依赖 python-dotenv，减少额外依赖。
    - 已存在的系统环境变量优先级更高，不会被 .env 覆盖。
    """

    env_file = Path(dotenv_path)
    if not env_file.exists() or not env_file.is_file():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def _get_env(key: str, default: str = "") -> str:
    """安全读取环境变量并去除首尾空白。"""

    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else default


def _get_float_env(key: str, default: float) -> float:
    """读取 float 环境变量，异常时回退默认值。"""

    raw = _get_env(key, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _get_int_env(key: str, default: int) -> int:
    """读取 int 环境变量，异常时回退默认值。"""

    raw = _get_env(key, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def load_config() -> AppConfig:
    """加载应用配置。"""

    # 优先加载项目根目录 .env（不覆盖系统环境变量）
    _load_dotenv(".env")

    # -----------------------------
    # 1) 飞书配置
    # -----------------------------
    feishu = FeishuConfig(
        app_id=_get_env("FEISHU_APP_ID"),
        app_secret=_get_env("FEISHU_APP_SECRET"),
        verification_token=_get_env("FEISHU_VERIFICATION_TOKEN"),
        encrypt_key=_get_env("FEISHU_ENCRYPT_KEY"),
        base_url=_get_env("FEISHU_BASE_URL", "https://open.feishu.cn/open-apis"),
    )

    # -----------------------------
    # 2) 嵌入模型配置（BGE-M3）
    # -----------------------------
    embedding_device_raw = _get_env("EMBEDDING_DEVICE", "cpu").lower()
    # 设备别名归一化：支持 gpu/cpu/cuda，默认走 CPU（更适合 Streamlit Cloud）
    if embedding_device_raw in {"gpu", "cuda", "cuda:0"}:
        embedding_device = "cuda"
    elif embedding_device_raw in {"cpu"}:
        embedding_device = "cpu"
    else:
        embedding_device = "cpu"

    embedding = EmbeddingConfig(
        model_name_or_path=_get_env(
            "EMBEDDING_MODEL",
            "BAAI/bge-m3",
        ),
        device=embedding_device,
        normalize_embeddings=_get_env("EMBEDDING_NORMALIZE", "true").lower() in {"1", "true", "yes", "y"},
    )

    # -----------------------------
    # 3) 向量库配置
    # -----------------------------
    vector_store = VectorStoreConfig(
        persist_directory=_get_env("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        collection_name=_get_env("CHROMA_COLLECTION_NAME", "feishu_knowledge_base"),
    )

    # -----------------------------
    # 4) 大模型配置（双模式）
    # -----------------------------
    llm_mode = _get_env("LLM_MODE", "ollama").lower()
    if llm_mode not in {"api", "ollama"}:
        llm_mode = "ollama"

    api_config = APIModelConfig(
        provider=_get_env("LLM_API_PROVIDER", "doubao"),
        # 兼容两种命名：LLM_API_KEY / OPENAI_API_KEY
        api_key=_get_env("LLM_API_KEY") or _get_env("OPENAI_API_KEY"),
        model_name=_get_env("LLM_API_MODEL", "doubao-pro-32k"),
        # 兼容豆包/通义千问 OpenAI 兼容入口，可自行替换
        base_url=_get_env("LLM_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
    )

    ollama_config = OllamaConfig(
        base_url=_get_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        model_name=_get_env("OLLAMA_MODEL", "qwen2.5:7b"),
    )

    llm = LLMConfig(
        mode=llm_mode,
        temperature=_get_float_env("LLM_TEMPERATURE", 0.2),
        max_tokens=_get_int_env("LLM_MAX_TOKENS", 1024),
        api=api_config,
        ollama=ollama_config,
    )

    rag_cache = RAGCacheConfig(
        enabled=_get_env("RAG_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "y"},
        redis_url=_get_env("RAG_CACHE_REDIS_URL") or _get_env("REDIS_URL"),
        ttl_seconds=_get_int_env("RAG_CACHE_TTL_SECONDS", 600),
        key_prefix=_get_env("RAG_CACHE_KEY_PREFIX", "feishu_rag:qa:"),
    )

    upload_dir = _get_env("UPLOAD_DIR", "./data/uploads")

    return AppConfig(
        feishu=feishu,
        embedding=embedding,
        vector_store=vector_store,
        llm=llm,
        rag_cache=rag_cache,
        upload_dir=upload_dir,
    )


# 全局配置对象：项目内统一 import 使用
CONFIG = load_config()
