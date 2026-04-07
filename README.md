# Feishu RAG Knowledge Base

一个基于飞书(Feishu)平台的企业级RAG（检索增强生成）知识库系统。支持文档上传、向量检索、智能问答，并与飞书机器人深度集成。

## ✨ 功能特性

- 🤖 **飞书机器人集成** - 支持飞书群聊和私聊问答
- 📚 **多格式文档支持** - 支持 PDF、Word、Excel、TXT 等格式
- 🔍 **智能检索** - 基于 BGE-M3 嵌入模型的语义检索
- 🧠 **双模式LLM** - 支持在线API（豆包/通义千问）和本地Ollama
- 💾 **向量数据库** - 使用 ChromaDB 进行高效向量存储
- ⚡ **高性能缓存** - Redis 缓存高频问答
- 🔐 **权限管理** - 支持用户权限映射和知识库访问控制
- 📊 **多维表格同步** - 支持飞书多维表格数据同步
- 🔄 **增量同步** - 知识库定时增量更新

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Redis（可选，用于缓存）
- 飞书开放平台账号

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/KeepMoving888/feishu-rag.git
cd feishu-rag
```

1. **创建虚拟环境**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

1. **安装依赖**

```bash
pip install -r requirements.txt
```

1. **配置环境变量**

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的配置
```

1. **启动服务**

```bash
# 启动后端API
python main.py

# 或启动管理后台（默认端口改为 8511）
streamlit run admin.py
# 若需临时指定端口：
# streamlit run admin.py --server.port 8511
```

## ⚙️ 配置说明

### 飞书开放平台配置

1. 访问 [飞书开放平台](https://open.feishu.cn/)
2. 创建企业自建应用
3. 获取 App ID 和 App Secret
4. 配置事件订阅（Webhook URL）
5. 开启机器人能力

### 环境变量配置

```env
# 飞书配置
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
FEISHU_VERIFICATION_TOKEN=your_token
FEISHU_ENCRYPT_KEY=your_encrypt_key

# LLM模式选择 (api/ollama)
LLM_MODE=api

# 在线API配置（豆包/通义千问）
LLM_API_KEY=your_api_key
LLM_API_MODEL=your_model_name
LLM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 本地Ollama配置
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=qwen2.5:7b

# 嵌入模型配置
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=gpu

# 向量数据库
CHROMA_PERSIST_DIR=./data/chroma_db

# Redis缓存（可选）
REDIS_URL=redis://localhost:6379/0
```

## 📖 使用指南

### API接口

| 接口                | 方法   | 说明       |
| ----------------- | ---- | -------- |
| `/health`         | GET  | 健康检查     |
| `/upload`         | POST | 上传文档到知识库 |
| `/ask`            | POST | RAG问答    |
| `/feishu/webhook` | POST | 飞书事件回调   |

### 上传文档示例

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### 问答示例

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "公司的年假政策是什么？",
    "user_id": "user_123"
  }'
```

## 🏗️ 项目结构

```
feishu-rag/
├── main.py                 # FastAPI 后端入口
├── admin.py                # Streamlit 管理后台
├── config.py               # 配置管理
├── rag_chain.py            # RAG问答链
├── vector_store.py         # 向量存储管理
├── document_processor.py   # 文档处理
├── feishu_client.py        # 飞书API客户端
├── requirements.txt        # 依赖列表
├── .env.example            # 环境变量模板
└── data/                   # 数据目录
    ├── uploads/            # 上传文件
    ├── chroma_db/          # 向量数据库
    └── state/              # 状态文件
```

## 🔧 技术栈

- **后端框架**: FastAPI
- **前端管理**: Streamlit
- **向量数据库**: ChromaDB
- **嵌入模型**: BGE-M3
- **LLM框架**: LangChain
- **缓存**: Redis
- **任务调度**: APScheduler

## 📝 开发计划

- [ ] 支持更多文档格式
- [ ] 多轮对话支持
- [ ] 知识库权限精细化管理
- [ ] 问答历史记录
- [ ] 模型性能监控

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding)
- [FastAPI](https://github.com/tiangolo/fastapi)

