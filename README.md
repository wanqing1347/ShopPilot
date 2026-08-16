# ShopPilot 跨境电商搜索 Agent

ShopPilot 是一个面向跨境购物场景的智能搜索与比价 Agent。用户只需要用自然语言描述商品、预算、平台和偏好，Agent 就会完成需求理解、商品检索、条件筛选、价格与物流比较，并生成可读的购物建议。

项目用于 AI Agent、商品检索和推荐流程的学习、演示与离线评测。

## 项目背景

传统电商搜索通常要求用户自己拆分关键词、筛选商品、比较价格和估算物流。ShopPilot 尝试把这些步骤串成一条自然语言交互链路，让用户直接表达“想买什么”，由 Agent 协调不同工具完成搜索和整理。

项目重点关注：

- 自然语言购物需求的结构化理解；
- 多平台商品检索与统一展示；
- 预算、材质、品类等条件筛选；
- 商品、价格和物流信息的综合比较；
- 可追踪的 Agent 执行过程和任务恢复。

## 主要功能

- 支持中文自然语言购物请求，识别平台、品类、预算和偏好。
- 支持 Amazon、Walmart、eBay 三个平台的本地离线商品搜索，并提供可选的 `public_demo` 模拟目录。
- 支持商品召回、硬条件过滤、排序、价格比较、物流估算和购物清单生成。
- 支持多轮 Agent 工具调用、任务状态保存、任务恢复和长期偏好记忆。
- 支持后端 API、前端页面、流式事件和任务轨迹查看。
- 提供基于 SIGIR 搜索行为数据的离线检索与排序评测数据。

## 技术栈

- Python 3.12
- LangChain、LangGraph
- FastAPI
- React、Vite
- SQLite（本地任务状态）
- BM25、Faiss、SentenceTransformers（商品检索与排序）

## 数据说明

项目默认使用 `data/offline_catalog` 中的商品快照，包含约 6,616 条已归一化商品观察，并映射到 Amazon、Walmart、eBay 三个平台分区。

这些数据用于演示、测试和离线评测，不代表平台当前库存、实时价格、官方授权商品或最终结算金额。SIGIR 行为数据也仅用于离线检索与排序实验，不代表三个电商平台的真实用户行为。

## 快速开始

### 1. 安装依赖

建议使用 Python 3.12：

```powershell
uv sync --extra dev --extra web --extra retrieval --extra embedding
```

如果没有 `uv`，也可以使用虚拟环境安装：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,retrieval,embedding]"
```

### 2. 配置模型

复制环境变量模板：

```powershell
copy .env.example .env
```

至少配置一个支持 Tool Calling 和结构化输出的 OpenAI-compatible 模型：

```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的密钥
LLM_MODEL_NAME=qwen-max
SHOPPILOT_DATASET_DIR=./data/offline_catalog
```

也可以将 `LLM_BASE_URL` 指向本地 vLLM 或其他兼容服务。

### 3. 运行命令行示例

```powershell
.venv312\Scripts\python.exe scripts/demo.py
.venv312\Scripts\python.exe scripts/demo.py "只在亚马逊找陶瓷咖啡杯，预算350，不要塑料"
```

任务产物默认保存在 `output/<thread_id>/`。

### 4. 启动后端 API

```powershell
uv run uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

API 启动后可用于创建任务、查询任务状态、恢复任务、查看事件和下载任务产物。

### 5. 启动前端

另开终端执行：

```powershell
cd frontend
npm install
npm run dev
```

### 6. 运行测试

```powershell
.venv312\Scripts\python.exe -m pytest -q
```

## 项目验证

项目提供一键验证脚本，可检查 Python 环境、依赖、离线数据集、检索模块和前端构建：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify.ps1
```

## 使用边界

- 默认商品数据是离线缓存快照，不提供实时库存或实时官方价格。
- 搜索结果适合项目演示、开发测试和离线评测，不应直接作为真实下单依据。
- 模型调用需要自行配置 API Key；API Key 不应提交到 GitHub。
