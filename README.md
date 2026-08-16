# ShopPilot 跨境电商搜索 Agent

本项目只保留 **LangChain Agent + LangGraph Runtime** 路径。模型通过多轮 Tool Calling 执行 Think → Act → Observe → Reflect，并可通过 `dispatch_tool` 动态 fork 同质子 AgentLoop。

> 文档示例中的 `langgraph.prebuilt.create_react_agent` 已被 LangGraph v1 弃用。本实现使用当前标准 `langchain.agents.create_agent`，底层运行时仍然是 LangGraph。

## 架构

```text
用户 Query
  ↓
LangChain create_agent（LangGraph runtime + checkpointer）
  ↓
主 AgentLoop：Think → Act → Observe → Reflect
  ├── Planner（LLM 结构化输出）
  ├── ChatFallback
  ├── WebSearch（可选 Tavily）
  ├── CategoryInsight
  ├── ItemSearch
  ├── PriceCompare
  ├── ShippingCalc
  ├── ItemPicker
  ├── ShoppingSummary（工具结果确定性渲染 + LLM 稳定偏好提取）
  └── dispatch_tool
        ↓
      同质子 AgentLoop
      - 同一模型
      - 同一提示词架构
      - 同一 FULL_TOOL_SET
      - 独立 thread_id / checkpoint / ShopPilotState
```

项目不存在固定工具执行顺序、离线 Agent 流水线或规则 Planner。是否调用工具、调用顺序、是否 fork 和何时终止，都由 AgentLoop 基于当前 Observation 决定。

商品检索、汇率换算、运费关税和精排仍是明确的业务工具实现；它们负责可测试的领域计算，不负责替代 Agent 决策。

## 已实现

- LangChain v1 `create_agent` + LangGraph checkpointer。
- 本地默认使用异步 SQLite checkpoint，完整状态可跨进程重启恢复；生产可切换官方 `AsyncPostgresSaver`。
- checkpoint 支持保留周期、启动清理、周期清理、批量上限、扫描上限和活动任务保护。
- OpenAI-compatible 模型接入，可使用 DashScope、OpenAI 或 vLLM 兼容服务。
- LLM 结构化 Planner；ShoppingSummary 使用工具结果确定性渲染，LLM 只提取稳定长期偏好。
- 主 Agent 与同质子 Agent 多轮工具调用。
- 主、子 Agent 的计划、召回、比价、物流、精排和总结均进入 `ShopPilotState`，由 checkpoint 持久化。
- fork 深度、全局/单任务并发上限、总预算、排队超时、队列容量和等价子任务去重。
- 主/子超时、模型调用数、工具调用数和重复调用护栏。
- Token 级模型流式输出；内部摘要模型和工具内部结构化 LLM 输出不会泄露到用户界面。
- 统一工具执行 Harness：错误分类、瞬时故障重试、分级超时、熔断半开恢复、checkpoint/in-flight 幂等复用。
- Cache Breakpoint 上下文治理：稳定摘要 epoch、最近工具轮保留、工具结果结构化压缩和缓存 token 指标。
- 版本化结构化长期记忆：偏好/排除项/历史、来源会话、置信度、冲突覆盖、相关性召回和旧 JSON 自动迁移。
- Schema v2 三平台离线回退快照直读：6,616 条已归一化缓存观察，统一进入 Amazon、Walmart、eBay 三个平台分区，`Candidate.model_validate_json()` 无运行时映射。
- 咖啡杯、旅行收纳、双肩包、键盘、耳机和保温杯均具备跨平台同款组、Query、行为与品类知识。
- Hybrid Retrieval v2：中文 BM25、BGE 中文语义 Embedding、Faiss HNSW、dev-only RRF 调参、预算/材质硬过滤和 Top-N 重排。
- Pairwise Learning-to-Rank：使用 train/dev Query 与行为先验训练 19 维线性 reranker，模型产物校验 provider/model，不兼容时自动降级规则重排。
- 离线检索评测：按 `same_group_id` 去重，分别报告 lexical、vector、Hybrid Rules、Hybrid LTR 的 Recall@K、HitRate@K、MRR、NDCG 和硬约束满足率。
- CategoryInsight Grounded Hybrid RAG：30 篇知识文档直读、品类分区 BM25+BGE+Faiss+RRF、真实模型结构化生成、引用 ID/数字/绝对化措辞校验、无证据拒答和商品统计洞察。
- 最终购物清单由代码从 `PickedItem` 确定性渲染，并原样附加已校验品类依据，避免自由文本模型编造“官方验证”或实时市场结论。
- AGUI 风格事件、每轮 `assistant_call`、`retrieval_search`、`knowledge_retrieval`、`knowledge_synthesis`、取消事件、WebSocket 推送和 Trace 文件。
- 可选 Tavily 实时搜索与 Langfuse 链路追踪。
- FastAPI 后端和 React + Vite 前端。

## 安装

建议使用项目指定的 Python 3.12：

```text
D:\Software\Python\python.exe
```

验证脚本会用该解释器创建 `.venv312`，不会再使用系统 Python 3.14。

```powershell
# 在 ShopPilot 项目根目录执行
powershell -ExecutionPolicy Bypass -File scripts/verify.ps1
```

可选依赖：

```powershell
uv sync --extra dev --extra web --extra observability --extra postgres --extra retrieval --extra embedding
```

没有 `uv` 时：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,retrieval,embedding]"
```

## 配置模型

```powershell
copy .env.example .env
```

至少填写：

```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的密钥
LLM_MODEL_NAME=qwen-max
```

模型必须支持 Tool Calling 和结构化输出。本地 vLLM/OpenAI-compatible 服务可以将 `LLM_BASE_URL` 改为本地地址；localhost 未配置密钥时程序使用 `EMPTY` 占位。

历史详情页支持手动运行 **LLM-as-a-Judge**。Judge 不参与 Agent 执行，只评价规划质量、工具选择、轨迹效率和最终答案质量；确定性错误仍由 Rule-based Trajectory Evaluator 判断。默认复用上面的 LLM 配置，如需使用更强或不同的评审模型，可单独配置：

```env
JUDGE_ENABLED=true
JUDGE_BASE_URL=
JUDGE_API_KEY=
JUDGE_MODEL_NAME=
JUDGE_TEMPERATURE=0
JUDGE_TIMEOUT_SEC=75
```

`JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL_NAME` 留空时分别回退到 `LLM_*` 配置。Judge 只在历史详情页点击“运行 LLM Judge”时调用，结果缓存到对应任务的 `evaluation.json`；普通刷新和再次打开历史记录不会重复消耗模型调用。

主商品数据集（面试演示默认使用）：

```env
SHOPPILOT_DATASET_DIR=./data/offline_catalog
SHOPPILOT_DATASET_SCHEMA_VERSION=2
```

正式运行直接读取项目内 `data/offline_catalog/products.jsonl`。当前快照包含 6,616 条缓存观察，统一映射到 Amazon、Walmart、eBay 三个在线平台对应的离线回退分区。离线结果不代表当前库存、官方实时价格或可结算商品。当前可用数据源设计见 [`docs/REAL_CATALOG_PROVIDER_PLAN.md`](docs/REAL_CATALOG_PROVIDER_PLAN.md)。可直接在项目根目录验证数据集：

```powershell
.venv312\Scripts\python.exe -c "from app.recall.catalog import load_catalog; print(len(load_catalog()))"
```

离线快照中的历史 Amazon、Walmart、eBay 及其他开放市场来源已统一映射到三个平台的离线分区；结果会标记为缓存快照，不代表实时库存、官方 API 或结算价格。

用户行为模拟：项目使用外部 SIGIR 电商搜索挑战数据的有界样本生成 `queries.jsonl`、`interactions.jsonl` 和 `users.jsonl`，当前快照包含 3,533 条查询、85,692 条行为事件和 15,425 个匿名会话用户。由于 SIGIR 的查询和 SKU 是向量/哈希，转换器通过确定性哈希映射到当前商品 `same_group_id`，并将事件标记为 `sigir_simulation`、平台标记为 `public_demo`；这些数据只用于离线 LTR 和检索评测，不代表 Amazon、Walmart 或 eBay 的真实用户行为。原始数据使用限制和转换命令见 [`scripts/import_sigir_behavior.py`](scripts/import_sigir_behavior.py) 及 [`data/offline_catalog/sigir_behavior_summary.json`](data/offline_catalog/sigir_behavior_summary.json)。

可选模拟电商目录（无需账号或 API Key，补充综合品类）：

```env
SHOPPILOT_PUBLIC_DEMO_ENABLED=false
SHOPPILOT_PUBLIC_DEMO_DATA_FILE=./data/public_demo/products.jsonl
```

项目内置 1,000 条 `public_demo` 商品，用作模拟电商商品目录和检索压力测试快照；它整合公开测试 API、Mock Store、只读 Demo Shop、历史开放价格观察等非交易数据，再统一归一化为 `Candidate`。当前快照包含 30 个统一品类键以及 200 多个来源子品类，覆盖手机、电脑、电子产品、服饰鞋包、工具、家居、美妆、食品、收藏玩具和图书等。构建器将任一统一品类限制在约 20% 以内；Open Prices 条目属于历史开放价格观察，不代表当前库存。所有结果都会明确标注为模拟/演示数据，不会冒充真实交易平台的实时商品。

重新构建约 1,000 条快照：

```powershell
.venv312\Scripts\python.exe scripts/build_public_demo_catalog.py --target 1000
```

仅刷新 web-scraping.dev 小型公开测试子集（可选）：

```powershell
.venv312\Scripts\python.exe scripts/scrape_public_demo.py --limit 25
```

面试演示可直接使用三个平台，例如：`在 amazon、walmart、ebay 分别搜索预算 300 元以内的咖啡杯。` 结果来自本地离线快照，不代表实时库存或官方结算价格；如需补充综合品类，可使用 `public_demo` 模拟电商目录，详细说明见 [`docs/PUBLIC_DEMO_SCRAPER.md`](docs/PUBLIC_DEMO_SCRAPER.md)。

Hybrid Retrieval 配置：

```env
SHOPPILOT_RETRIEVAL_BACKEND=hybrid
SHOPPILOT_RETRIEVAL_EMBEDDING_PROVIDER=sentence_transformers
SHOPPILOT_RETRIEVAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
# 留空时 BGE 中文模型自动使用检索 Query 指令。
SHOPPILOT_RETRIEVAL_EMBEDDING_QUERY_PROMPT=
SHOPPILOT_RETRIEVAL_EMBEDDING_DIMENSION=512
SHOPPILOT_RETRIEVAL_INDEX_DIR=./data/retrieval
SHOPPILOT_RETRIEVAL_WARMUP_TIMEOUT_SEC=240
SHOPPILOT_RETRIEVAL_CANDIDATE_POOL=80
SHOPPILOT_RETRIEVAL_RRF_K=60
SHOPPILOT_RETRIEVAL_BM25_WEIGHT=1.5
SHOPPILOT_RETRIEVAL_VECTOR_WEIGHT=0.25
SHOPPILOT_RETRIEVAL_RERANK_WEIGHT=0.25
SHOPPILOT_RETRIEVAL_RERANKER=auto
SHOPPILOT_RETRIEVAL_RERANKER_MODEL=app/recall/artifacts/ltr-v1.json
SHOPPILOT_RETRIEVAL_RERANK_TOP_N=60
SHOPPILOT_RETRIEVAL_HNSW_M=32
SHOPPILOT_RETRIEVAL_HNSW_EF_CONSTRUCTION=80
SHOPPILOT_RETRIEVAL_HNSW_EF_SEARCH=64
SHOPPILOT_RETRIEVAL_FAISS_THREADS=1

SHOPPILOT_KNOWLEDGE_TOP_K=5
SHOPPILOT_KNOWLEDGE_CANDIDATE_POOL=20
SHOPPILOT_KNOWLEDGE_BM25_WEIGHT=1.0
SHOPPILOT_KNOWLEDGE_VECTOR_WEIGHT=1.0
SHOPPILOT_KNOWLEDGE_RRF_K=20
SHOPPILOT_KNOWLEDGE_MIN_EVIDENCE=2
SHOPPILOT_KNOWLEDGE_SYNTHESIS_ENABLED=true
SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_ATTEMPTS=2
SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_CLAIMS=3
SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_CLAIMS=6
SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_TOKEN_OVERLAP=0.05
```

正式配置使用 `BAAI/bge-small-zh-v1.5` 的 512 维语义向量；测试仍可切换 `hashing` 作为无模型依赖回退。商品按平台和品类建立独立 Faiss HNSW 分区。Embedding 缓存按实际向量文本内容 + provider/model 生成稳定指纹，不再因为 `products.jsonl` 的 mtime 或复制路径变化而无效；BGE 优先从本地 Hugging Face snapshot 加载，只有本地不存在时才访问 Hub。进程首次使用时会在主 Agent 300 秒执行预算开始前预热检索器，并通过 single-flight 锁保证并发多平台 fork 只初始化一份模型/索引。`auto` reranker 只有在包内模型与当前 provider/model 完全匹配时才启用 LTR，否则退回规则重排；未知品类不会回退到整个平台商品。

本地 checkpoint：

```env
SHOPPILOT_CHECKPOINT_BACKEND=sqlite
SHOPPILOT_CHECKPOINT_DB=./data/checkpoints.sqlite3
LANGGRAPH_STRICT_MSGPACK=true
```

SQLite 适合本地开发和单机运行。生产多实例配置：

```env
SHOPPILOT_CHECKPOINT_BACKEND=postgres
SHOPPILOT_CHECKPOINT_POSTGRES_DSN=postgresql://user:password@db:5432/shoppilot
SHOPPILOT_CHECKPOINT_AUTO_SETUP=true
```

首次部署默认调用官方 saver 的 `setup()` 建表；严格权限环境可在迁移完成后设置为 `false`。

保留与清理策略：

```env
SHOPPILOT_CHECKPOINT_RETENTION_DAYS=30
SHOPPILOT_CHECKPOINT_CLEANUP_INTERVAL_SEC=3600
SHOPPILOT_CHECKPOINT_CLEANUP_BATCH_SIZE=100
SHOPPILOT_CHECKPOINT_CLEANUP_SCAN_LIMIT=5000
SHOPPILOT_CHECKPOINT_CLEANUP_ON_START=true
```

`RETENTION_DAYS=0` 表示禁用自动清理。清理只删除 checkpoint，不会自动删除 `output/` 中的审计产物。

长期记忆配置：

```env
SHOPPILOT_MEMORY_FILE=./data/preferences.json
SHOPPILOT_MEMORY_RETRIEVAL_LIMIT=20
SHOPPILOT_MEMORY_MIN_RELEVANCE=0.35
SHOPPILOT_MEMORY_MAX_ENTRIES_PER_USER=200
SHOPPILOT_MEMORY_CONFIDENCE_INCREMENT=0.06
```

旧版 `user_id -> [字符串]` 文件会在首次读取时自动迁移为 version 2 结构。相同偏好在不同会话再次出现会增加 `mention_count` 和置信度；同一会话重复收尾保持幂等。相反偏好使用“新证据生效、旧记录 superseded”的策略，并保留冲突链供审计和手动恢复。

工具执行韧性：

```env
SHOPPILOT_TOOL_MAX_RETRIES=2
SHOPPILOT_TOOL_RETRY_INITIAL_DELAY_SEC=0.5
SHOPPILOT_TOOL_RETRY_MAX_DELAY_SEC=5
SHOPPILOT_TOOL_RETRY_JITTER=true
SHOPPILOT_TOOL_CIRCUIT_FAILURE_THRESHOLD=3
SHOPPILOT_TOOL_CIRCUIT_RESET_SEC=30
SHOPPILOT_TOOL_IDEMPOTENCY_TTL_SEC=3600
SHOPPILOT_TOOL_TIMEOUT_LLM_SEC=75
SHOPPILOT_TOOL_TIMEOUT_SEARCH_SEC=30
SHOPPILOT_TOOL_TIMEOUT_COMPUTE_SEC=15
SHOPPILOT_TOOL_TIMEOUT_WEB_SEC=25
SHOPPILOT_TOOL_TIMEOUT_SUB_AGENT_SEC=95
```

只对 `rate_limit / timeout / network` 瞬时错误自动重试；参数校验、权限、业务约束等 `business` 错误不会重试。连续瞬时失败达到阈值后按工具名打开进程级熔断器，冷却后只允许一个半开探针。成功工具结果会写入 checkpoint 幂等缓存；同一线程中的并发等价调用复用同一个 in-flight Task。`dispatch_tool` 继续使用专门的 fork 去重器，不重复缓存子线程结果。

主要护栏：

```env
SHOPPILOT_MAX_FORK_DEPTH=2
SHOPPILOT_MAX_CONCURRENT_FORKS=8
SHOPPILOT_MAX_CONCURRENT_FORKS_PER_TASK=4
SHOPPILOT_MAX_FORKS_PER_TASK=12
SHOPPILOT_MAX_FORK_QUEUE_SIZE=64
SHOPPILOT_FORK_QUEUE_TIMEOUT_SEC=30
SHOPPILOT_FORK_DEDUP_TTL_SEC=300
SHOPPILOT_MAX_TOOL_CALLS=30
SHOPPILOT_MAX_MODEL_STEPS=30
SHOPPILOT_MAIN_TIMEOUT_SEC=300
SHOPPILOT_SUB_AGENT_TIMEOUT_SEC=90
SHOPPILOT_TOOL_RESULT_MAX_CHARS=16000
SHOPPILOT_CONTEXT_COMPACTION_TRIGGER_MESSAGES=40
SHOPPILOT_CONTEXT_COMPACTION_TRIGGER_CHARS=48000
SHOPPILOT_CONTEXT_KEEP_RECENT_TOOL_CALLS=3
SHOPPILOT_CONTEXT_KEEP_RECENT_MESSAGES=12
SHOPPILOT_CONTEXT_COMPACTION_MIN_MESSAGES=8
SHOPPILOT_CONTEXT_SUMMARY_MAX_CHARS=6000
SHOPPILOT_CONTEXT_TOOL_MESSAGE_MAX_CHARS=4000
```

上下文压缩不会删除 checkpoint 中的原始消息。模型调用时使用：

```text
固定 system prompt + 当前阶段工具 schema
+ 稳定工作记忆摘要（只在跨越阈值时更新一个 cache epoch）
+ 最近 K 个完整工具调用轮次
```

同一 epoch 内摘要和历史尾部前缀保持不变，避免每轮微压缩破坏 Prompt Cache。OpenAI 官方模型可选设置：

```env
SHOPPILOT_PROMPT_CACHE_KEY_ENABLED=true
SHOPPILOT_PROMPT_CACHE_KEY_PREFIX=shoppilot-agent-v1
```

通用 OpenAI-compatible 服务默认关闭该参数，避免供应商拒绝未知字段；即使关闭，稳定且完全匹配的前缀仍可被支持自动 Prompt Cache 的服务复用。

## 运行 CLI

```powershell
.venv312\Scripts\python.exe scripts/demo.py
.venv312\Scripts\python.exe scripts/demo.py "只在亚马逊找陶瓷咖啡杯，预算350，不要塑料"
```

产物位于 `output/<thread_id>/`：

- `shopping-list.md`：最终购物清单。
- `result.json`：结构化中间产物。
- `trace.json`：阶段、工具、fork、重试、熔断和错误事件。Token 增量是瞬时事件，不进入有限的审计回放缓冲。

## 启动 API

```powershell
uv run uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

主要接口：

| 接口 | 作用 |
| --- | --- |
| `POST /api/task` | 启动任务并返回 thread_id |
| `GET /api/task/{thread_id}` | 查询状态和结果 |
| `POST /api/task/{thread_id}/resume` | 从持久化 checkpoint 恢复任务 |
| `GET /api/task/{thread_id}/checkpoint` | 查看最新结构化 checkpoint |
| `DELETE /api/task/{thread_id}/checkpoint` | 删除该线程全部 checkpoint |
| `POST /api/checkpoints/cleanup` | 手动执行一次过期 checkpoint 批量清理 |
| `POST /api/task/{thread_id}/cancel` | 取消任务 |
| `GET /api/users/{user_id}/memories` | 查看全部记忆，或通过 query 做相关性召回 |
| `DELETE /api/users/{user_id}/memories/{memory_id}` | 删除单条长期记忆 |
| `POST /api/users/{user_id}/memories/resolve` | 选择冲突中的权威记忆并覆盖相反项 |
| `GET /api/task/{thread_id}/events` | 获取事件历史 |
| `WS /ws/{thread_id}` | 订阅实时事件 |
| `GET /api/files/{thread_id}/{filename}` | 下载产物 |
| `POST /api/upload?thread_id=...` | 上传参考文件 |

前端：

```powershell
cd frontend
npm install
npm run dev
```

## 测试

单元测试使用 Fake LLM 验证 Planner、Summary、工具注册和 Agent 入口，不调用真实模型：

```powershell
.venv312\Scripts\python.exe -m pytest -q
```

一键验证会确认基础解释器确实为 Python 3.12、校验 Schema v2 数据集、安装 Faiss 与 SentenceTransformers、运行 Hybrid Retrieval/LTR/知识检索/SQLite 恢复测试、生成商品检索与品类知识检索报告，并构建前端：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify.ps1
```

检索调参、Reranker 训练和最终评测严格分开：

```powershell
# 仅使用 dev 调整 BM25/Vector/RRF 与规则重排权重。
.venv312\Scripts\python.exe scripts/tune_retrieval.py

# train 生成 pairwise 样本，dev 选择正则强度，最终用 train+dev 重训。
.venv312\Scripts\python.exe scripts/train_reranker.py

# 模型和参数锁定后才运行 test。
.venv312\Scripts\python.exe scripts/evaluate_retrieval.py --split test --k 5 10 20

# 评测 30 条品类知识主题检索用例。
.venv312\Scripts\python.exe scripts/evaluate_knowledge.py --k 1 3 5
```

主要报告位于 `output/retrieval-tuning-dev.json`、`output/reranker-training-report.json` 和 `output/retrieval-evaluation.json`。当前 54 条 test Query 的结果：

| 通道 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | 硬约束@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 + Rules | 0.637037 | 0.811111 | 0.976852 | 0.806064 | 0.764815 |
| BGE Hybrid + Rules | 0.629630 | 0.807407 | 0.969136 | 0.799942 | 0.788889 |
| BGE Hybrid + LTR | 0.744444 | 0.862963 | 0.990741 | 0.877316 | 0.772222 |

LTR 显著提升 Recall/NDCG，但硬约束@5 有所下降，因此预算、可售和排除材质始终在召回后、学习排序前做强制过滤。以上是合成数据集离线指标，不能等同于真实线上效果。

知识检索报告写入 `output/knowledge-retrieval-evaluation.json`。当前 30 条主题用例中 Hybrid Recall@1 和 MRR@5 均为 1.0；该知识集每个品类只有 5 篇且主题清晰，因此结果只证明索引、品类隔离和引用链正确，不代表开放域 RAG 效果。

真实 Grounded RAG 评测：

```powershell
.venv312\Scripts\python.exe scripts/evaluate_grounded_rag.py
```

报告写入 `output/grounded-rag-evaluation.json`。当前 `qwen-flash` 的 6 个已知品类和 1 个未知品类用例中，grounded 成功率、无证据拒答率、引用覆盖率和数字依据率均为 1.0，虚构引用为 0；校验器额外丢弃了 1 条“占比最高”的过度推断。该结果仍基于合成知识与小型固定用例。

设置 `RUN_LIVE_RAG_TEST=1` 可让验证脚本运行真实 Grounded RAG 评测；设置 `RUN_LIVE_AGENT_TEST=1` 会额外执行完整 AgentLoop 冒烟。两项都会产生模型调用费用。

## 关键目录

```text
app/agent/
├── main_agent.py       # 唯一 AgentLoop 入口
├── graph_runtime.py    # create_agent、恢复执行、动态工具权限和步骤流
├── checkpoint.py       # SQLite/PostgreSQL saver、保留周期与批量清理
├── context_governance.py # Cache Breakpoint、稳定摘要 epoch 与缓存指标
├── tool_reliability.py # 工具重试、分级超时、熔断和幂等复用
├── state.py            # checkpointed ShopPilotState 与并行 reducer
├── llm.py              # OpenAI-compatible 模型适配
├── prompts.py          # 主/子 Agent 系统提示词
├── tool_registry.py    # FULL_TOOL_SET 和工具包装
├── dispatch_tool.py    # 同质子 Agent fork
├── fork_scheduler.py   # 全局/单任务并发、排队、预算与去重
├── workspace.py        # 旧模块迁移标记，不再保存运行状态
├── fork_guard.py       # fork 深度护栏
└── settings.py         # 超时和调用预算

app/memory/
├── models.py           # 结构化 MemoryEntry、状态、置信度和写入报告
└── store.py            # 旧格式迁移、相关性召回、冲突处理和原子写入

app/knowledge/
├── catalog.py          # 30 篇 category_knowledge JSONL 直读与校验
├── models.py           # KnowledgeDocument、Hit 和检索结果
├── retriever.py        # 品类分区 BM25+BGE+Faiss+RRF
├── synthesis.py        # 真实模型 grounded claims、引用/数字/措辞校验和安全降级
├── grounded_evaluation.py # 6+1 真实 RAG 固定评测集
└── evaluation.py       # 知识 Recall/MRR/NDCG 基准

app/recall/
├── catalog.py          # Schema v2 JSONL 直读、版本校验、可售过滤和缓存
├── tokenizer.py        # 中英文混合 token、中文 bigram/trigram
├── bm25.py             # 可复现 BM25Okapi
├── embeddings.py       # Hashing / BGE SentenceTransformers provider
├── vector_index.py     # Faiss HNSW 与精确回退
├── hybrid.py           # 分区索引、RRF、硬过滤和可插拔重排
├── ltr.py              # 19 维 LTR 特征、模型加载、兼容检查和推理
├── ltr_training.py     # pairwise 样本、行为先验、dev 选参与训练
├── artifacts/ltr-v1.json # train+dev 训练的小型线性模型
└── evaluation.py       # Recall/MRR/NDCG/硬约束评测

app/tools/
├── planner.py          # LLM 结构化意图规划
├── category_insight.py
├── item_search.py
├── price_compare.py
├── shipping_calc.py
├── item_picker.py
└── shopping_summary.py # 确定性购物清单渲染、偏好提取和已校验引用附加
```

## 尚未接入的生产组件

当前长期记忆后端仍是本地原子 JSON 文件，生产多实例可继续替换为 PostgreSQL/向量 Store。默认商品数据是多来源历史离线快照，平台字段用于来源分区，不代表当前官方库存或实时结算价；30 篇品类知识仍是可复现测试语料。当前 LTR 是可解释的线性 pairwise 模型，不是大规模 CrossEncoder。CategoryInsight 已完成真实模型 grounded synthesis 和引用校验，但评测集规模仍小；后续重点是扩充人工知识与对抗评测集、三塔个性化召回、Cross-Encoder 对照，以及实时汇率与 HS Code 税则服务。
