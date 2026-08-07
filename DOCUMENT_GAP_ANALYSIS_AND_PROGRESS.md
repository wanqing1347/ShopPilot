# ShopPilot 文档差距分析与改造进度

> 创建时间：2026-08-05 12:34（Asia/Taipei）  
> 最近更新：2026-08-05 21:55（Asia/Taipei）  
> 目标：持续记录当前代码与《电商搜索 Agent》文档的差距、优先级、处理记录和验证结果。

## 1. 当前总体判断

当前项目已经具备 LangChain `create_agent` + LangGraph runtime、主 AgentLoop、同质子 Agent fork、checkpointed 业务状态、阶段化工具权限、持久化 SQLite/PostgreSQL checkpoint、fork 调度、Cache Breakpoint、结构化长期记忆、工具可靠性 Harness、Schema v2 数据集、BGE/Faiss Hybrid Retrieval、Pairwise LTR、真实模型 CategoryInsight Grounded RAG、确定性终结清单、FastAPI 和 WebSocket。

核心 Agent 编排、持久化、上下文治理、用户记忆、执行韧性、数据底座、语义召回、学习排序、品类知识生成和引用校验已经进入可恢复、可审计、可评测状态；仍待重点建设的是人工对抗 RAG 评测、个性化 User Tower、完整观测、外部任务队列和真实业务服务。

## 2. 优先问题清单

### P0：当前代码缺陷

- [x] 修复 `app/api/server.py` 中已删除 `get_agent_mode()` 的残留调用。
- [x] 清理前端 `Offline Reproduction` 旧文案。
- [x] 完成 Python 编译、单测、LangGraph 构建和前端生产构建验证。

### P1：核心 Agent 运行一致性

- [x] 将计划、搜索结果、比价、物流、精排、总结等业务状态迁入 `ShopPilotState`。
- [x] 使用 `state_schema=ShopPilotState`，使业务中间产物进入 LangGraph checkpoint，而不再只保存消息历史。
- [x] 工具使用 `ToolRuntime` 读取 state，并使用 `Command(update=...)` 写入 state。
- [x] 为并行 `search_outputs` 和 `sub_agent_results` 增加 reducer。
- [x] 将 `shopping_summary` 和 `chat_fallback` 标记为 `return_direct=True`，同时写入 `terminated` 和 `terminal_tool`。
- [x] 增加阶段化动态工具权限；终结后不再向模型暴露工具。
- [x] 子 Agent 平台和品类范围进入 state，并在 `item_search` 执行层强制检查，不再只依赖 Prompt。
- [x] 子 Agent 完整结构化结果按 `sub_thread_id` 写入 `sub_agent_results`，同时合并召回结果和品类知识。
- [x] 将 `InMemorySaver` 替换为 `AsyncSqliteSaver` 本地持久化 checkpointer；生产环境仍规划 PostgreSQL。
- [x] 实现进程重启后的恢复入口、checkpoint 查看接口和按 thread 删除接口。
- [x] 增加自动保留周期、批量过期清理和 PostgreSQL 生产后端。

### P2：事件流与 Harness

- [x] 使用 LangGraph `astream(..., stream_mode="updates", version="v2")` 获取每轮模型节点更新。
- [x] 增加 `assistant_call` 事件，展示每轮模型选择的工具。
- [x] 增加 `task_cancelled` 事件，并在后台任务取消时推送。
- [x] 工具结果超长时生成合法的结构化 JSON 预览，不再从字符串中间截断导致非法 JSON。
- [x] 增加阶段化工具过滤，减少不合时序的工具调用。
- [x] 增加进程级和单主任务 Semaphore、唯一 fork 总预算、等价平台子任务去重、FIFO 排队、排队超时和队列容量上限。
- [x] 使用 `stream_mode=["messages", "updates"]` 增加主模型 Token 流；内部上下文摘要和工具内部 LLM 输出不会泄露到用户界面。
- [x] 增加工具错误分类、瞬时故障重试、分级超时、进程级熔断、checkpoint/in-flight 幂等复用和对应事件。

### P3：文档中的基础设施差距

- [x] Cache Breakpoint：稳定摘要 epoch、断点定位、缓存 token 统计和 cache-aware 压缩。
- [x] 结构化长期记忆：类别、来源会话、置信度、时间戳、冲突处理和相关性检索。
- [x] 六品类 Schema v2 合成数据集：1,200 条四平台商品、同款组、Query、行为、知识文档和项目 Candidate 直读。
- [x] BM25、BGE/Faiss HNSW、RRF、标量过滤、学习型 Reranker 和召回评测。
- [x] train/dev Pairwise LTR、行为先验、模型兼容校验和 test-only 最终评测。
- [x] CategoryInsight 本地 Hybrid RAG 检索层：知识直读、BM25+BGE+Faiss、引用和无证据降级。
- [x] 使用真实 LLM 的 grounded synthesis、引用 ID/数字/绝对化措辞校验和 6+1 固定评测。
- [ ] 扩充人工审核与对抗 RAG 评测集，并评估生产 OpenSearch RAG。
- [ ] 三塔个性化召回与 Cross-Encoder 对照实验。
- [ ] 真实平台商品、实时汇率、HS Code 和物流服务。
- [ ] Redis/外部任务队列、多实例任务状态共享和 PostgreSQL 容器级集成验证。
- [ ] Langfuse 完整 Trace/Span/Score、Rubric、SFT/RL、自进化和漂移检测。

## 3. 本轮完成的代码改动

### LangGraph State

新增：

- `app/agent/state.py`
  - `ShopPilotState`
  - `initial_state()`
  - `state_payload()`
  - 并行 mapping reducer

迁移状态字段：

- `plan`
- `insight`
- `search_outputs`
- `compared`
- `shipping`
- `picker`
- `summary`
- `sub_agent_results`
- `allowed_platforms`
- `allowed_category`
- `terminated`
- `terminal_tool`

`app/agent/workspace.py` 已降级为迁移标记，不再保存运行状态。

### 工具状态写入

`app/agent/tool_registry.py` 已改为：

- 使用 `ToolRuntime` 读取 checkpoint state。
- 使用 `Command(update=...)` 写入工具结果。
- 每个 Command 都带匹配 `tool_call_id` 的 `ToolMessage`。
- 原始结构化结果放在 `ToolMessage.artifact`，给模型的内容使用受限 JSON 预览。
- `shopping_summary` 和 `chat_fallback` 为终结工具。

### 子 Agent

`app/agent/dispatch_tool.py` 已改为：

- 子 Agent 创建独立 `ShopPilotState`。
- `allowed_platforms`、`allowed_category` 成为强制执行范围。
- 子 Agent 的完整状态进入父 Agent 的 `sub_agent_results`。
- 主子仍共享模型、提示词架构和 `FULL_TOOL_SET`。

### Fork 调度与背压

新增 `app/agent/fork_scheduler.py`：

- 进程级 `asyncio.Semaphore` 控制所有主任务共享的子 Agent 并发量。
- 每个 root `thread_id` 拥有独立 Semaphore，避免单个请求占满全局并发。
- 每个主任务限制唯一 fork 尝试总数，重复请求不重复消耗预算。
- 对 `demands + platform + category` 做 Unicode/空白归一化和稳定指纹去重。
- 同时到达的等价子任务复用同一个 in-flight Task；成功结果在当前主任务内按 TTL 复用。
- 全局队列设置硬容量和排队超时，防止多用户场景无界堆积。
- 主任务结束、取消或超时时取消孤儿子任务并清理预算、缓存和 Semaphore。
- 新增 `fork_queued`、`fork_dequeued`、`fork_deduplicated`、`fork_rejected` 事件；可恢复的 fork 降级不再误报为主任务 `error`。

### Cache Breakpoint 上下文治理

新增 `app/agent/context_governance.py`：

- checkpoint 中保留完整原始消息，不通过删除历史换取短上下文。
- 根据最近 K 个 `ToolMessage` 定位安全 breakpoint，并回溯到对应 `AIMessage.tool_calls`，不会切断工具调用配对。
- 只有新增历史跨过消息数或字符数阈值时才推进一个 `cache_epoch`；同一 epoch 内摘要与历史尾部保持字节稳定。
- 摘要仅增量处理上一个 breakpoint 到新 breakpoint 之间的历史，保留用户目标、硬约束、关键数值、失败尝试和下一步。
- 模型输入由稳定工作摘要和最近完整工具轮组成；持久化状态仍可审计和恢复。
- 大型工具结果在模型输入层生成合法 JSON 压缩副本，不修改 checkpoint 中的原始 `ToolMessage`。
- 记录原始/模型消息数、字符节省量、压缩工具结果数、cache read/create tokens 和 cache epoch。
- 摘要模型失败时保留完整历史尾部继续执行，不因压缩故障中断购物任务。
- 可选为 OpenAI 官方模型传入稳定 `prompt_cache_key`；通用 OpenAI-compatible 服务默认关闭该字段。
- 新增 `context_compaction`、`context_compaction_failed` 和 `prompt_cache` 事件及前端标签。

### 结构化长期记忆

新增 `app/memory/models.py`，重写 `app/memory/store.py`：

- 记忆记录包含 `kind`、`scope`、`content`、`category`、`status`、`confidence`、`mention_count`、来源会话和完整时间戳。
- 支持 `preference / blacklist / history` 三类记忆，以及 `global / category` 两种作用域。
- 旧版 `user_id -> [字符串]` JSON 在首次读取时自动迁移为 version 2，并使用原文件时间作为导入时间。
- 相同记忆在不同会话再次出现时提升置信度和提及次数；同一来源会话重复收尾保持幂等。
- 相反偏好使用新证据覆盖旧记录，旧记录标记为 `superseded`，同时保存 `conflict_group` 和 `supersedes` 审计链。
- 相关性召回综合词项重叠、品类匹配、全局作用域、置信度、时间衰减和排除项优先级。
- `ShopPilotState` 同时保存工具使用的纯文本偏好和完整 `long_term_memory` 结构化快照。
- 系统提示词区分排除项、软偏好和购买历史；当前输入与长期记忆冲突时以当前输入为准。
- 新增用户记忆查看、单条删除和冲突手动选择接口。
- 新增 `memory_retrieved`、`memory_updated` 事件及前端标签。

### Schema v2 合成商品数据集

改造 ShopPilot synthetic dataset 与项目商品目录：

- 新增 `coffee_cup` 咖啡杯品类，与旅行收纳、双肩包、键盘、耳机、保温杯共同形成六品类。
- 默认数据扩展为 300 个跨平台同款组、1,200 条平台商品、360 条 Query、6,010 条行为记录和 30 篇知识文档。
- 商品行新增 `schema_version / price / currency` 等项目规范字段，同时保留 `price_raw / currency_raw` 历史兼容字段。
- `attributes` 原生包含中文品类、稳定 category_key、category_path、材质、风格、功能和 tags。
- 项目 `Candidate` 扩展为标准商品 Contract，`same_group_id` 贯穿召回、比价、运费和最终精选。
- `app/recall/catalog.py` 从手写商品列表改为 Schema v2 JSONL 直读、版本校验、可售过滤和文件修改时间缓存。
- 正式 `item_search` 直接处理 `Candidate`，不再执行数据集私有字段到项目字段的运行时映射。
- 数据集校验器新增 Schema、六品类、咖啡杯四平台、tags、一致性和引用完整性检查。
- 一键项目验证新增数据集校验步骤；数据集批处理脚本生成后自动执行校验。

### Hybrid Retrieval v1

新增 `app/recall/tokenizer.py`、`bm25.py`、`embeddings.py`、`vector_index.py`、`hybrid.py` 和 `evaluation.py`：

- 中英文混合 tokenizer 对英文词项及中文连续串、bigram、trigram 建立稳定 token。
- 使用项目内 BM25Okapi 实现关键词召回，不依赖外部分词服务。
- 保留无需下载模型的 Hashing fallback；正式配置使用 `sentence_transformers` provider 与 BGE 中文模型。
- 商品向量按数据集指纹持久化到 `data/retrieval/`，数据未变化时直接复用。
- 使用 Faiss HNSW + 归一化内积执行向量召回；缺少可选依赖时保留精确向量回退。
- 按全库、品类、平台、平台×品类建立分区索引，避免全库召回后过滤导致品类漏召。
- 使用 RRF 融合 BM25 与向量排名，并按偏好、预算、评分、销量、时效和质量做轻量重排。
- 预算、可售状态和排除材质在召回层执行硬过滤；未知品类返回空结果而不是整个平台兜底。
- `ItemSearchOutput.retrieval` 保存通道数量、Faiss 引擎、Embedding provider、耗时和 Top score 审计信息。
- 新增 `retrieval_search` 事件及前端标签。
- 离线评测按 `same_group_id` 去重，对 lexical/vector/hybrid 分别计算 Recall@K、HitRate@K、MRR、NDCG 和硬约束满足率。
- 融合权重仅在 dev 集调优，锁定后在 test 集生成 `output/retrieval-evaluation.json`。

### BGE 语义 Embedding 与 Pairwise LTR

新增 `app/recall/ltr.py`、`ltr_training.py`、`app/recall/artifacts/ltr-v1.json`、`scripts/tune_retrieval.py` 和 `scripts/train_reranker.py`：

- 正式检索配置切换为 `BAAI/bge-small-zh-v1.5`，使用 512 维归一化语义向量。
- 按 BGE 模型卡为 Query 增加中文检索指令；Document 不加 Query 指令。
- SentenceTransformers 和 Hashing provider 均增加有界 Query 向量缓存。
- 仅使用 dev split 搜索 BM25/Vector RRF 与规则重排权重，最终锁定为 `1.5 / 0.25 / 0.25`。
- LTR 使用 19 个特征：RRF、BM25、Vector、双路排名、材质/风格/功能、偏好覆盖、预算、评分、销量、时效、质量与行为先验。
- 行为先验只聚合允许训练的 query_id，包含平滑 CTR、收藏率、购买率和负反馈率。
- train split 生成 pairwise 正负样本；dev 只选择 Logistic Regression 正则强度；选定后使用 train+dev 重训最终模型。
- 训练过程明确记录 `test_split_used=false`，test 不参与样本生成、行为先验或超参数选择。
- 包内 LTR artifact 校验 feature version、Embedding provider 和模型名；不兼容时 `auto` 自动回退规则重排。
- learned reranker 只重排候选 Top-N；预算、可售与排除材质仍在学习排序前做硬过滤。
- `ItemSearchOutput.retrieval` 增加 reranker 模式、模型训练元数据和 learned score 审计字段。

### CategoryInsight Grounded Hybrid RAG

新增 `app/knowledge/catalog.py`、`models.py`、`retriever.py`、`evaluation.py`、`synthesis.py` 和 `grounded_evaluation.py`：

- 直接读取并校验 `category_knowledge.jsonl` 的 30 篇六品类知识文档。
- 与商品检索共享 tokenizer、Embedding provider、Query cache、Faiss HNSW 和向量缓存目录。
- 按品类建立五篇文档分区，使用 BM25+BGE+Faiss+RRF 检索，未知品类返回空证据。
- `CategoryInsightOutput` 新增 `category_key / citations / evidence_summary / answer_mode / retrieval`。
- 每条引用保留 `doc_id / source / updated_at / snippet / BM25 rank / vector rank`。
- 无在线 LLM 时，使用知识证据加 1,200 条商品统计生成材质/风格/功能分布、价格带和热销同款。
- 配置模型后使用结构化 Tool Calling 生成 grounded claims；每条 claim 必须携带允许列表中的引用 ID。
- 校验虚构引用、缺失引用、数字依据、词项支持度、文档编号误判以及“多数、全部、官方、保证”等过度推断措辞。
- 校验失败时最多修复一次，仍不合格则自动回退确定性证据，不把未验证文本交给 Agent。
- 新增 `knowledge_retrieval`、`knowledge_synthesis` 事件、前端标签和系统 Prompt 引用约束。
- 新增 30 条知识主题评测和 6 个已知品类+1 个未知品类真实模型评测。
- 当前知识 Hybrid Recall@1/MRR@5 为 1.0；`qwen-flash` 固定 Grounded RAG 用例的生成成功率、拒答率、引用覆盖率和数字依据率均为 1.0，虚构引用为 0。
- `shopping_summary` 改为工具结果确定性渲染，LLM 仅提取稳定偏好；已校验品类依据由代码原样附加。

### Token 流与工具执行 Harness

新增 `app/agent/tool_reliability.py`，并升级 `app/agent/graph_runtime.py`：

- Agent 使用 `stream_mode=["messages", "updates"]`，在保留步骤级 `assistant_call` 的同时发送 `assistant_token` 增量。
- Token 流只接受 `langgraph_node=model` 的主模型文本；上下文压缩模型通过 tag 排除，工具内部结构化 LLM 输出按节点排除。
- Token 增量不写入有限 replay/trace 环形缓冲，避免高频事件挤掉工具、熔断和任务完成记录。
- 工具异常分为 `rate_limit / timeout / network / business / internal`；只对前三类瞬时错误自动重试。
- 重试使用指数退避、最大延迟和可选 jitter；每次尝试都有独立分级超时。
- 按工具类型区分 LLM、搜索、计算、Web 和子 Agent 超时预算。
- 连续瞬时失败达到阈值后按工具名打开进程级熔断器；冷却后只允许一个 half-open 探针，成功后恢复。
- 幂等键由工具名、规范化参数和相关 checkpoint state 指纹生成；恢复执行可直接重放 checkpoint 结果。
- 同一线程中的并发等价工具调用共享一个 in-flight Task，并为当前 tool call 重写正确的 `tool_call_id`。
- `dispatch_tool` 保留专用 fork 去重，不在通用工具缓存中重复保存子线程结果。
- `ShopPilotState` 新增 `tool_idempotency` 和 `tool_reliability`，`result.json` 记录缓存条目数量及最后执行指标。
- 新增 `tool_attempt`、`tool_retry`、`tool_success`、`tool_failure`、`tool_idempotency_hit`、`tool_circuit_*` 事件和前端展示。

### 动态工具权限与事件

`app/agent/graph_runtime.py` 已改为：

- `state_schema=ShopPilotState`。
- 根据 checkpoint state 动态过滤工具。
- 使用 `astream` 的 `messages + updates` 多模式同时获取 Token 与模型步骤。
- 每个模型节点推送 `assistant_call`，主模型文本增量推送 `assistant_token`。
- 运行结束后从 `aget_state()` 获取最终 checkpoint 快照。

### API 与前端

- 修复 `/api/task` 的 `get_agent_mode()` 残留错误。
- 取消任务时推送 `task_cancelled`。
- 前端增加 `assistant_call`、`assistant_token`、工具重试、幂等和熔断事件展示。
- 前端标题改为 `LangGraph AgentLoop`。

## 4. 验证结果

### Python 编译

```text
python -m compileall -q app tests scripts
结果：通过
```

### LangGraph 构建

使用本地占位 OpenAI-compatible 配置构建 Agent：

```text
结果类型：CompiledStateGraph
```

同时检查了工具 Schema，`ToolRuntime` 均未暴露给模型参数。

### 自动测试

```text
python -m pytest -q
结果：60 passed
```

测试覆盖：

- LLM 结构化 Planner。
- LLM ShoppingSummary。
- checkpointed 初始状态。
- 主入口持久化业务状态。
- 完整工具集。
- 终结工具 `return_direct`。
- 阶段化动态工具权限。
- 并行 mapping reducer。
- 子 Agent 平台/品类执行层权限。
- `astream v2 + aget_state` 步骤流与 checkpoint 快照。
- `shopping_summary` 终结工具更新 state 并直接收敛。
- 路径越权保护。
- 等价 fork 的 in-flight/TTL 去重。
- 全局与单主任务并发上限。
- 唯一 fork 总预算及主任务结束后重置。
- 排队超时后的许可证和指纹清理。
- 进程级队列硬容量。
- Cache Breakpoint 不切断 AI tool call / ToolMessage 配对。
- 工具结果压缩副本保持合法 JSON。
- 同一 cache epoch 的稳定摘要前缀复用。
- cache read token、字符节省量和 epoch 指标写入 checkpoint state。
- 摘要失败时保留完整历史继续执行。
- `create_agent` 集成下模型使用压缩窗口、checkpoint 保留完整消息。
- 旧字符串记忆文件自动迁移为 version 2 结构。
- 相同记忆的跨会话置信度强化与同会话幂等。
- 相反偏好的 superseded 冲突链与手动恢复。
- 全局记忆保留、无关品类历史过滤和相关性排序。
- 单条长期记忆删除。
- 主模型 Token 流与内部摘要/工具 LLM Token 过滤。
- 瞬时网络错误重试成功和指数退避次数记录。
- 工具超时分类与结构化错误 ToolMessage。
- checkpoint 幂等重放并重写当前 `tool_call_id`。
- 并发等价工具调用的 in-flight 单次执行复用。
- 熔断器打开、快速拒绝、half-open 探针和成功恢复。
- Token 瞬时事件不占用持久事件环形缓冲。
- 中文 BM25 对咖啡杯、耳机和旅行用品词项的稳定排序。
- Faiss HNSW 平台×品类分区、Hashing 回退和 BGE 512 维语义向量召回。
- BM25/向量 RRF 融合、预算与材质硬过滤、规则重排和 learned Top-N 重排。
- 未知品类返回空召回，避免整个平台错误兜底。
- LTR artifact 的特征版本、provider/model 兼容、test split 隔离和自动降级。
- 材质匹配学习特征、learned score 输出与行为先验。
- lexical/vector/hybrid rules/hybrid LTR 的 Recall、MRR、NDCG 和硬约束评测。
- Schema v2 的 1,200 条商品可直接反序列化为 `Candidate`。
- 咖啡杯 50 个同款组完整覆盖四个平台。
- `item_search` 直接从 JSONL 目录检索咖啡杯，不经过运行时字段映射。

### 检索评测

使用 54 条 test Query、按 `same_group_id` 去重；BGE、融合参数和 LTR 均已在 train/dev 锁定：

| 通道 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | 硬约束@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 + Rules | 0.637037 | 0.811111 | 0.976852 | 0.806064 | 0.764815 |
| BGE Vector + Rules | 0.448148 | 0.640741 | 0.819444 | 0.602272 | 0.787603 |
| BGE Hybrid + Rules | 0.625926 | 0.807407 | 0.967593 | 0.799917 | 0.785185 |
| BGE Hybrid + LTR | 0.748148 | 0.862963 | 0.990741 | 0.877783 | 0.772222 |

BGE 单路向量在这套规则生成标签上弱于 BM25，但与 lexical 融合后为 LTR 提供了补充特征。LTR 显著提升 Recall 和 NDCG；硬约束@5 从规则 Hybrid 的 0.894444 降到 0.838889，因此学习排序不能替代前置硬过滤。以上指标仅代表合成数据离线实验。

### 前端

```text
npm run build
结果：通过
```

### 当前验证环境

项目已固定使用：

```text
D:\\Software\\Python\\python.exe
Python 3.12.10
```

验证脚本使用该解释器创建 `.venv312`，避免系统 Python 3.14 的 LangChain/Pydantic 兼容性警告。

首次 Python 3.12 验证发现并修复：

- setuptools 自动包发现把 `app/data/output/frontend` 识别成多个顶级包；已在 `pyproject.toml` 明确只打包 `app*`。
- PowerShell 原生命令失败不会自动触发 `$ErrorActionPreference`；验证脚本现检查每一步 `$LASTEXITCODE`。
- pytest 清理系统临时目录出现 Windows 权限错误；已固定使用项目内 `.pytest-tmp`。

## 5. 技术依据

本轮实现遵循当前 LangChain/LangGraph 官方模式：

- 自定义 Agent State 通过 `state_schema` 注册。
- 工具通过 `ToolRuntime` 读取 state。
- 工具通过 `Command(update=...)` 更新 state，并附带匹配 tool call 的 `ToolMessage`。
- 并行工具更新同一字段时使用 reducer。
- 动态工具权限通过 `wrap_model_call` 修改本轮暴露给模型的工具。
- Agent 同时使用 `astream` 的 `messages` 与 `updates` 模式获取 Token 增量和完整模型步骤。
- 自定义 `AgentMiddleware.awrap_model_call` 可覆盖本轮模型消息，并通过 `ExtendedModelResponse + Command` 追加状态指标。
- 自定义 `AgentMiddleware.awrap_tool_call` 可围绕每次工具调用实施重试、超时、熔断、幂等和状态更新；一次模型工具调用的内部重试不会重复消耗 ToolCallLimit 配额。
- Prompt Cache 依赖完全一致的前缀；因此摘要按粗粒度 epoch 更新，而不是每轮微改写历史。
- checkpointer 持久化图状态；跨线程长期信息应进入 Store。

参考：LangChain 官方 Tools、Context、Middleware、Streaming 和 LangGraph Persistence 文档（2026-08-05 核对）。

## 6. 下一阶段建议顺序

1. 建立 30～50 条人工审核和对抗 RAG 问答集，评估 citation precision、faithfulness、answer relevance 与拒答边界。
2. 使用 users/interactions 建立 User Tower 与个性化召回、重排实验。
3. 增加小型 Cross-Encoder 对照实验，与当前线性 LTR 比较效果、延迟和过拟合风险。
4. 将本地 JSON 记忆后端替换为 PostgreSQL/向量 Store，并补多实例一致性测试。
5. 为 PostgreSQL checkpoint 后端补真实容器集成测试和数据库迁移发布流程。

## 7. 进度日志

### 2026-08-05 12:34

- [x] 完成当前代码与文档差距分析。
- [x] 创建本进度文件。

### 2026-08-05 13:18

- [x] 修复 P0 残留错误和前端旧文案。
- [x] 完成业务状态向 `ShopPilotState` 迁移。
- [x] 完成 `ToolRuntime + Command` 工具状态写入。
- [x] 完成终结工具、动态工具权限和子 Agent 强制范围。
- [x] 完成子 Agent 全量结构化产物记录。
- [x] 完成 `assistant_call` 和 `task_cancelled` 事件。
- [x] 完成合法 JSON 压缩预览。
- [x] 通过 Python 编译、14 个自动测试、LangGraph 构建和前端生产构建。

### 2026-08-05 13:20

- [x] 确认并使用 `D:\\Software\\Python\\python.exe`，版本为 Python 3.12.10。
- [x] 创建并使用 `.venv312` 隔离环境。
- [x] 将本地 checkpointer 从 `InMemorySaver` 升级为 `AsyncSqliteSaver`。
- [x] 加入 WAL、`busy_timeout`、安全 msgpack 配置和连接生命周期管理。
- [x] 增加 `POST /api/task/{thread_id}/resume` 恢复接口。
- [x] 增加 checkpoint 查看和按线程删除接口。
- [x] 增加 SQLite 关闭后重新打开并恢复完整状态的自动测试。
- [x] 修复 setuptools 包发现、PowerShell 退出码检查和 pytest 临时目录权限问题。
- [x] Python 3.12 验证结果：源码编译通过、`CompiledStateGraph` 构建通过、15 tests passed、前端生产构建通过。

### 2026-08-05 14:22

- [x] 新增 `ForkScheduler`，完成进程级与单主任务两层 Semaphore。
- [x] 增加每个主任务的唯一 fork 总预算和运行结束自动清理。
- [x] 增加等价平台/品类子任务的 in-flight 合并和短期成功结果复用。
- [x] 增加队列硬容量、FIFO 等待、排队超时和超时后许可证/指纹清理。
- [x] 增加 `fork_queued`、`fork_dequeued`、`fork_deduplicated`、`fork_rejected` AGUI 事件和前端展示。
- [x] 将 fork 预算耗尽、队列满、排队超时和子 Agent 失败标记为可恢复降级，避免前端误判主任务结束。
- [x] 主任务完成、取消、恢复结束及服务退出时清理孤儿子 Agent。
- [x] 项目版本更新为 `0.3.0`。
- [x] Python 3.12.10 完整验证：源码编译通过、`CompiledStateGraph` 构建通过、20 tests passed、前端生产构建通过。

### 2026-08-05 14:46

- [x] checkpoint 后端扩展为 `memory / sqlite / postgres`，生产配置使用官方 `AsyncPostgresSaver`。
- [x] 新增 `postgres` 可选依赖组，并在 Python 3.12 环境确认适配器、psycopg binary 和连接池驱动可导入。
- [x] PostgreSQL DSN 在健康检查中只展示主机、端口和数据库，不泄露用户名或密码。
- [x] 新增 checkpoint 保留周期、启动清理、周期清理、扫描上限和单次删除批次上限。
- [x] 清理逻辑通过统一 saver `alist/adelete_thread` 运行，可复用于 SQLite、PostgreSQL 和内存后端。
- [x] 自动清理跳过正在运行的主任务；删除后同步清理进程内结果缓存。
- [x] 新增 `POST /api/checkpoints/cleanup` 手动清理接口。
- [x] `SHOPPILOT_CHECKPOINT_RETENTION_DAYS=0` 可关闭自动清理；清理不删除 `output/` 审计产物。
- [x] 新增过期线程、活动线程保护、批次限制、禁用开关和 DSN 脱敏测试。
- [x] 项目版本更新为 `0.4.0`。
- [x] Python 3.12 完整验证：源码编译通过、`CompiledStateGraph` 构建通过、`AsyncPostgresSaver` 导入通过、24 tests passed、前端生产构建通过。

### 2026-08-05 15:23

- [x] 移除通用 `SummarizationMiddleware`，新增自定义 `CacheBreakpointMiddleware`。
- [x] checkpoint 保留完整消息，模型调用使用稳定摘要与最近完整工具轮构成的压缩窗口。
- [x] breakpoint 回溯对应 AI tool call，避免切断 `AIMessage` / `ToolMessage` 配对。
- [x] 摘要按消息数或字符阈值粗粒度推进 `cache_epoch`，同一 epoch 内前缀稳定。
- [x] 工具大结果生成合法 JSON 的模型输入副本，原始 artifact 和 checkpoint 不变。
- [x] 新增上下文字符节省、消息数、工具压缩数和 Prompt Cache token 指标。
- [x] 新增 `context_compaction`、`context_compaction_failed`、`prompt_cache` 事件及前端标签。
- [x] 摘要失败时降级为完整历史尾部，不中断 AgentLoop。
- [x] OpenAI `prompt_cache_key` 设为可选，通用 OpenAI-compatible 服务默认不发送未知参数。
- [x] 项目版本更新为 `0.5.0`。
- [x] Python 3.12 完整验证：5 个上下文治理测试通过、全部 29 tests passed、`CompiledStateGraph` 构建通过、`AsyncPostgresSaver` 导入通过、前端生产构建通过。

### 2026-08-05 15:53

- [x] 新增 version 2 结构化长期记忆模型，覆盖偏好、排除项和购买历史。
- [x] 记录全局/品类作用域、来源会话、创建/更新时间、访问次数、置信度和提及次数。
- [x] 旧版 `user_id -> [字符串]` 文件首次读取时自动迁移，保留原文件时间和 legacy 来源。
- [x] 相同证据跨会话提升置信度；同一会话重复收尾不重复强化。
- [x] 相反偏好建立 `conflict_group / supersedes` 链，新证据生效并将旧记录标记为 `superseded`。
- [x] 相关性召回加入词项、品类、作用域、置信度、时间衰减和 blacklist 优先级。
- [x] `ShopPilotState` 新增 `long_term_memory`，Prompt 明确区分硬排除、软偏好和历史背景。
- [x] 新增记忆查看、单条删除和冲突手动恢复接口。
- [x] 新增 `memory_retrieved`、`memory_updated` 事件及前端标签。
- [x] 项目版本更新为 `0.6.0`。
- [x] Python 3.12 完整验证：结构化记忆专项 5 passed、全部 34 tests passed、`CompiledStateGraph` 构建通过、`AsyncPostgresSaver` 导入通过、前端生产构建通过。

### 2026-08-05 17:17

- [x] Agent 流模式升级为 `messages + updates`，新增主模型 `assistant_token` WebSocket 增量。
- [x] 上下文摘要 LLM 使用专用 tag，工具内部 LLM 按节点过滤，避免内部文本泄露到用户实时回答。
- [x] Token 增量设为瞬时事件，不占用 200 条 replay/trace 环形缓冲。
- [x] 新增 `ToolReliabilityMiddleware`，统一工具错误分类、指数退避重试和分级超时。
- [x] 仅重试 rate-limit、timeout、network 瞬时错误；business/internal 错误结构化返回模型处理。
- [x] 新增按工具名的进程级熔断器、冷却窗口、单探针 half-open 和成功恢复。
- [x] 新增基于工具参数与相关 checkpoint state 指纹的幂等键，支持 checkpoint 重放和 in-flight 合并。
- [x] 幂等重放会为当前模型调用重写正确的 `tool_call_id`，避免 `INVALID_TOOL_RESULTS`。
- [x] `ShopPilotState` 新增 `tool_idempotency` 与 `tool_reliability`，结果产物记录缓存数量和工具指标。
- [x] 新增工具 attempt/retry/success/failure/idempotency/circuit 事件和前端实时展示。
- [x] 中间件顺序调整为 ToolCallLimit 在可靠性重试外层，内部重试不重复消耗工具调用预算。
- [x] 新增 8 个专项测试；全部自动测试更新为 42 passed。
- [x] 项目版本更新为 `0.7.0`。
- [x] Python 3.12.10 完整验证：可编辑安装通过、源码编译通过、`CompiledStateGraph` 构建通过、`AsyncPostgresSaver` 导入通过、42 tests passed、前端生产构建通过。

### 2026-08-05 18:25

- [x] 数据集新增 `coffee_cup` 品类，并为咖啡杯生成四平台同款商品、Query、行为和五篇知识文档。
- [x] 数据集升级为 Schema v2，原生输出项目需要的 `price / currency / attributes.category / tags` 等字段。
- [x] 默认 generated 数据重建为 300 个商品组、1,200 条商品、360 条 Query、6,010 条行为和 30 篇知识文档。
- [x] sample 数据重建为 6 个商品组、24 条商品、18 条 Query、36 条行为和 30 篇知识文档。
- [x] 数据集校验器覆盖 Schema、六品类、咖啡杯四平台、到手价、tags、同款组和引用完整性。
- [x] 项目 `Candidate` 扩展为共享商品 Contract，`same_group_id` 贯穿比价、运费和精选。
- [x] 正式商品目录从手写 Python 列表切换为 Schema v2 JSONL 直读，不再执行运行时字段映射。
- [x] 新增数据集目录配置、版本校验、可售过滤、自动重载缓存和缺失数据错误提示。
- [x] 一键验证新增数据集校验步骤，数据集批处理生成后自动校验。
- [x] 新增 3 个数据集直读专项测试；全部自动测试更新为 45 passed。
- [x] 项目版本更新为 `0.8.0`。
- [x] Python 3.12.10 完整验证：Schema v2 数据集校验通过、可编辑安装通过、源码编译通过、`CompiledStateGraph` 构建通过、`AsyncPostgresSaver` 导入通过、45 tests passed、前端生产构建通过。

### 2026-08-05 18:54

- [x] 新增中文/英文混合 tokenizer 和项目内 BM25Okapi 召回。
- [x] 新增 Hashing 与 SentenceTransformers 两种 Embedding provider；默认离线模式不下载模型。
- [x] 新增 Faiss HNSW 向量索引和缺依赖时的精确向量回退。
- [x] 按全库、品类、平台和平台×品类建立检索分区，Embedding 使用数据集指纹缓存。
- [x] 新增 BM25/向量 RRF 融合、预算/可售/排除材质过滤和轻量规则重排。
- [x] `item_search` 接入 Hybrid Retriever，并输出通道、排名、引擎与耗时诊断。
- [x] 新增 `retrieval_search` 事件和前端标签。
- [x] 新增 lexical/vector/hybrid 离线评测脚本和 `output/retrieval-evaluation.json`。
- [x] 融合权重只在 dev 集调优；test 集 54 条 Query 指标如验证章节所示。
- [x] 新增 4 个 Hybrid Retrieval 专项测试；全部自动测试更新为 49 passed。
- [x] 项目版本更新为 `0.9.0`。
- [x] Python 3.12.10 九步完整验证：数据集校验、Faiss 1.15.0、可编辑安装、源码编译、`CompiledStateGraph`、`AsyncPostgresSaver`、49 tests、test 集评测和前端生产构建全部通过。

### 2026-08-05 19:56

- [x] 安装并接入 SentenceTransformers 5.6.1、Torch 2.13.0 和 `BAAI/bge-small-zh-v1.5`。
- [x] 为 BGE Query 增加中文检索指令，并生成 1,200 条商品的 512 维向量缓存。
- [x] 新增 Query embedding 有界缓存，避免调参和评测重复编码。
- [x] 新增 dev-only 检索权重搜索，锁定 BM25/Vector/规则重排权重为 `1.5/0.25/0.25`。
- [x] 新增 19 维 Pairwise LTR、行为先验、模型持久化、兼容校验和规则降级。
- [x] train 生成 32,180 个 pairwise 样本，dev 选择 `C=0.03`，最终 train+dev 生成 39,488 个样本重训。
- [x] 包内 `ltr-v1.json` 明确记录 `test_split_used=false`，模型仅与 BGE provider/model 匹配时启用。
- [x] test 集 Hybrid+LTR：Recall@10 `0.862963`、MRR@10 `0.990741`、NDCG@10 `0.877783`。
- [x] 新增 5 个 LTR 专项测试；全部自动测试更新为 54 passed。
- [x] 项目版本更新为 `0.10.0`。
- [x] Python 3.12.10 九步完整验证：BGE 依赖安装、Schema v2 校验、源码编译、`CompiledStateGraph`、`AsyncPostgresSaver`、54 tests、锁定 test 评测和前端生产构建全部通过。
- [x] 标准 wheel 构建通过，并确认包含 `app/recall/artifacts/ltr-v1.json`。

### 2026-08-05 20:32

- [x] 移除 `CategoryInsight` 三品类手写知识字典和未知品类旅行收纳回退。
- [x] 新增 30 篇品类知识 JSONL 直读、字段校验、文档 ID 去重和文件变更缓存。
- [x] 与商品检索共享 Embedding provider、BGE 模型实例、Query cache、Faiss HNSW 和索引目录。
- [x] 新增品类分区 BM25+BGE+Faiss+RRF 知识检索，未知品类返回 `no_evidence`。
- [x] `CategoryInsightOutput` 新增可审计 citations、evidence summary、answer mode 和 retrieval diagnostics。
- [x] 使用知识证据与商品统计确定性生成热销同款、材质/风格/功能分布和到手价价格带。
- [x] 新增 `knowledge_retrieval` 事件、前端标签和系统 Prompt 引用约束。
- [x] 新增 30 条知识主题 Recall/MRR/NDCG 基准；BGE Hybrid Recall@1 与 MRR@5 均为 1.0。
- [x] 新增 6 个 CategoryInsight/RAG 专项测试；全部自动测试更新为 60 passed。
- [x] 一键验证扩展为十步，新增知识检索评测。
- [x] 项目版本更新为 `0.11.0`。
- [x] Python 3.12.10 十步完整验证：依赖安装、Schema v2 校验、源码编译、`CompiledStateGraph`、`AsyncPostgresSaver`、60 tests、BGE+LTR test 评测、知识检索评测和前端生产构建全部通过。
- [x] 用户完成真实在线 LLM 配置；后续 Grounded RAG 与完整 AgentLoop 冒烟均已执行。

### 2026-08-05 21:55

- [x] 验证当前 `qwen-flash` OpenAI-compatible 服务支持 LangChain `function_calling` 结构化输出。
- [x] 新增 `GroundedClaim / grounded_answer / citation_validation` 数据结构和真实模型 synthesis 模块。
- [x] 新增允许引用白名单、缺失/虚构引用、词项支持度、数字等值/百分比和绝对化措辞校验。
- [x] 新增安全修复重试、模型异常回退和未知品类不调用生成模型的拒答路径。
- [x] 将商品目录统计注册为 `CATALOG_STATS:<category_key>` 可审计证据源。
- [x] 新增 6 个已知品类 + 1 个未知品类真实 Grounded RAG 评测和 `output/grounded-rag-evaluation.json`。
- [x] `qwen-flash` 最终固定评测：grounded 成功率 1.0、无证据拒答率 1.0、引用覆盖率 1.0、数字依据率 1.0、虚构引用 0；校验器丢弃 1 条“占比最高”的过度推断。
- [x] 完整 AgentLoop 冒烟发现自由文本总结会编造“平台官方验证”；已将 ShoppingSummary 改为工具结果确定性渲染。
- [x] 最终清单原样附加已校验品类依据，并明确合成数据、非实时市场和非官方验证边界。
- [x] 新增 `knowledge_synthesis` 事件、前端标签和可选 `RUN_LIVE_RAG_TEST` 验证入口。
- [x] 新增 7 个 Grounded RAG/确定性总结专项测试；全部自动测试更新为 67 passed。
- [x] 项目版本更新为 `0.12.0`。
- [x] Python 3.12.10 十步完整验证：0.12.0 可编辑安装、Schema v2 校验、源码编译、`CompiledStateGraph`、`AsyncPostgresSaver`、67 tests、BGE+LTR test 评测、知识检索评测和前端生产构建全部通过。
