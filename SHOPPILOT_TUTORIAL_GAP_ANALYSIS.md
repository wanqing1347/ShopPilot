# ShopPilot 与《电商搜索 Agent-基础部分》教程差距分析

> 分析时间：2026-08-07
> 对比对象：当前 `ShopPilot/` 工程 vs `电商搜索Agent-基础部分/` 教程 1～19 章
> 目的：明确当前项目已经完成、已经超过教程、仍未补齐的能力，并给出后续迭代优先级。

---

## 1. 总体结论

当前 ShopPilot 已经不是“基础教程尚未做完”的状态。

在以下方向上，当前工程已经达到或超过教程早期版本的教学实现：

- LangChain `create_agent` + LangGraph Runtime 的 AgentLoop；
- 主 Agent + 同质子 Agent 动态 fork；
- 独立 thread/checkpoint 与可恢复状态；
- Cache Breakpoint 上下文治理；
- 结构化长期记忆；
- 工具错误分类、重试、超时、熔断、幂等；
- Fork 调度、背压、去重和并发预算；
- Hybrid Retrieval：BM25 + BGE + Faiss + RRF；
- Pairwise Learning-to-Rank；
- Grounded RAG + Citation/数字/措辞校验；
- FastAPI + WebSocket + React 前端；
- Rule-based Trajectory Evaluator + LLM-as-a-Judge。

如果按当前教程全部章节的“功能覆盖”粗略估计，ShopPilot 已覆盖约 **70% 左右**。

但是，这个 70% 不能理解为“已经是 70% 的工业级电商生产系统”。真正的生产化、自进化和训练闭环仍有较大缺口。

当前最主要的剩余差距集中在：

1. 真正的 User Tower / 三塔个性化召回；
2. Embedding 和 Cross-Encoder 的真实训练；
3. Rubric → SFT/RL 的训练闭环；
4. Agent 自进化体系；
5. 统一 Harness Hook Pipeline；
6. Silent Drift 漂移检测；
7. Docker / Redis / vLLM / K8s / 完整 Langfuse 等生产化能力；
8. 真实 Amazon/Shopee/AliExpress/eBay 平台数据与实时业务服务。

---

## 2. 逐章对比

| 教程部分 | ShopPilot 当前状态 | 判断 |
| --- | --- | --- |
| 1-2 AgentLoop | `create_agent + LangGraph runtime`，Think → Act → Observe → Reflect | ✅ 已完成，且实现更新 |
| 3 多 Agent fork | 独立 thread/checkpoint、并发限制、fork 去重、超时、背压 | ✅ 明显加强 |
| 4-0 三塔召回 | 当前是 Query/Item 的 BGE + BM25 + 行为 LTR，没有真正 User Tower | ⚠️ 部分完成 |
| 4-1 OpenSearch | 当前使用 Faiss HNSW 分区索引 | ⚠️ 本地版完成，生产 OpenSearch 未做 |
| 4-2 Embedding/Reranker 训练 | BGE 使用预训练模型；自己训练的是线性 Pairwise LTR | ⚠️ 部分完成 |
| 5 Cache Breakpoint | stable epoch、摘要、工具结果压缩、cache token 指标 | ✅ 很完整 |
| 6 长期记忆 | 结构化 memory、冲突、置信度、相关性召回 | ✅ 已加强 |
| 7 AGUI + WebSocket | 事件流 + WebSocket + Token streaming | ✅ 功能完成 |
| 8 Rubric/SFT/RL | Rule evaluator + LLM Judge 已有，但 SFT/RL 没有 | ⚠️ 评测有，训练无 |
| 9-10 项目工程 | 配置、Agent、API、状态、监控等 | ✅ |
| 11 商品检索 | 四个平台名称存在，但核心仍主要是离线模拟数据 | ⚠️ |
| 12 比价/物流 | 有 PriceCompare / ShippingCalc | ✅ Demo 级 |
| 13 Category RAG | Hybrid RAG + Grounded synthesis + citations | ✅ 甚至更强 |
| 14 主 AgentLoop | 已完成 | ✅ |
| 15 FastAPI + React | 已完成，还有历史评测页 | ✅ |
| 16-1 Docker Compose | 没有完整 docker 目录、compose、Dockerfile | ❌ |
| 16-2 vLLM/GPU | 可调用 OpenAI-compatible vLLM，但没有自己部署 GPU 推理服务 | ⚠️ |
| 16-3 Langfuse | 有 CallbackHandler，但没有完整 Span/Score/告警/看板 | ⚠️ |
| 16-4 Token Budget | 有 model/tool 次数限制和 Context 压缩，但没有完整 token/cost budget + 多模型路由 | ⚠️/❌ |
| 16-5 熔断/排队 | 工具熔断较完整；当前主要是 fork queue，不是完整请求优先级队列 | ⚠️ |
| 16-6 Security/K8s | 基本未实现 | ❌ |
| 17-1 Harness | reliability/context/permission 等思想已经存在 | ✅ |
| 17-2 Hook Pipeline | 使用 LangChain middleware，但没有教程统一六 Hook HarnessMiddleware | ⚠️ |
| 17-3 Silent Drift | 没有 semantic assertion / drift detector | ❌ |
| 17-4 阶段状态机 | 有 `select_tool_names()` 动态工具权限，但不是完整 PhaseStateMachine | ⚠️ |
| 18 Agent 自进化 | 基本没有完整闭环 | ❌ |
| 19 Prompt 工程 | XML 分层、fork policy 已有；Prompt 版本/A-B/Skill 不完整 | ⚠️ |

---

## 3. 最大差距一：真正的三塔个性化召回

教程中的目标是：

```text
User Tower
     ↓
个性化召回
     +
Query Tower → Item Tower
     ↓
语义召回
```

当前 ShopPilot 更接近：

```text
Query
 ├─ BM25
 └─ BGE Embedding
       ↓
    Faiss
       ↓
      RRF
       ↓
规则特征 + 行为特征
       ↓
Linear Pairwise LTR
```

当前方案本身是合理的工程实现，而且更容易测试和做离线评测，但它不能称为“完整三塔召回”。

目前仍缺：

- User Embedding；
- 用户历史行为编码；
- User → Item ANN 召回通道；
- Semantic / Personalized 双通道融合；
- 多任务训练；
- User / Query / Item 三塔联合训练。

### 建议

这是后续比较值得补的核心能力。可以在现有 Hybrid Retrieval 上增加一条 Personalized Recall，而不是推翻当前架构。

建议演进为：

```text
              ┌─ Semantic Recall ─ Query → Item
User Query ───┤
              └─ Personalized Recall ─ User → Item
                         ↓
                        RRF
                         ↓
                     Reranker
```

---

## 4. 最大差距二：Embedding / Cross-Encoder 尚未真正训练

当前 Embedding 使用：

```text
BAAI/bge-small-zh-v1.5
       ↓
直接推理
```

当前 Reranker 更准确地说是：

```text
19 维手工/行为特征
      ↓
Pairwise Logistic Regression
```

这是标准的 Learning-to-Rank 方法，也是当前项目一个很好的工程亮点。

但教程 4-2 更进一步要求：

```text
Embedding CPT
→ Contrastive SFT
→ Hard Negative Mining
→ Cross-language Alignment
```

以及：

```text
CrossEncoder
→ LoRA
→ Pairwise/Listwise Training
→ Preference Alignment
```

因此当前项目介绍中建议使用：

> BGE Hybrid Retrieval + Pairwise Learning-to-Rank

不要直接说已经实现：

> 完整三塔训练 + CrossEncoder 精排训练。

---

## 5. 最大差距三：有评测系统，但还没有训练飞轮

当前已经实现：

```text
trace.json
     ↓
Rule-based Trajectory Evaluator
     +
LLM-as-a-Judge
     ↓
evaluation.json
```

LLM Judge 已经覆盖：

- planning_quality；
- tool_selection；
- trajectory_efficiency；
- final_answer_quality。

这说明 ShopPilot 已经具备比较完整的 Evaluation System。

但教程真正目标是：

```text
Agent Trajectory
      ↓
动态 Rubric
      ↓
P0 / P1 / P2
      ↓
Bad Case
 ┌────┼─────┐
Rule  SFT    RL
 │     │      │
 └─────┴──────┘
       ↓
新模型
       ↓
重新评测
```

目前缺失：

- trajectory admission；
- 高分轨迹训练集；
- SFT 数据构建；
- loss mask；
- rollout；
- reward function；
- GSPO / PPO 等 RL 训练；
- 模型版本对比评测。

因此更准确的项目定义是：

> 当前已经完成 Evaluation System，还没有完成 Evaluation & Training Flywheel。

对于 Agent 工程岗位，这部分不需要作为当前最高优先级。

---

## 6. 最大差距四：Agent 自进化体系尚未完成

教程 18 章的整体闭环是：

```text
Monitor
 ↓
Bad Case Collector
 ↓
Analyzer
 ↓
P0 / P1 / P2 Router
 ↓
Evolution
```

当前项目没有完整的：

- `app/evolution/`；
- bad-case collector；
- 自动 P0/P1/P2 路由；
- prompt evolution；
- prompt A/B 自动放量；
- strategy extractor；
- fork optimizer；
- failure lessons；
- 策略生命周期管理。

### 当前长期记忆 vs 教程记忆自进化

当前 memory 主要保存：

```text
用户说：
“不喜欢塑料”
 ↓
Memory Store
```

教程 18-4 进一步要求：

```text
高分 Agent trajectory
 ↓
提炼成功策略
 ↓
Strategy Memory
 ↓
下一次 Agent 自动参考
```

也就是说，当前保存的是：

> 用户是谁、喜欢什么。

教程后续还希望保存：

> Agent 自己过去怎么做更容易成功。

这是当前明显未覆盖的一层。

---

## 7. 最大差距五：Harness 尚未统一成完整 Hook Pipeline

ShopPilot 目前已经拥有很多 Harness 控制能力：

```text
CacheBreakpointMiddleware
ToolReliabilityMiddleware
stage_tool_permissions
ModelCallLimitMiddleware
ToolCallLimitMiddleware
ForkScheduler
Checkpoint
```

因此从工程思想上看，Harness 已经存在。

但教程 17-2 希望进一步统一为：

```text
pre_think
post_think
pre_tool_call
post_tool_call
pre_reflect
post_reflect
```

并通过统一的 `HarnessMiddleware` 管理：

```text
HarnessMiddleware
 ├─ tool_whitelist
 ├─ loop_detector
 ├─ truncate
 ├─ schema_validator
 ├─ sequencing
 ├─ drift_detector
 └─ phase_transition
```

当前项目的问题不是“完全没有这些控制能力”，而是这些能力仍分散在多个 runtime/middleware 模块中。

### 建议

后续可以建立 `app/harness/`，把已有控制逻辑逐步归一化，而不是重新实现一次。

---

## 8. 最大差距六：Silent Drift 尚未实现

当前系统已经可以发现：

- timeout；
- network error；
- rate limit；
- 工具重复调用；
- 缺平台；
- 工具失败；
- fork 失败；
- 调用预算超限。

但是还缺少一种没有异常、却非常危险的失败：

```text
用户：我要买咖啡杯

Agent：
搜咖啡杯
→ 看材料
→ 搜保温杯
→ 查物流
→ 开始讨论旅行收纳
→ 最终正常结束
```

这里可能：

- 没有 crash；
- 没有 exception；
- 所有工具都执行成功；
- 但 Agent 已经偏离用户目标。

这就是 Silent Drift。

目前项目还缺：

- Semantic Assertion；
- Goal Similarity Check；
- 每 N 轮 Drift Detection；
- Drift Correction；
- 连续严重 drift 后强制终止。

这是一个很值得作为 P1 补齐的功能。

---

## 9. 最大差距七：生产化能力仍有明显缺口

当前已经具备：

```text
FastAPI
React
WebSocket
SQLite Checkpoint
PostgreSQL Saver 支持
Langfuse Callback
Circuit Breaker
Fork Queue
```

但教程 16 章继续要求的生产化能力，目前大部分还没有完整落地：

```text
Docker
Docker Compose

Redis
全局 Task Queue

vLLM GPU Deployment
Reranker GPU Service

Token Budget
Model Router
Cost Accounting

Kubernetes
Ingress
Graceful Shutdown
Canary Release

Prompt Injection Defense
Output Guard
Log Redaction
```

### 9.1 Langfuse 当前状态

当前代码确实已经存在 Langfuse Callback：

```python
CallbackHandler()
```

并注入：

```text
langfuse_session_id
langfuse_tags
```

所以不能说 Langfuse 完全没有。

但教程目标是完整：

```text
Trace
 ├─ LLM generation
 ├─ tool span
 ├─ fork span
 ├─ token usage
 ├─ latency
 └─ rubric score
```

并进一步增加：

- alerts；
- dashboard；
- bad-case 定位 SOP；
- score 回写。

因此当前更准确的定义是：

> Langfuse basic tracing integration

而不是完整 observability platform。

---

## 10. 最大差距八：跨平台搜索仍不是真实四平台生产数据

当前核心面试数据中，平台字段主要模拟：

```text
Amazon
Shopee
AliExpress
eBay
```

但它们主要来自可复现的离线合成数据集，并不是实时官方平台数据。

因此当前结构更接近：

```text
Amazon       ┐
Shopee       │
AliExpress   ├─ Synthetic Dataset
Ebay         ┘
```

而不是：

```text
Amazon API
Shopee API
AliExpress API
Ebay API
```

项目已经存在 provider 抽象和部分真实/公开数据源验证，例如：

- eBay provider；
- Rakuten provider；
- public_demo provider。

但完整真实业务还缺：

- 官方/合规平台 API；
- OAuth；
- 实时库存；
- 实时价格；
- 实时汇率；
- HS Code；
- 实际关税规则；
- 真实物流价格和时效。

### 面试表达建议

如果被问：

> 你的 Agent 能实时搜索 Amazon 商品吗？

当前应准确回答：

> 核心面试演示使用可复现的离线同构数据，避免平台 API 稳定性和授权问题干扰 Agent 算法验证。工程层已经抽象 Catalog Provider，并对部分公开 API/数据源做过接入验证；真实交易平台和实时库存属于下一阶段生产接入工作。

---

## 11. ShopPilot 已经超过教程或明显加强的部分

### 11.1 Checkpoint

当前已经具备：

```text
SQLite
PostgreSQL optional
Retention
Cleanup
Resume
Thread Delete
Structured Business State
```

这已经超过很多普通教学 Demo。

### 11.2 Fork Scheduler

当前实现：

```text
Global Concurrency
Per-task Concurrency
Fork Budget
FIFO Waiting
Queue Timeout
Queue Capacity
In-flight Dedup
TTL Dedup
Orphan Cancellation
```

这是当前项目非常好的工程亮点。

### 11.3 Tool Reliability Harness

当前已经实现：

```text
错误分类
Timeout
Retry
Exponential Backoff
Jitter
Circuit Breaker
Half-open Recovery
Idempotency
In-flight Reuse
```

这部分比“单纯用了 LangGraph”更有面试价值。

### 11.4 Grounded RAG

当前 CategoryInsight 不是简单：

```text
Vector Search → LLM
```

而是：

```text
BM25
+BGE
+Faiss
+RRF
 ↓
Grounded LLM
 ↓
Citation Validator
 ↓
数字依据校验
 ↓
绝对化措辞检测
 ↓
失败自动降级
```

这是当前项目最值得重点讲的模块之一。

### 11.5 Retrieval Evaluation

当前已经真正实现：

- Recall@K；
- HitRate@K；
- MRR；
- NDCG；
- Hard Constraint Rate；
- train/dev/test 分离；
- BM25 / Hybrid / LTR 对照实验。

这使得检索模块不是“凭感觉有效”，而是可以用实验结果说明改造收益。

---

## 12. 后续优先级建议

不建议为了和教程一模一样而按 16 → 17 → 18 → 19 章机械补齐。

更建议根据当前项目价值和秋招面试目标安排优先级。

### P0：核心 Demo 300s 超时（已修复）

历史真实运行曾出现：

```text
item_search 第 3/3 次执行
主 AgentLoop 超过 300s
```

2026-08-07 已定位并修复。根因不是单纯的“主 Agent 思考过久”，而是检索冷启动与 Tool Reliability 超时发生了放大：

1. 四个平台 fork 冷启动时会并发调用 `get_hybrid_retriever()`；`lru_cache` 只能保证缓存字典线程安全，不能阻止并发 cache miss 重复执行构造函数，因此曾同时加载多份 BGE/索引；
2. 2,200 条商品的 BGE Embedding 缓存 fingerprint 包含 `products.jsonl` 的 mtime/path，文件被复制或重新生成后即使向量文本没变也会整库重算；
3. `SentenceTransformer` 在本地已有模型时仍可能访问 Hugging Face Hub，当前环境仅远端检查就约 30 秒，恰好撞上 `item_search` 的 30 秒工具超时；
4. `asyncio.wait_for(asyncio.to_thread(...))` 超时只能取消等待协程，不能停止已经运行的后台线程，随后 retry 会继续创建新的等待/计算线程，从而出现“30s timeout → retry → 更多冷启动任务”的雪崩。

已实施：

- Embedding Provider 初始化增加进程级 single-flight 锁；
- HybridRetriever 初始化增加进程级 single-flight 锁；
- BGE 改为本地 Hugging Face snapshot 优先，本地不存在时才联网；
- Embedding 缓存 fingerprint 改为“实际向量文本内容 + provider/model/dimension”，不再依赖 mtime/path；
- 兼容当前已有 2,200 商品 legacy BGE cache；
- 主 Agent 300 秒预算开始前先预热商品检索器，冷启动不再占用 `item_search` 30 秒 retry 窗口；
- 新增 `SHOPPILOT_RETRIEVAL_WARMUP_TIMEOUT_SEC=240`；
- 新增并发 cache miss 与稳定 fingerprint 回归测试。

验证结果：

```text
修复前：4 路并发冷初始化 >150s，且实际创建 4 个模型实例
修复后：4 路并发冷初始化约 17s，unique_instances=1
热路径：约 0.001s
```

真实四平台 Demo 额外把主 Agent 超时从 300 秒临时收紧到 240 秒执行，仍成功完成；本次 Trace 中 4 次 `item_search` 均为一次成功，耗时约 20~66ms，无 retry、无 tool failure、无 fork rejected。整个事件链约 167 秒完成。

---

### P1：补 Semantic Assertion + Silent Drift

建议新增：

```text
app/harness/
├── step_validator.py
├── sequencing.py
└── drift_detector.py
```

并将当前已有的：

```text
ToolReliability
StagePermission
ContextGovernance
ForkScheduler
```

正式组织成：

> ShopPilot Harness Engineering

这样可以明显提升项目架构完整度。

---

### P1：补真正 User Tower

建议在现有 Hybrid Retrieval 上增加 Personalized Recall：

```text
Semantic Recall：Query → Item
Personalized Recall：User → Item
        ↓
       RRF
        ↓
     Reranker
```

这样就可以真正对齐教程“三塔 + 双通道”的设计。

---

### P1：补 Prompt Version + A/B

不必一开始做完整自动自进化。

可以先实现：

```text
prompts/
├── v1.0.yml
├── v1.1.yml
└── registry.py
```

使用：

```text
thread_id hash
    ↓
Prompt A / Prompt B
```

然后利用现有：

```text
Rule Evaluator
LLM Judge
```

比较：

- score；
- tool calls；
- latency；
- success rate；
- retry/fork 情况。

这样现有 Evaluation System 就开始真正服务于 Prompt 优化。

---

### P2：Docker + Langfuse 完善

建议补：

```text
docker/
├── Dockerfile
└── docker-compose.yml
```

以及完整 Langfuse：

```text
Agent Trace
  ├─ Model Span
  ├─ Tool Span
  ├─ Fork Span
  └─ Judge Score
```

这已经足够应对大部分生产化面试追问。

K8s 可以之后再做。

---

### P3：SFT / RL

建议最后做。

对于 AI 应用 / Agent 工程岗位来说：

```text
稳定 Agent
+
检索/RAG
+
评测
+
Harness
+
可观测性
```

通常比“照教程跑了一遍 RL”更有说服力。

如果没有真实的大规模 Agent trajectory 数据，强行做 SFT/RL 很容易变成展示性质功能，而不是能够说明业务收益的工程能力。

---

## 13. 当前项目最准确的定位

建议将 ShopPilot 定位为：

> **一个基于 LangChain Agent + LangGraph Runtime，自研 Harness 能力的跨境电商搜索 Agent。项目重点实现了多 Agent 动态 Fork、持久化 Checkpoint、Cache Breakpoint、结构化长期记忆、工具可靠性治理、Hybrid Retrieval + Pairwise LTR、Grounded RAG、轨迹评测以及 FastAPI/WebSocket 实时事件流。**

当前还不应该宣称已经完整实现：

- 三塔训练；
- CrossEncoder 训练；
- Agent 自进化；
- SFT/RL；
- K8s 生产化；
- 真实四平台实时商品搜索。

---

## 14. 如果只补 5 项，建议补什么

如果以“秋招项目完成度”和“投入产出比”为标准，只继续补 5 项：

1. **✅ 已完成：修复 Agent 300s 超时并完成四平台真实 Demo 回归；**
2. **增加 Semantic Assertion + Silent Drift，补齐 Harness 故事；**
3. **增加 User Tower 个性化召回，真正完成三塔；**
4. **增加 Prompt Version/A-B + Judge，形成最小自进化闭环；**
5. **增加 Docker + 完整 Langfuse Trace/Score，补生产化。**

完成这 5 项以后，不建议继续为了“教程章节覆盖率”无限增加功能。

后续更应该投入到：

- 架构图；
- 核心链路时序图；
- 性能指标；
- Bad Case；
- 关键技术取舍；
- 对照实验；
- 面试讲解材料。

因为到那时，ShopPilot 已经足够作为一个较强的 Agent 工程类秋招项目。

---

## 15. 一句话总结

ShopPilot 当前最明显的特征不是“教程没做完”，而是：

> **Agent 主链路、检索、RAG、Checkpoint、Memory 和执行可靠性已经比较扎实；真正还需要补的是 User Tower、Silent Drift、自进化、训练闭环和生产化基础设施。**

后续应优先提升“稳定性、可评测性、可解释性和面试可讲性”，而不是机械追求教程所有章节 100% 复刻。
