# ShopPilot Agent 轨迹评估问题与优化复盘

## 1. 背景

在 ShopPilot 跨境电商搜索 Agent 中，我们已经实现了两层评估：

1. **Rule-based Trajectory Evaluator**：使用确定性代码检查 Agent 生命周期、工具依赖、重试、终止状态、预算、平台等硬约束。
2. **LLM-as-a-Judge**：使用独立 Judge 模型评估规划质量、工具选择、轨迹效率和最终答案质量。

某次真实历史任务运行后，LLM Judge 给出了以下问题：

- 连续多次重复调用 `item_search`，缺乏去重或聚合逻辑。
- `dispatch_tool` 执行时间过长，影响流程效率。
- 未有效利用 fork 结果，子任务被拒绝后仍重复主路径搜索。

以及以下建议：

- 引入结果缓存或去重机制，避免重复调用相同工具。
- 优化 `dispatch_tool` 的并发策略，减少单点阻塞。
- 在 fork 被拒绝后，应主动回退至主路径并调整搜索策略。

这次优化的重点不是单纯“修改 Judge Prompt 把分数调高”，而是先核对真实 Trace，区分 **Judge 假阳性** 和 **Agent 真问题**，再分别修复评估层和执行层。

---

## 2. 问题一：Judge 将跨平台搜索误判为重复 `item_search`

### 2.1 Judge 的原始判断

Judge 看到主 Agent 连续多次调用：

```text
item_search
item_search
item_search
item_search
```

因此认为存在明显重复搜索。

### 2.2 核对真实 Trace

进一步检查 `trace.json` 中每次 `tool_start` 的参数后发现，这 4 次搜索其实分别对应不同平台：

```text
item_search(platform=amazon)
item_search(platform=shopee)
item_search(platform=aliexpress)
item_search(platform=ebay)
```

因此这不是“完全相同的数据重复搜索”，而是在完成用户要求的跨平台覆盖。

### 2.3 根因

问题并不完全出在 Agent，而是 **Judge Context 信息不足**。

原来的 Judge Context 主要保留：

```text
Step 4: item_search
Step 5: item_search
Step 6: item_search
Step 7: item_search
```

但没有把关键参数一起提供给 Judge，例如：

- `platform`
- `category`
- `query`
- `top_k`

Judge 只能看到工具名相同，很容易把“同工具、不同目标”误判成“重复调用”。

### 2.4 解决方案

#### 方案 A：升级 Judge Context

现在每次工具调用都会尽量携带对应 `tool_start.args`：

```json
{
  "tool": "item_search",
  "status": "success",
  "args": {
    "platform": "amazon",
    "category": "travel_storage",
    "budget_cny": 300,
    "top_k": 20
  }
}
```

Judge 因此能够区分：

```text
item_search(amazon)
item_search(shopee)
item_search(aliexpress)
item_search(ebay)
```

和真正的重复：

```text
item_search(amazon)
item_search(amazon)
item_search(amazon)
```

#### 方案 B：升级 Judge Rubric

Judge 版本升级为：

```text
trajectory_judge_v2
judge_context_v2
```

Prompt 中明确加入原则：

> 相同工具名称本身不代表重复调用。必须比较调用参数和任务目标；不同平台的 `item_search` 属于跨平台覆盖，不应仅因为工具名相同而扣分。

### 2.5 进一步防御：Agent State 级搜索去重

虽然这次 Judge 对 4 个不同平台的判断属于假阳性，但系统仍然增加了真正的状态级去重机制。

在 `item_search` 入口先检查：

```python
state["search_outputs"].get(platform)
```

如果这个平台已经存在有效搜索结果，则直接复用 checkpoint 中的结果，不再重新执行底层检索：

```text
LLM 再次请求 item_search(amazon)
              ↓
检查 search_outputs["amazon"]
              ↓
已有结果
              ↓
直接复用
              ↓
不重新执行 BM25 / Vector / RRF / Rerank
```

同时记录 `tool_idempotency_hit`，来源为：

```text
checkpoint_search_outputs
```

因此现在有两层防重复：

```text
精确调用级去重：idempotency_key
          +
状态级去重：search_outputs[platform]
```

---

## 3. 问题二：`dispatch_tool` 执行约 90 秒

### 3.1 表面现象

Rule-based Evaluator 和 LLM Judge 都观察到：

```text
dispatch_tool duration ≈ 90s
```

这明显拖慢了整个 Agent 运行。

### 3.2 原本容易产生的错误判断

最开始容易怀疑：

- 向量检索太慢；
- BGE embedding 太慢；
- `item_search` 本身太慢；
- 并发数量不足。

但真实 Trace 表明这次 `dispatch_tool` 的 90 秒并不是商品搜索本身造成的。

### 3.3 真实根因：子 Agent 递归 fork 导致自等待

真实轨迹大致如下：

```text
Main Agent
   ↓
dispatch_tool
   ↓
fork Child Agent
   ↓
Child Agent
   ↓
planner
   ↓
category_insight
   ↓
dispatch_tool     ← 子 Agent 再次调用 fork
   ↓
fork_deduplicated(source=inflight)
   ↓
等待等价正在运行的 fork
```

与此同时父 Agent 又在等待子 Agent 返回：

```text
Parent Agent
    ↓ waits for
Child Agent
    ↓ waits for
Equivalent inflight fork
```

最终子 Agent触发：

```text
sub_agent_timeout = 90s
```

因此主 `dispatch_tool` 也几乎正好耗时 90 秒。

本质上，这是一个 **递归 fork + inflight dedup 引起的自等待链**。

---

## 4. 解决 `dispatch_tool` 自等待问题

### 4.1 子 Agent 禁止递归 `dispatch_tool`

原设计更接近“同质 Agent”：

```text
Main Agent
├── planner
├── category_insight
├── item_search
├── price_compare
├── shipping_calc
├── item_picker
├── shopping_summary
└── dispatch_tool

Sub Agent
├── planner
├── category_insight
├── item_search
├── price_compare
├── shipping_calc
├── item_picker
├── shopping_summary
└── dispatch_tool
```

问题是子 Agent 仍然能够继续 fork。

现在将子 Agent 明确调整成 **Leaf Worker（叶子执行器）**：

```text
Sub Agent
├── planner
├── category_insight
└── item_search

禁止：
- dispatch_tool
- 再次 fork 子 Agent
```

代码层通过 `select_tool_names(state)` 根据：

```python
state["is_sub_agent"]
```

隐藏 `dispatch_tool`。

同时 `dispatch_tool` 自身也增加防御性判断：

```python
if runtime.state.get("is_sub_agent"):
    # 直接拒绝递归 fork
```

即使以后某处工具权限配置出现问题，也不会再次产生递归 fork。

这是典型的 **Defense in Depth**。

---

## 5. 子 Agent 职责从“完整购物 Agent”调整为“搜索 Worker”

### 5.1 原来的低效模式

子 Agent 有可能完整执行：

```text
planner
↓
category_insight
↓
item_search
↓
price_compare
↓
shipping_calc
↓
item_picker
↓
shopping_summary
```

但父 Agent 实际主要需要的是：

```text
search_outputs
category insight
```

因此子 Agent 后续的：

- `price_compare`
- `shipping_calc`
- `item_picker`
- `shopping_summary`

很可能与主 Agent 重复。

### 5.2 优化后的职责划分

现在采用：

```text
Sub Agent = Search Worker
Main Agent = Orchestrator + Aggregator
```

子 Agent：

```text
planner
↓
category_insight（需要时）
↓
item_search
↓
返回父 Agent
```

父 Agent：

```text
汇总各平台 search_outputs
↓
price_compare
↓
shipping_calc
↓
item_picker
↓
shopping_summary
```

这样减少了子 Agent 中重复的后处理流程。

---

## 6. 问题三：Fork 失败后的恢复策略不够明确

Judge 提出：

> 未有效利用 fork 结果，子任务被拒绝后仍重复主路径搜索。

这条反馈有一定合理性，但原系统缺少一个更明确的状态约束：**到底哪些平台已经搜索完成，哪些还缺失？**

### 6.1 原来的问题

假设计划需要：

```text
amazon
shopee
aliexpress
ebay
```

原逻辑主要判断：

```python
if state.get("search_outputs"):
    # 已经有搜索结果
```

这意味着只要搜到一个平台：

```text
amazon ✓
```

就可能已经进入下一阶段。

但实际上还缺：

```text
shopee
aliexpress
ebay
```

---

## 7. 引入 Platform Coverage Gate

现在搜索阶段不再只判断：

```text
有没有 search_outputs
```

而是计算：

```text
expected_platforms - searched_platforms
```

例如：

```text
expected_platforms:
amazon
shopee
aliexpress
ebay

searched_platforms:
amazon
shopee

missing_platforms:
aliexpress
ebay
```

只要：

```text
missing_platforms != []
```

就仍处于搜索阶段，不开放：

```text
price_compare
```

只有所有计划平台都完成覆盖：

```text
amazon      ✓
shopee      ✓
aliexpress  ✓
ebay        ✓
```

才进入：

```text
price_compare
↓
shipping_calc
↓
item_picker
↓
shopping_summary
```

这使跨平台任务的状态机更加确定。

---

## 8. Fork 失败后的恢复方式

引入 Coverage Gate 后，fork 恢复可以自然变成“只补缺口”。

例如：

```text
Amazon fork       ✓
Shopee fork       ✓
AliExpress fork   × timeout
Ebay fork         ✓
```

当前状态：

```text
search_outputs:
amazon
shopee
ebay
```

系统可以计算出：

```text
missing_platforms = [aliexpress]
```

主 Agent 只需要补：

```text
item_search(platform=aliexpress)
```

而不是重新搜索：

```text
amazon
shopee
ebay
```

即使模型误调用已经搜索过的平台，也会被前面增加的 State-level Cache 拦截并直接复用。

最终恢复逻辑形成：

```text
fork result
    ↓
merge search_outputs
    ↓
calculate missing platforms
    ↓
只补搜缺失平台
    ↓
已经覆盖的平台 → checkpoint cache reuse
```

---

## 9. 优化跨平台 Fork 策略

### 9.1 原 Prompt 的问题

原 Prompt 只写：

> 跨多个独立平台时，优先并行调用 `dispatch_tool`。

这个描述比较模糊，模型可能生成一个大而泛化的 fork：

```text
dispatch_tool(
  demands="搜索 Amazon、Shopee、AliExpress、eBay"
)
```

这样子 Agent 范围太宽，容易继续规划、继续 fork 或长时间执行。

### 9.2 改成 Scoped Fork

现在明确要求：

> 跨多个平台时，每个平台使用一个带 platform scope 的 `dispatch_tool`。

理想执行方式：

```text
dispatch_tool(platform="amazon")
dispatch_tool(platform="shopee")
dispatch_tool(platform="aliexpress")
dispatch_tool(platform="ebay")
```

如果模型支持并行 Tool Calling，则结构为：

```text
                         Main Agent
                             │
         ┌───────────────────┼───────────────────┐
         ↓                   ↓                   ↓
 Amazon Child          Shopee Child       AliExpress Child       eBay Child
         ↓                   ↓                   ↓                   ↓
 item_search           item_search         item_search           item_search
         └───────────────────┼───────────────────┘
                             ↓
                      merge search_outputs
                             ↓
                       price_compare
                             ↓
                       shipping_calc
                             ↓
                         item_picker
                             ↓
                     shopping_summary
```

每个子 Agent 的边界更加清晰。

---

## 10. 补强 `dispatch_tool` 可观测性

原来 `dispatch_tool` 的 trace 对参数记录不够完整。

Judge 能看到：

```text
dispatch_tool
```

但不容易知道：

```text
platform = ?
category = ?
demands = ?
```

现在 `dispatch_tool` 增加：

```text
tool_start
```

并记录：

```json
{
  "tool_name": "dispatch_tool",
  "args": {
    "platform": "amazon",
    "category": "travel_storage",
    "demands": "..."
  }
}
```

这样后续 Judge 可以进一步判断：

- fork 是否有明确 scope；
- 是否存在过大的子任务；
- 是否重复 fork 同一平台；
- fork 失败后是否合理恢复。

这属于对 **Observability** 的补强。

---

## 11. Judge 缓存版本问题

由于这次不仅修改了 Agent，还修改了：

- Judge Rubric；
- Judge Context；
- 工具参数传递方式；

因此旧的 Judge 评分不能继续视为当前版本的有效结论。

旧版本：

```text
trajectory_judge_v1
judge_context_v1
```

当前版本：

```text
trajectory_judge_v2
judge_context_v2
```

系统现在会检测缓存中的：

```text
judge_version
context_version
```

如果与当前版本不一致：

```text
stale = true
```

历史页面提示用户：

> 当前结果来自旧版 Judge Rubric / Context，建议按新版重新评审。

避免旧评估结果长期污染当前实验。

---

## 12. 优化前后的整体对比

### 优化前

```text
Main Agent
    ↓
dispatch_tool（大范围任务）
    ↓
Child Agent
    ↓
planner
    ↓
category_insight
    ↓
dispatch_tool
    ↓
inflight dedup
    ↓
等待
    ↓
90s timeout
    ↓
Main Agent 恢复
    ↓
item_search × 4
    ↓
price_compare
    ↓
...
```

主要问题：

- 子 Agent 可以递归 fork；
- 出现自等待；
- `dispatch_tool` 被拖到 90 秒；
- fork scope 不明确；
- 平台覆盖状态不明确；
- Judge 看不到工具参数，容易误判重复搜索。

### 优化后

```text
Main Agent
    ↓
按平台 scoped dispatch
    ↓
┌────────────┬────────────┬────────────┬────────────┐
↓            ↓            ↓            ↓
Amazon      Shopee      AliExpress    eBay
Leaf Child  Leaf Child  Leaf Child    Leaf Child
↓            ↓            ↓            ↓
item_search  item_search  item_search  item_search
└────────────┴────────────┴────────────┴────────────┘
                     ↓
             merge search_outputs
                     ↓
             coverage check
                     ↓
          missing platform ?
             ↓           ↓
            Yes          No
             ↓           ↓
        只补缺失平台   price_compare
                         ↓
                    shipping_calc
                         ↓
                     item_picker
                         ↓
                  shopping_summary
```

同时：

```text
重复同平台 item_search
       ↓
checkpoint search_outputs hit
       ↓
直接复用
```

---

## 13. 这次优化体现的工程思路

### 13.1 不盲信 LLM Judge

LLM Judge 的结论不是 Ground Truth。

正确做法是：

```text
Judge 发现问题
    ↓
读取真实 Trace
    ↓
验证证据
    ↓
区分：
- Agent 真问题
- Judge 假阳性
    ↓
分别修复
```

这次“4 次 item_search 是重复搜索”就是典型例子。

---

### 13.2 能用状态机保证的，不只靠 Prompt

如果只修改 Prompt：

> 请不要重复搜索。

模型仍然可能犯错。

因此实际加入：

- `search_outputs[platform]` 状态级缓存；
- `missing_platforms` Coverage Gate；
- 子 Agent 工具权限限制；
- `dispatch_tool` 内部二次保护。

也就是：

```text
Prompt Policy
     +
Runtime Permission
     +
State Machine
     +
Cache / Idempotency
```

共同约束 Agent。

---

### 13.3 Multi-Agent 不等于无限递归 Agent

原来追求“同质子 Agent”，但在实际业务中并不意味着所有层级都应该拥有完整工具权限。

这次最终采用：

```text
Main Agent = Orchestrator
Sub Agent  = Leaf Worker
```

比无限递归 fork 更可控。

---

### 13.4 Evaluation 本身也需要版本管理

LLM-as-a-Judge 的评分会受到：

- Prompt；
- Rubric；
- Context Builder；
- Judge Model；

影响。

因此必须记录：

```text
judge_version
context_version
model
evaluated_at
```

否则不同版本的分数不能公平比较。

这也是后续做 Regression Evaluation 的基础。

---

## 14. 当前验证结果

本轮优化完成后，相关后端测试：

```text
23 passed
```

覆盖：

- Agent architecture；
- 子 Agent leaf 权限；
- 多平台 Coverage Gate；
- `item_search` checkpoint 去重；
- Judge Context 参数传递；
- LLM Judge；
- Rule-based Trajectory Evaluator；
- Pipeline；
- Streaming Runtime。

前端构建：

```text
npm run build

✓ built
```

说明修改没有破坏现有历史详情、评估 UI 和 Agent 主流程。

---

## 15. 后续验证方式

旧任务只能验证 **新版 Judge 是否减少误判**，无法验证已经执行完的旧 Agent 轨迹是否变快。

因此需要分两步测试。

### 测试一：重新 Judge 旧历史

在历史详情点击：

```text
按新版重新评审
```

重点观察 Judge 是否仍然错误认为：

```text
amazon / shopee / aliexpress / ebay
```

四个平台搜索是“重复 item_search”。

### 测试二：重新运行相同购物请求

真正验证 Agent 优化是否生效，应重新提交一次相同 Query。

重点观察新 Trace 是否从：

```text
dispatch_tool
↓
child dispatch_tool
↓
90s timeout
↓
main item_search × 4
```

变成：

```text
scoped dispatch_tool
↓
leaf child item_search
↓
merge search_outputs
↓
coverage gate
↓
只补缺失平台
↓
price_compare
↓
shipping_calc
↓
item_picker
↓
shopping_summary
```

如果新任务不再出现子 Agent 递归 fork，同时 `dispatch_tool` 耗时显著下降，就说明本次优化真正解决了执行层问题。

---

## 16. 面试时可以如何描述

可以概括为：

> 我在 Agent 运行链路上实现了 Rule-based Trajectory Evaluator 和 LLM-as-a-Judge。Judge 曾发现重复搜索、低效 fork 和恢复策略问题，但我没有直接把 Judge 输出当作 Ground Truth，而是回溯 trace.json 验证实际调用参数。最终发现跨平台 item_search 属于 Judge 假阳性，而 dispatch_tool 的 90 秒延迟是真问题，根因是子 Agent 递归 fork 后与 inflight dedup 形成自等待。我随后把子 Agent 改为 leaf worker，禁止递归 dispatch，引入平台 coverage gate 和 checkpoint search cache，使 fork 失败后只补搜缺失平台，同时升级 Judge Context，让评审模型能够看到工具参数而不是只看工具名。

这段经历可以体现：

- Agent Observability；
- Trajectory Evaluation；
- LLM-as-a-Judge；
- Judge False Positive 分析；
- Multi-Agent Orchestration；
- State Machine；
- Idempotency / Cache；
- Failure Recovery；
- Evaluation Versioning；
- AI 应用工程化调试能力。

---

## 17. 最终结论

本次问题不是简单的“Agent 调用了太多工具”，而是三个不同层面的问题：

| 问题 | 类型 | 最终处理 |
|---|---|---|
| 四次 `item_search` 被认为重复 | Judge 假阳性 | Judge Context 增加参数 + Judge Rubric v2 |
| 真正重复同平台搜索的风险 | Agent Reliability | `search_outputs[platform]` 状态级缓存 |
| `dispatch_tool` 约 90 秒 | Agent 架构问题 | 禁止子 Agent 递归 fork，改为 Leaf Worker |
| fork 后恢复不明确 | 状态机问题 | Platform Coverage Gate，只补缺失平台 |
| fork 任务范围过大 | Orchestration 问题 | 每个平台一个 scoped dispatch |
| Judge 无法理解 dispatch scope | Observability 问题 | `dispatch_tool` 增加 `tool_start(args)` |
| 旧 Judge 结果与新规则不一致 | Evaluation 管理问题 | Judge/Context 版本检查 + stale 标记 |

最终目标不是让 Judge 给出更高分，而是让：

```text
Agent 执行更正确
+ Agent 恢复更稳定
+ Trace 更可解释
+ Judge 评价更可信
```

这也为下一阶段的 **Evaluation Dataset + Regression Evaluation** 打下了基础。
