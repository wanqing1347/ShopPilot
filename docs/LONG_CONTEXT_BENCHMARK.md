# ShopPilot Long-Context Benchmark

这个 benchmark 用于验证 Cache Breakpoint 上下文治理的两个量化目标：

1. 10~20 轮长上下文下，provider 实际报告的单轮 `input_tokens` 不超过 30,000。
2. 去除前几轮 warmup 后，Prompt Cache 的 `cache_read_tokens / input_tokens` 稳态命中率达到 80% 或以上。

它不会用字符数假装 token，也不会在 provider 不返回缓存指标时把结果写成 0%。缺少真实缓存元数据时，结果为 `UNSUPPORTED`。

## Benchmark 场景

每一轮固定追加一组接近 Agent Tool Calling 的历史：

```text
HumanMessage
  -> AIMessage(item_search tool_call)
  -> ToolMessage(约 9k chars 商品 JSON)
  -> benchmark model response
```

这批历史经过生产代码里的 `CacheBreakpointMiddleware`，因此会真实覆盖：

- 安全 breakpoint，不拆开 `AIMessage.tool_calls` / `ToolMessage`。
- 最近工具轮完整保留。
- 旧历史增量摘要化。
- 大 ToolMessage 的结构化 JSON 压缩。
- `cache_epoch` 推进。
- 可选稳定 `prompt_cache_key`。
- provider 返回的 `input_tokens / cache_read_tokens / cache_creation_tokens`。

benchmark 不是 Fake Token 估算。真实运行时，最终 token 与 cache 结论完全来自当前 `LLM_*` 配置所连接的模型服务。

## 运行

先配置和普通 ShopPilot Agent 相同的模型环境变量：

```env
LLM_BASE_URL=https://...
LLM_API_KEY=...
LLM_MODEL_NAME=...
```

默认跑 16 轮：

```powershell
.venv312\Scripts\python.exe scripts/benchmark_long_context.py
```

严格验证“30k + 80%”：

```powershell
.venv312\Scripts\python.exe scripts/benchmark_long_context.py `
  --rounds 16 `
  --warmup-rounds 3 `
  --max-input-tokens 30000 `
  --min-cache-hit-ratio 0.80 `
  --strict
```

如果当前 provider 明确支持 ShopPilot 传入的 `prompt_cache_key`，可以额外打开：

```powershell
.venv312\Scripts\python.exe scripts/benchmark_long_context.py `
  --rounds 16 `
  --prompt-cache-key `
  --strict
```

通用 OpenAI-compatible provider 不确定是否接受该参数时，不要加 `--prompt-cache-key`。即使不显式传 key，支持自动 Prefix Cache 的 provider 仍然可以利用同一 `cache_epoch` 内稳定的 prompt 前缀。

## 输出

默认生成到：

```text
output/benchmarks/
  long-context-<model>-<timestamp>.json
  long-context-<model>-<timestamp>.md
```

每轮记录：

- `input_tokens`
- `cache_read_tokens`
- `cache_creation_tokens`
- 单轮 cache hit ratio
- `cache_epoch`
- `summary_until`
- `breakpoint_index`
- 原始/模型消息数
- `original_chars / model_chars / saved_chars`
- 被压缩 ToolMessage 数量

汇总记录：

- 最大单轮 input tokens
- 全程 cache hit ratio
- 去除 warmup 后的 steady-state cache hit ratio
- 观测到的 cache epochs
- 30k token 门槛结果
- 80% cache hit 门槛结果
- 最终 `PASS / FAIL / UNSUPPORTED`

## 判定规则

### Token budget

只有全部轮次都拿到 provider 的 `input_tokens` 时才做结论：

```text
max(input_tokens_per_round) <= 30000
```

如果 provider 没有提供输入 token usage，则该项为 `UNSUPPORTED`。

### Prompt Cache

默认前三轮作为 warmup，不参与稳态判定。剩余轮次计算：

```text
steady_cache_hit_ratio
= sum(cache_read_tokens) / sum(input_tokens)
```

只有所有稳态轮次都能确认缓存元数据时才做 80% 判定。

这意味着：

- `PASS`：provider 真实数据证明达到门槛。
- `FAIL`：provider 真实数据证明没有达到门槛。
- `UNSUPPORTED`：provider 没暴露足够 usage/cache 元数据，无法证明。

## CI / 单元测试

测试文件：

```text
tests/test_long_context_benchmark.py
```

CI 不调用真实外部 LLM，而使用可控 usage 的 fake model 验证：

- 12 轮 harness 能实际穿过 `CacheBreakpointMiddleware`。
- 至少发生一次 `cache_epoch` 推进。
- 压缩后 `saved_chars > 0`。
- 30k / 80% 的 PASS 判定正确。
- 缺少 provider cache metadata 时返回 `UNSUPPORTED`。
- cache hit 不到 80% 时返回 `FAIL`。
- benchmark 严格限制为 10~20 轮。

运行：

```powershell
.venv312\Scripts\python.exe -m pytest tests/test_long_context_benchmark.py -q
```

## 简历数据怎么写

只有在真实 provider 上执行 `--strict` 并拿到 `PASS` 报告后，才建议写：

> 10+ 轮长对话单轮 Prompt 控制在 30k tokens 内，稳态 Prompt Cache 命中率达到 80%+。

建议保留对应 JSON/Markdown 报告作为面试追问时的实验依据，并注明模型、轮数、warmup 轮数和 benchmark 日期。
