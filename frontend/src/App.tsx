import { useEffect, useMemo, useRef, useState } from "react";

import EvaluationPanel, { type TrajectoryEvaluation } from "./EvaluationPanel";
import MarkdownContent from "./MarkdownContent";

const DEMO_USER_ID = "demo-user";

type AguiEvent = {
  event: string;
  message: string;
  data: Record<string, unknown>;
  timestamp?: string;
};

type HistoryPlan = {
  category?: string | null;
  category_key?: string | null;
  budget_cny?: number | null;
  platforms?: string[];
  hard_constraints?: string[];
  soft_preferences?: string[];
};

type HistoryItem = {
  thread_id: string;
  user_id?: string | null;
  query: string;
  final_preview: string;
  completed_at: string;
  plan: HistoryPlan;
  files: string[];
};

type HistoryDetail = {
  thread_id: string;
  user_id?: string | null;
  query: string;
  completed_at: string;
  plan: HistoryPlan;
  final_answer: string;
  files: string[];
  events: AguiEvent[];
  evaluation: TrajectoryEvaluation;
};

const labels: Record<string, string> = {
  session_created: "会话",
  assistant_call: "模型决策",
  assistant_token: "Token 流",
  stage: "阶段",
  fork: "子 Agent",
  fork_queued: "Fork 排队",
  fork_dequeued: "Fork 启动",
  fork_deduplicated: "Fork 复用",
  fork_rejected: "Fork 降级",
  context_compaction: "上下文压缩",
  context_compaction_failed: "压缩降级",
  prompt_cache: "Prompt Cache",
  memory_retrieved: "记忆召回",
  memory_updated: "记忆更新",
  retrieval_search: "语义检索 / LTR",
  knowledge_retrieval: "品类知识检索",
  knowledge_synthesis: "证据引用生成",
  tool_start: "工具开始",
  tool_end: "工具完成",
  tool_attempt: "工具尝试",
  tool_retry: "工具重试",
  tool_success: "工具成功",
  tool_failure: "工具降级",
  tool_idempotency_hit: "幂等复用",
  tool_circuit_open: "熔断打开",
  tool_circuit_rejected: "熔断拒绝",
  tool_circuit_recovered: "熔断恢复",
  task_result: "完成",
  task_cancelled: "已取消",
  error: "错误",
};

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function formatBudget(value: number | null | undefined): string | null {
  if (value == null || Number.isNaN(value)) return null;
  return `预算 ¥${value.toFixed(0)}`;
}

const stageLabels: Record<string, string> = {
  prepare: "准备",
  resume: "恢复",
  think: "决策",
  act: "执行",
  observe: "观察结果",
  reflect: "总结",
};

const timelineGroupMeta: Record<string, { title: string; description: string }> = {
  session: { title: "会话准备", description: "创建会话、恢复状态并召回长期偏好" },
  think: { title: "Agent 决策", description: "模型分析当前状态，决定下一步动作" },
  act: { title: "工具与并行检索", description: "调用商品检索、知识检索、比价和子 Agent" },
  context: { title: "上下文与可靠性", description: "压缩上下文、复用结果或处理瞬时故障" },
  finish: { title: "总结与完成", description: "整理结果、更新记忆并返回最终答案" },
  error: { title: "异常与取消", description: "任务异常、取消或可恢复的降级" },
};

function eventGroupId(event: AguiEvent): string {
  if (event.event === "session_created") return "session";
  if (event.event === "stage") {
    const stage = String(event.data?.stage ?? "");
    if (stage === "think") return "think";
    if (stage === "reflect") return "finish";
    return "session";
  }
  if (["assistant_call"].includes(event.event)) return "think";
  if (["task_result", "memory_updated"].includes(event.event)) return "finish";
  if (["error", "task_cancelled"].includes(event.event)) return "error";
  if (["context_compaction", "context_compaction_failed", "prompt_cache"].includes(event.event)) {
    return "context";
  }
  return "act";
}

function actorLabel(value: unknown): string {
  const actor = String(value ?? "main");
  if (actor === "main") return "主 Agent";
  return `子 Agent · ${actor.slice(0, 8)}`;
}

function eventDetail(event: AguiEvent): string {
  const data = event.data ?? {};
  const details: string[] = [];
  if (event.event === "stage" && data.stage) details.push(`阶段：${stageLabels[String(data.stage)] ?? String(data.stage)}`);
  if (data.tool_calls && Array.isArray(data.tool_calls) && data.tool_calls.length > 0) {
    details.push(`工具：${data.tool_calls.map(String).join("、")}`);
  }
  if (data.tool_name) details.push(`工具：${String(data.tool_name)}`);
  if (data.platform) details.push(`平台：${String(data.platform)}`);
  if (data.category_key) details.push(`品类：${String(data.category_key)}`);
  if (data.returned_count != null) details.push(`返回 ${String(data.returned_count)} 条候选`);
  if (data.claim_count != null) details.push(`${String(data.claim_count)} 条依据`);
  if (data.duration_ms != null) details.push(`${String(data.duration_ms)} ms`);
  if (data.wait_ms != null) details.push(`排队 ${String(data.wait_ms)} ms`);
  if (data.attempt != null && data.max_attempts != null) details.push(`第 ${String(data.attempt)}/${String(data.max_attempts)} 次`);
  if (data.count != null) details.push(`${String(data.count)} 条记忆`);
  if (data.reason) details.push(`原因：${String(data.reason)}`);
  return details.join(" · ");
}

function hasEventData(data: Record<string, unknown>): boolean {
  return Object.keys(data).some((key) => key !== "actor_thread_id");
}

function EventTimeline({ events, emptyText }: { events: AguiEvent[]; emptyText: string }) {
  const groups = useMemo(() => {
    const grouped = new Map<string, AguiEvent[]>();
    events.forEach((event) => {
      const id = eventGroupId(event);
      grouped.set(id, [...(grouped.get(id) ?? []), event]);
    });
    return Array.from(grouped.entries()).map(([id, groupEvents]) => ({ id, events: groupEvents }));
  }, [events]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const latest = groups.length > 0 ? groups[groups.length - 1].id : undefined;
    if (!latest) return;
    setExpanded((previous) => (previous[latest] ? previous : { ...previous, [latest]: true }));
  }, [groups]);

  function toggleGroup(id: string) {
    setExpanded((previous) => ({ ...previous, [id]: !previous[id] }));
  }

  function setAllGroups(next: boolean) {
    setExpanded(Object.fromEntries(groups.map((group) => [group.id, next])));
  }

  return (
    <div className="timeline">
      {events.length === 0 && <p className="muted">{emptyText}</p>}
      {events.length > 0 && (
        <div className="timeline-toolbar">
          <span>已将 {events.length} 条事件整理为 {groups.length} 个阶段</span>
          <div>
            <button className="timeline-action" onClick={() => setAllGroups(true)} type="button">全部展开</button>
            <button className="timeline-action" onClick={() => setAllGroups(false)} type="button">全部折叠</button>
          </div>
        </div>
      )}
      {groups.map((group, groupIndex) => {
        const meta = timelineGroupMeta[group.id] ?? { title: group.id, description: "执行事件" };
        const isExpanded = Boolean(expanded[group.id]);
        const isLatest = groupIndex === groups.length - 1;
        return (
          <section className={`timeline-group ${isLatest ? "timeline-group-latest" : ""}`} key={group.id}>
            <button
              className="timeline-group-toggle"
              onClick={() => toggleGroup(group.id)}
              type="button"
              aria-expanded={isExpanded}
            >
              <span className="timeline-step">{groupIndex + 1}</span>
              <span className="timeline-group-copy">
                <strong>{meta.title}</strong>
                <small>{meta.description}</small>
              </span>
              <span className="timeline-group-count">{group.events.length} 条</span>
              <span className="timeline-chevron" aria-hidden="true">{isExpanded ? "−" : "+"}</span>
            </button>
            {isExpanded && (
              <div className="timeline-group-events">
                {group.events.map((event, index) => {
                  const data = event.data ?? {};
                  const detail = eventDetail(event);
                  return (
                    <article className={`event event-${event.event}`} key={`${event.timestamp ?? "event"}-${index}`}>
                      <span className="event-marker" aria-hidden="true" />
                      <div className="event-content">
                        <div className="event-heading">
                          <span className="badge">{labels[event.event] ?? event.event}</span>
                          <strong>{event.message}</strong>
                        </div>
                        <small>
                          {actorLabel(data.actor_thread_id ?? data.sub_thread_id)}
                          {detail ? ` · ${detail}` : ""}
                          {event.timestamp ? ` · ${formatDate(event.timestamp)}` : ""}
                        </small>
                        {hasEventData(data) && (
                          <details className="event-details">
                            <summary>查看原始数据</summary>
                            <pre>{JSON.stringify(data, null, 2)}</pre>
                          </details>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}

function HistoryList({
  items,
  loading,
  error,
  onOpen,
}: {
  items: HistoryItem[];
  loading: boolean;
  error: string;
  onOpen: (threadId: string) => void;
}) {
  return (
    <section className="panel history-panel">
      <div className="panel-heading">
        <div>
          <h2>历史搜索</h2>
          <p className="panel-description">已完成的搜索保存在后端，刷新页面后仍可查看。</p>
        </div>
        {items.length > 0 && <span className="history-count">{items.length} 条</span>}
      </div>

      {loading && <p className="muted">正在加载历史搜索…</p>}
      {!loading && error && <p className="history-error">{error}</p>}
      {!loading && !error && items.length === 0 && (
        <p className="muted">还没有历史搜索。完成一次搜索后会出现在这里。</p>
      )}

      {!loading && !error && items.length > 0 && (
        <div className="history-list">
          {items.map((item) => {
            const budget = formatBudget(item.plan?.budget_cny);
            return (
              <button
                className="history-item"
                key={item.thread_id}
                onClick={() => onOpen(item.thread_id)}
                type="button"
              >
                <span className="history-copy">
                  <strong className="history-title">{item.query || "未命名搜索"}</strong>
                  <span className="history-meta">
                    {formatDate(item.completed_at)}
                    {item.plan?.category ? ` · ${item.plan.category}` : ""}
                    {budget ? ` · ${budget}` : ""}
                  </span>
                </span>
                <span className="history-arrow" aria-hidden="true">›</span>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function SearchPage({ onOpenHistory }: { onOpenHistory: (threadId: string) => void }) {
  const [query, setQuery] = useState(
    "想买便宜又抗造的旅行收纳三件套，预算300，不要塑料，偏好小众",
  );
  const [threadId, setThreadId] = useState<string | null>(null);
  const [events, setEvents] = useState<AguiEvent[]>([]);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [finalAnswer, setFinalAnswer] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const streamMessageIdRef = useRef<string | null>(null);

  async function loadHistory() {
    try {
      setHistoryError("");
      const response = await fetch(`/api/history?user_id=${encodeURIComponent(DEMO_USER_ID)}&limit=20`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const body = (await response.json()) as { items?: HistoryItem[] };
      setHistoryItems(body.items ?? []);
    } catch (error) {
      setHistoryError(`历史搜索加载失败：${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    void loadHistory();
    return () => wsRef.current?.close();
  }, []);

  function connect(tid: string) {
    wsRef.current?.close();
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws/${tid}`);
    wsRef.current = ws;
    ws.onmessage = (message) => {
      if (message.data === "pong") return;
      const payload = JSON.parse(message.data) as AguiEvent;
      if (!payload.event) return;
      if (payload.event === "assistant_token") {
        const delta = String(payload.data.delta ?? "");
        const messageId = String(payload.data.message_id ?? "anonymous");
        if (streamMessageIdRef.current !== messageId) {
          streamMessageIdRef.current = messageId;
          setStreamingAnswer(delta);
        } else {
          setStreamingAnswer((previous) => previous + delta);
        }
        return;
      }
      setEvents((previous) => [...previous, payload]);
      if (payload.event === "task_result") {
        setFinalAnswer(String(payload.data.final_answer ?? ""));
        setFiles((payload.data.files as string[] | undefined) ?? []);
        setStreamingAnswer("");
        streamMessageIdRef.current = null;
        setRunning(false);
        void loadHistory();
      }
      if (payload.event === "error" || payload.event === "task_cancelled") {
        setRunning(false);
      }
    };
  }

  async function startTask() {
    if (!query.trim()) return;
    setRunning(true);
    setEvents([]);
    setStreamingAnswer("");
    streamMessageIdRef.current = null;
    setFinalAnswer("");
    setFiles([]);
    const response = await fetch("/api/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, user_id: DEMO_USER_ID }),
    });
    if (!response.ok) {
      setRunning(false);
      return;
    }
    const body = (await response.json()) as { thread_id: string };
    setThreadId(body.thread_id);
    connect(body.thread_id);
  }

  async function cancelTask() {
    if (!threadId) return;
    await fetch(`/api/task/${threadId}/cancel`, { method: "POST" });
    setRunning(false);
  }

  return (
    <main>
      <header>
        <p className="eyebrow">ShopPilot · LangGraph AgentLoop</p>
        <h1>跨境电商搜索 Agent</h1>
        <p className="subtitle">可视化主循环、并行 fork、工具调用、比价与到手价精排。</p>
      </header>

      <section className="composer">
        <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
        <div className="actions">
          <button onClick={running ? cancelTask : startTask}>
            {running ? "取消任务" : "开始搜索"}
          </button>
          {threadId && <code>{threadId}</code>}
        </div>
      </section>

      <HistoryList
        items={historyItems}
        loading={historyLoading}
        error={historyError}
        onOpen={onOpenHistory}
      />

      <section className="panel">
        <h2>Agent 事件流</h2>
        <EventTimeline
          events={events}
          emptyText="任务开始后会显示 Think / Act / Observe / Reflect 与工具事件。"
        />
      </section>

      {(finalAnswer || streamingAnswer) && (
        <section className="panel result">
          <h2>{finalAnswer ? "最终购物清单" : "模型实时输出"}</h2>
          {finalAnswer ? (
            <MarkdownContent source={finalAnswer} />
          ) : (
            <div className="streaming-output" aria-live="polite">
              <MarkdownContent source={streamingAnswer} />
              <span className="streaming-caret" aria-hidden="true" />
            </div>
          )}
          <div className="files">
            {files.map((filename) => (
              <a key={filename} href={`/api/files/${threadId}/${filename}`}>
                下载 {filename}
              </a>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

function HistoryDetailPage({
  threadId,
  onBack,
}: {
  threadId: string;
  onBack: () => void;
}) {
  const [detail, setDetail] = useState<HistoryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [judgeRunning, setJudgeRunning] = useState(false);
  const [judgeError, setJudgeError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        setLoading(true);
        setError("");
        const response = await fetch(
          `/api/history/${encodeURIComponent(threadId)}?user_id=${encodeURIComponent(DEMO_USER_ID)}`,
          { signal: controller.signal },
        );
        if (!response.ok) throw new Error(response.status === 404 ? "历史任务不存在" : `HTTP ${response.status}`);
        setDetail((await response.json()) as HistoryDetail);
      } catch (loadError) {
        if (controller.signal.aborted) return;
        setError(loadError instanceof Error ? loadError.message : String(loadError));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [threadId]);

  async function runJudge(force: boolean) {
    try {
      setJudgeRunning(true);
      setJudgeError("");
      const response = await fetch(
        `/api/history/${encodeURIComponent(threadId)}/judge?user_id=${encodeURIComponent(DEMO_USER_ID)}&force=${force ? "true" : "false"}`,
        { method: "POST" },
      );
      const body = (await response.json()) as {
        evaluation?: TrajectoryEvaluation;
        detail?: string;
      };
      if (!response.ok || !body.evaluation) {
        throw new Error(body.detail || `HTTP ${response.status}`);
      }
      setDetail((previous) => previous ? { ...previous, evaluation: body.evaluation! } : previous);
    } catch (judgeLoadError) {
      setJudgeError(judgeLoadError instanceof Error ? judgeLoadError.message : String(judgeLoadError));
    } finally {
      setJudgeRunning(false);
    }
  }

  return (
    <main>
      <header className="history-detail-header">
        <button className="secondary-button back-button" onClick={onBack} type="button">
          ← 返回搜索
        </button>
        <p className="eyebrow">ShopPilot · History</p>
        <h1>历史搜索</h1>
        <p className="subtitle">查看当时的搜索条件、Agent 执行轨迹和最终购物清单。</p>
      </header>

      {loading && <section className="panel"><p className="muted">正在加载历史任务…</p></section>}
      {!loading && error && (
        <section className="panel history-empty-state">
          <h2>无法打开历史任务</h2>
          <p>{error}</p>
          <button onClick={onBack} type="button">返回搜索</button>
        </section>
      )}

      {!loading && detail && (
        <>
          <section className="panel history-summary">
            <div className="panel-heading">
              <div>
                <p className="history-detail-label">原始搜索</p>
                <h2 className="history-query">{detail.query}</h2>
              </div>
              <span className="history-status">已完成</span>
            </div>
            <div className="history-facts">
              <span>{formatDate(detail.completed_at)}</span>
              {detail.plan?.category && <span>品类：{detail.plan.category}</span>}
              {formatBudget(detail.plan?.budget_cny) && <span>{formatBudget(detail.plan?.budget_cny)}</span>}
              <span className="history-thread">任务 ID：{detail.thread_id}</span>
            </div>
          </section>

          <EvaluationPanel
            evaluation={detail.evaluation}
            onRunJudge={runJudge}
            judgeRunning={judgeRunning}
            judgeError={judgeError}
          />

          <section className="panel result">
            <h2>最终购物清单</h2>
            {detail.final_answer ? (
              <MarkdownContent source={detail.final_answer} />
            ) : (
              <p className="muted">该历史任务没有保存最终文本。</p>
            )}
            <div className="files">
              {detail.files.map((filename) => (
                <a key={filename} href={`/api/files/${detail.thread_id}/${filename}`}>
                  下载 {filename}
                </a>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>历史 Agent 事件流</h2>
                <p className="panel-description">来自该任务完成时持久化的 trace.json。</p>
              </div>
              <span className="history-count">{detail.events.length} 条</span>
            </div>
            <EventTimeline events={detail.events} emptyText="该历史任务没有保存事件流。" />
          </section>
        </>
      )}
    </main>
  );
}

export default function App() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const handlePopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function navigate(nextPath: string) {
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setPath(nextPath);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const detailMatch = path.match(/^\/history\/([^/]+)$/);
  if (detailMatch) {
    return (
      <HistoryDetailPage
        threadId={decodeURIComponent(detailMatch[1])}
        onBack={() => navigate("/")}
      />
    );
  }

  return <SearchPage onOpenHistory={(threadId) => navigate(`/history/${encodeURIComponent(threadId)}`)} />;
}
