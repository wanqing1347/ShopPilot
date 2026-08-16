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

type MemoryItem = {
  id: string;
  kind: "preference" | "blacklist" | "history";
  scope: "global" | "category";
  content: string;
  category?: string | null;
  status: "active" | "superseded" | "conflicted";
  confidence: number;
  mention_count: number;
  updated_at: string;
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

function actorLabel(value: unknown, rootThreadId?: string | null): string {
  const actor = String(value ?? "main");
  // The backend uses the real root thread UUID for main-agent events. Only
  // treating the literal string "main" as the root mislabels every main event
  // as a sub-agent in the execution summary.
  if (actor === "main" || (rootThreadId && actor === rootThreadId)) return "主 Agent";
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

function processStatus(events: AguiEvent[], running: boolean): string {
  if (!running) return "搜索完成，推荐结果已生成";
  const latest = events[events.length - 1];
  if (!latest) return "Agent 正在理解你的购物需求";
  const tools = eventToolNames(latest);
  if (latest.event === "knowledge_retrieval" || tools.includes("category_insight")) return "正在补充品类信息";
  if (latest.event === "retrieval_search" || tools.includes("item_search")) {
    const platform = latest.data?.platform ? String(latest.data.platform) : "";
    return platform ? `正在 ${platform} 搜索商品` : "正在搜索平台商品";
  }
  if (tools.some((tool) => ["price_compare", "shipping_calc", "item_picker"].includes(tool))) return "正在筛选并比较商品";
  if (tools.includes("shopping_summary")) return "正在整理推荐结果";
  if (tools.includes("planner")) return "正在理解你的购物需求";
  return "Agent 正在处理当前任务";
}

function hasEventData(data: Record<string, unknown>): boolean {
  return Object.keys(data).some((key) => key !== "actor_thread_id");
}

type ProgressStepId = "understand" | "insight" | "search" | "compare" | "recommend";

type ProgressStep = {
  id: ProgressStepId;
  title: string;
  description: string;
  events: AguiEvent[];
};

const progressStepMeta: Record<ProgressStepId, Omit<ProgressStep, "events">> = {
  understand: { id: "understand", title: "理解你的需求", description: "识别品类、预算、平台和偏好" },
  insight: { id: "insight", title: "补充品类信息", description: "读取与当前品类相关的知识依据" },
  search: { id: "search", title: "搜索平台商品", description: "在指定平台查找并召回匹配商品" },
  compare: { id: "compare", title: "筛选并比较", description: "比较价格、物流和预算约束" },
  recommend: { id: "recommend", title: "整理推荐", description: "挑选候选并生成最终购物清单" },
};

const progressStepOrder: ProgressStepId[] = ["understand", "insight", "search", "compare", "recommend"];

function eventToolNames(event: AguiEvent): string[] {
  const names: string[] = [];
  if (event.data?.tool_name) names.push(String(event.data.tool_name));
  if (Array.isArray(event.data?.tool_calls)) names.push(...event.data.tool_calls.map(String));
  return names;
}

function progressStepId(event: AguiEvent): ProgressStepId {
  const tools = eventToolNames(event);
  if (
    event.event === "task_result" ||
    event.event === "memory_updated" ||
    event.data?.stage === "reflect" ||
    tools.includes("shopping_summary")
  ) return "recommend";
  if (tools.some((tool) => ["price_compare", "shipping_calc", "item_picker"].includes(tool))) return "compare";
  if (event.event === "retrieval_search" || event.event.startsWith("fork") || tools.includes("item_search")) return "search";
  if (event.event === "knowledge_retrieval" || tools.includes("category_insight")) return "insight";
  return "understand";
}

function progressSignal(event: AguiEvent): string | null {
  const tools = eventToolNames(event);
  if (event.event === "retrieval_search" || event.event === "knowledge_retrieval") return event.message;
  if (event.event === "memory_retrieved") return "已读取相关长期偏好";
  if (event.event === "fork") return "已派发并行 Agent 处理平台搜索";
  if (event.event === "task_result") return "推荐结果已生成";
  if (event.event === "tool_end") {
    const labelsByTool: Record<string, string> = {
      planner: "已完成需求解析",
      category_insight: "已完成品类信息检索",
      item_search: "已完成商品搜索",
      price_compare: "已完成价格比较",
      shipping_calc: "已完成物流估算",
      item_picker: "已完成候选筛选",
      shopping_summary: "已完成购物清单整理",
    };
    return labelsByTool[tools[0]] ?? null;
  }
  return null;
}

function progressStatus(step: ProgressStep, activeStepId: ProgressStepId | undefined, running: boolean): "done" | "active" | "pending" {
  if (!step.events.length) return "pending";
  if (running && step.id === activeStepId) return "active";
  return "done";
}

function progressStatusLabel(status: "done" | "active" | "pending"): string {
  if (status === "active") return "进行中";
  if (status === "done") return "已完成";
  return "待处理";
}

function EventTimeline({
  events,
  emptyText,
  running = false,
  rootThreadId,
}: {
  events: AguiEvent[];
  emptyText: string;
  running?: boolean;
  rootThreadId?: string | null;
}) {
  const steps = useMemo<ProgressStep[]>(() => {
    const grouped = Object.fromEntries(
      progressStepOrder.map((id) => [id, { ...progressStepMeta[id], events: [] as AguiEvent[] }]),
    ) as Record<ProgressStepId, ProgressStep>;
    events.forEach((event) => grouped[progressStepId(event)].events.push(event));
    return progressStepOrder.map((id) => grouped[id]);
  }, [events]);
  const [expandedStep, setExpandedStep] = useState<ProgressStepId | null>(null);
  const latestEvent = events[events.length - 1];
  const activeStepId = latestEvent ? progressStepId(latestEvent) : undefined;

  useEffect(() => {
    if (running && activeStepId) setExpandedStep(activeStepId);
  }, [activeStepId, running]);

  if (events.length === 0) return <p className="muted">{emptyText}</p>;

  const completedCount = steps.filter((step) => progressStatus(step, activeStepId, running) === "done").length;

  return (
    <div className="agent-progress">
      <div className="agent-progress-summary">
        <div>
          <span className="agent-progress-label">Agent 工作进度</span>
          <strong>{running ? progressStepMeta[activeStepId ?? "understand"].title : "推荐结果已准备好"}</strong>
        </div>
        <span className="agent-progress-count">{completedCount}/{steps.length} 项完成</span>
      </div>
      <div className="agent-progress-steps">
        {steps.map((step, index) => {
          const status = progressStatus(step, activeStepId, running);
          const signals = step.events
            .map((event, eventIndex) => ({ event, eventIndex, text: progressSignal(event) }))
            .filter((item): item is { event: AguiEvent; eventIndex: number; text: string } => Boolean(item.text))
            .filter((item, itemIndex, items) => items.findIndex((candidate) => candidate.text === item.text) === itemIndex)
            .slice(-5);
          const isExpanded = expandedStep === step.id;
          return (
            <div className={`agent-progress-step progress-status-${status}`} key={step.id}>
              <button
                className="agent-progress-step-toggle"
                onClick={() => setExpandedStep(isExpanded ? null : step.id)}
                type="button"
                aria-expanded={isExpanded}
              >
                <span className="progress-index">{status === "done" ? "✓" : index + 1}</span>
                <span className="progress-step-copy">
                  <strong>{step.title}</strong>
                  <small>{step.description}</small>
                </span>
                <span className="progress-step-status">{progressStatusLabel(status)}</span>
                <span className="progress-step-chevron" aria-hidden="true">{isExpanded ? "⌃" : "⌄"}</span>
              </button>
              {isExpanded && (
                <div className="agent-progress-details">
                  {signals.length === 0 && <p className="muted">Agent 正在准备这一项工作…</p>}
                  {signals.map(({ event, eventIndex, text }) => {
                    const data = event.data ?? {};
                    return (
                      <details className="progress-detail" key={`${event.timestamp ?? "event"}-${eventIndex}`}>
                        <summary>{text}</summary>
                        <small>{actorLabel(data.actor_thread_id ?? data.sub_thread_id, rootThreadId)}{eventDetail(event) ? ` · ${eventDetail(event)}` : ""}</small>
                        {hasEventData(data) && <pre>{JSON.stringify(data, null, 2)}</pre>}
                      </details>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
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

type HistoryGroup = {
  label: string;
  items: HistoryItem[];
};

function historyGroupLabel(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "更早";
  const today = new Date();
  const startOfToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  const startOfDate = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const dayDistance = Math.floor((startOfToday - startOfDate) / 86400000);
  if (dayDistance <= 0) return "今天";
  if (dayDistance === 1) return "昨天";
  if (dayDistance <= 7) return "最近 7 天";
  return "更早";
}

function SessionSidebar({
  items,
  loading,
  error,
  collapsed,
  activeView,
  activeThreadId,
  memoryCount,
  onNewSession,
  onOpenHistory,
  onOpenMemory,
  onShowHistory,
  onToggle,
}: {
  items: HistoryItem[];
  loading: boolean;
  error: string;
  collapsed: boolean;
  activeView: "new" | "history" | "memory";
  activeThreadId: string | null;
  memoryCount: number;
  onNewSession: () => void;
  onOpenHistory: (threadId: string) => void;
  onOpenMemory: () => void;
  onShowHistory: () => void;
  onToggle: () => void;
}) {
  const [historyQuery, setHistoryQuery] = useState("");
  const filteredHistory = items.filter((item) => {
    const normalizedQuery = historyQuery.trim().toLowerCase();
    if (!normalizedQuery) return true;
    return `${item.query} ${item.plan?.category ?? ""}`.toLowerCase().includes(normalizedQuery);
  });
  const historyGroups = useMemo<HistoryGroup[]>(() => {
    const order = ["今天", "昨天", "最近 7 天", "更早"];
    return order
      .map((label) => ({ label, items: filteredHistory.filter((item) => historyGroupLabel(item.completed_at) === label) }))
      .filter((group) => group.items.length > 0);
  }, [filteredHistory]);

  return (
    <aside className={`session-sidebar ${collapsed ? "session-sidebar-collapsed" : ""}`} aria-label="会话导航">
      <div className="session-sidebar-brand">
        <div>
          <strong>ShopPilot</strong>
        </div>
        <button
          className="session-sidebar-toggle"
          onClick={onToggle}
          type="button"
          aria-label={collapsed ? "展开侧边栏" : "隐藏侧边栏"}
          title={collapsed ? "展开侧边栏" : "隐藏侧边栏"}
        >
          {collapsed ? "›" : "‹"}
        </button>
      </div>

      <div className="session-sidebar-collapsed-actions" aria-label="折叠侧边栏功能">
        <button className="session-sidebar-rail-button" onClick={onNewSession} type="button" title="新建会话" aria-label="新建会话">
          <span aria-hidden="true">＋</span>
          <small>新建</small>
        </button>
        <button className={`session-sidebar-rail-button ${activeView === "memory" ? "session-sidebar-rail-button-active" : ""}`} onClick={onOpenMemory} type="button" title="长期记忆" aria-label="长期记忆">
          <span aria-hidden="true">✦</span>
          <small>记忆</small>
        </button>
        <button className="session-sidebar-rail-button" onClick={onShowHistory} type="button" title="历史会话" aria-label="历史会话">
          <span aria-hidden="true">≡</span>
          <small>历史</small>
        </button>
      </div>

      <button className="session-new-button" onClick={onNewSession} type="button">
        <span aria-hidden="true">＋</span>
        新建会话
      </button>

      <nav className="session-nav" aria-label="主要导航">
        <button className={`session-nav-item ${activeView === "new" ? "session-nav-item-active" : ""}`} onClick={onNewSession} type="button">
          <span className="session-nav-icon" aria-hidden="true">⌂</span>
          <span>当前会话</span>
        </button>
        <button className={`session-nav-item ${activeView === "memory" ? "session-nav-item-active" : ""}`} onClick={onOpenMemory} type="button">
          <span className="session-nav-icon" aria-hidden="true">✦</span>
          <span>长期记忆</span>
          {memoryCount > 0 && <span className="session-nav-count">{memoryCount}</span>}
        </button>
      </nav>

      <label className="session-history-search">
        <span aria-hidden="true">⌕</span>
        <input
          aria-label="搜索历史会话"
          placeholder="搜索历史会话"
          value={historyQuery}
          onChange={(event) => setHistoryQuery(event.target.value)}
        />
      </label>
      <div className="session-history-heading">
        <span>历史会话</span>
        {items.length > 0 && <span>{filteredHistory.length}</span>}
      </div>
      <div className="session-history-list">
        {loading && <p className="session-sidebar-muted">正在加载…</p>}
        {!loading && error && <p className="session-sidebar-error">加载失败</p>}
        {!loading && !error && items.length === 0 && <p className="session-sidebar-muted">完成一次搜索后，会话会显示在这里。</p>}
        {!loading && !error && items.length > 0 && filteredHistory.length === 0 && <p className="session-sidebar-muted">没有匹配的历史会话。</p>}
        {!loading && !error && historyGroups.map((group) => (
          <section className="session-history-group" key={group.label}>
            <div className="session-history-group-heading">
              <span>{group.label}</span>
              <small>{group.items.length}</small>
            </div>
            {group.items.map((item) => (
              <button
                className={`session-history-item ${activeThreadId === item.thread_id ? "session-history-item-active" : ""}`}
                key={item.thread_id}
                onClick={() => onOpenHistory(item.thread_id)}
                type="button"
              >
                <strong>{item.query || "未命名会话"}</strong>
                <small>{formatDate(item.completed_at)}{item.plan?.category ? ` · ${item.plan.category}` : ""}</small>
              </button>
            ))}
          </section>
        ))}
      </div>
    </aside>
  );
}

const memoryKindLabels: Record<MemoryItem["kind"], string> = {
  preference: "偏好",
  blacklist: "排除项",
  history: "历史背景",
};

function MemoryPanel({
  items,
  loading,
  error,
  onBack,
}: {
  items: MemoryItem[];
  loading: boolean;
  error: string;
  onBack: () => void;
}) {
  return (
    <section className="memory-view" aria-labelledby="memory-view-title">
      <div className="memory-view-header">
        <div>
          <p className="eyebrow">ShopPilot · Memory</p>
          <h2 id="memory-view-title">长期记忆</h2>
          <p>Agent 会在新会话中参考这些稳定偏好，但本轮明确需求始终优先。</p>
        </div>
        <button className="secondary-button" onClick={onBack} type="button">新建会话</button>
      </div>

      {loading && <div className="memory-empty"><p>正在加载长期记忆…</p></div>}
      {!loading && error && <div className="memory-empty"><p className="history-error">{error}</p></div>}
      {!loading && !error && items.length === 0 && (
        <div className="memory-empty">
          <strong>还没有长期记忆</strong>
          <p>完成几次购物搜索后，Agent 会自动整理稳定的偏好和排除项。</p>
        </div>
      )}
      {!loading && !error && items.length > 0 && (
        <div className="memory-list">
          {items.map((item) => (
            <article className={`memory-item memory-item-${item.status}`} key={item.id}>
              <div className="memory-item-topline">
                <span className="memory-kind">{memoryKindLabels[item.kind]}</span>
                <span className="memory-status">{item.status === "active" ? "生效中" : item.status === "superseded" ? "已替代" : "有冲突"}</span>
              </div>
              <strong>{item.content}</strong>
              <div className="memory-item-meta">
                <span>{item.scope === "category" && item.category ? `品类：${item.category}` : "跨品类"}</span>
                <span>提及 {item.mention_count} 次</span>
                <span>置信度 {Math.round(item.confidence * 100)}%</span>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function historyRequestSummary(detail: HistoryDetail): string {
  const plan = detail.plan ?? {};
  const parts = [
    plan.category ? `围绕「${plan.category}」` : "围绕你的购物需求",
    plan.budget_cny != null ? `预算控制在 ¥${plan.budget_cny.toFixed(0)} 以内` : "比较商品价格",
    plan.platforms?.length ? `优先查看 ${plan.platforms.join("、")}` : "结合可用平台进行搜索",
  ];
  const preferences = [...(plan.hard_constraints ?? []), ...(plan.soft_preferences ?? [])].filter(Boolean);
  if (preferences.length > 0) parts.push(`并参考${preferences.slice(0, 3).join("、")}`);
  return `我会${parts.join("，")}，再比较价格和商品匹配度后给你推荐。`;
}

function HistorySessionView({
  detail,
  loading,
  error,
  onNewSession,
}: {
  detail: HistoryDetail | null;
  loading: boolean;
  error: string;
  onNewSession: () => void;
}) {
  if (loading) {
    return <section className="chat-shell session-state-panel"><p className="muted">正在打开历史会话…</p></section>;
  }
  if (error || !detail) {
    return (
      <section className="chat-shell session-state-panel">
        <h2>无法打开会话</h2>
        <p className="history-error">{error || "历史会话不存在"}</p>
        <button onClick={onNewSession} type="button">新建会话</button>
      </section>
    );
  }

  return (
    <section className="chat-shell history-session-shell">
      <div className="session-view-heading">
        <div>
          <h2>{detail.query}</h2>
          <small>{formatDate(detail.completed_at)}{detail.plan?.category ? ` · ${detail.plan.category}` : ""}</small>
        </div>
        <button className="secondary-button" onClick={onNewSession} type="button">新建会话</button>
      </div>
      <div className="chat-messages history-session-messages">
        <div className="chat-message user-message history-user-message">
          <div className="chat-bubble chat-bubble-user">{detail.query}</div>
        </div>
        <div className="chat-message assistant-message history-intro-message">
          <span className="agent-avatar" aria-hidden="true">SP</span>
          <div className="chat-bubble chat-bubble-assistant">
            <strong>ShopPilot Agent</strong>
            <p>{historyRequestSummary(detail)}</p>
          </div>
        </div>
        <section className="history-session-process">
          <div className="history-session-process-heading">
            <div>
              <strong>Agent 执行摘要</strong>
            </div>
            <span>5 项动作</span>
          </div>
          <EventTimeline events={detail.events} emptyText="该会话没有保存执行摘要。" rootThreadId={detail.thread_id} />
        </section>
        <section className="chat-message assistant-message final-message">
          <span className="agent-avatar" aria-hidden="true">SP</span>
          <div className="chat-bubble chat-bubble-assistant final-answer-bubble">
            <div className="final-answer-heading">
              <div>
                <strong>推荐结果</strong>
              </div>
              <span className="result-status">已完成</span>
            </div>
            <MarkdownContent source={detail.final_answer || "该会话没有保存最终文本。"} />
          </div>
        </section>
      </div>
    </section>
  );
}

function SearchPage({ initialThreadId = null }: { initialThreadId?: string | null }) {
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [events, setEvents] = useState<AguiEvent[]>([]);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [finalAnswer, setFinalAnswer] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [processOpen, setProcessOpen] = useState(true);
  const [running, setRunning] = useState(false);
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const [activeView, setActiveView] = useState<"new" | "history" | "memory">(initialThreadId ? "history" : "new");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(initialThreadId);
  const [selectedHistory, setSelectedHistory] = useState<HistoryDetail | null>(null);
  const [historyDetailLoading, setHistoryDetailLoading] = useState(Boolean(initialThreadId));
  const [historyDetailError, setHistoryDetailError] = useState("");
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [memoryLoading, setMemoryLoading] = useState(true);
  const [memoryError, setMemoryError] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
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

  async function loadMemories() {
    try {
      setMemoryError("");
      const response = await fetch(`/api/users/${encodeURIComponent(DEMO_USER_ID)}/memories?include_inactive=true`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const body = (await response.json()) as { memories?: MemoryItem[] };
      setMemories(body.memories ?? []);
    } catch (error) {
      setMemoryError(`长期记忆加载失败：${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setMemoryLoading(false);
    }
  }

  async function loadHistoryDetail(historyId: string) {
    try {
      setHistoryDetailLoading(true);
      setHistoryDetailError("");
      const response = await fetch(
        `/api/history/${encodeURIComponent(historyId)}?user_id=${encodeURIComponent(DEMO_USER_ID)}`,
      );
      if (!response.ok) throw new Error(response.status === 404 ? "历史会话不存在" : `HTTP ${response.status}`);
      setSelectedHistory((await response.json()) as HistoryDetail);
    } catch (error) {
      setSelectedHistory(null);
      setHistoryDetailError(error instanceof Error ? error.message : String(error));
    } finally {
      setHistoryDetailLoading(false);
    }
  }

  useEffect(() => {
    void loadHistory();
    void loadMemories();
    if (initialThreadId) void loadHistoryDetail(initialThreadId);
    return () => wsRef.current?.close();
  }, [initialThreadId]);

  function startNewSession() {
    wsRef.current?.close();
    setActiveView("new");
    setSelectedHistoryId(null);
    setSelectedHistory(null);
    setHistoryDetailError("");
    setQuery("");
    setThreadId(null);
    setSubmittedQuery("");
    setEvents([]);
    setStreamingAnswer("");
    setFinalAnswer("");
    setFiles([]);
    setRunning(false);
    setProcessOpen(true);
    window.history.replaceState({}, "", "/");
  }

  function openHistory(historyId: string) {
    wsRef.current?.close();
    setActiveView("history");
    setSelectedHistoryId(historyId);
    setSelectedHistory(null);
    setHistoryDetailError("");
    void loadHistoryDetail(historyId);
  }

  function openMemory() {
    wsRef.current?.close();
    setActiveView("memory");
  }

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
        setProcessOpen(false);
        void loadHistory();
        void loadMemories();
      }
      if (payload.event === "error" || payload.event === "task_cancelled") {
        setRunning(false);
      }
    };
  }

  async function startTask() {
    if (!query.trim()) return;
    setActiveView("new");
    setSelectedHistoryId(null);
    setSelectedHistory(null);
    setSubmittedQuery(query.trim());
    setRunning(true);
    setEvents([]);
    setStreamingAnswer("");
    streamMessageIdRef.current = null;
    setFinalAnswer("");
    setFiles([]);
    setProcessOpen(true);
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
    <main className="chat-page">
      <div className={`workspace-layout ${sidebarCollapsed ? "workspace-layout-sidebar-collapsed" : ""}`}>
        <SessionSidebar
          items={historyItems}
          loading={historyLoading}
          error={historyError}
          collapsed={sidebarCollapsed}
          activeView={activeView}
          activeThreadId={selectedHistoryId}
          memoryCount={memories.filter((item) => item.status === "active").length}
          onNewSession={startNewSession}
          onOpenHistory={openHistory}
          onOpenMemory={openMemory}
          onShowHistory={() => setSidebarCollapsed(false)}
          onToggle={() => setSidebarCollapsed((collapsed) => !collapsed)}
        />

        <section className="workspace-main">
          {activeView === "memory" ? (
            <MemoryPanel items={memories} loading={memoryLoading} error={memoryError} onBack={startNewSession} />
          ) : activeView === "history" ? (
            <HistorySessionView
              detail={selectedHistory}
              loading={historyDetailLoading}
              error={historyDetailError}
              onNewSession={startNewSession}
            />
          ) : (
            <section className="chat-shell">
              <div className="chat-messages" aria-live="polite">
                <div className="chat-message assistant-message welcome-message">
                  <span className="agent-avatar" aria-hidden="true">SP</span>
                  <div className="chat-bubble chat-bubble-assistant">
                    <strong>ShopPilot Agent</strong>
                    <p>你好！告诉我想买什么、预算和偏好，我会帮你搜索并比较合适的商品。</p>
                  </div>
                </div>

                {submittedQuery && (
                  <div className="chat-message user-message">
                    <div className="chat-bubble chat-bubble-user">{submittedQuery}</div>
                  </div>
                )}

                {(running || events.length > 0) && (
                  <section className={`chat-message agent-process-message ${running ? "agent-process-running" : ""}`}>
                    <div className="chat-message-meta">
                      <span className="agent-avatar agent-avatar-small" aria-hidden="true">SP</span>
                      <div>
                        <strong>ShopPilot Agent</strong>
                        <span>{running ? "正在搜索" : "搜索已完成"}</span>
                      </div>
                    </div>
                    <button
                      className="agent-process-toggle"
                      onClick={() => setProcessOpen((open) => !open)}
                      type="button"
                      aria-expanded={processOpen}
                    >
                      <span className={`agent-status-dot ${running ? "agent-status-dot-active" : ""}`} aria-hidden="true" />
                      <span className="agent-process-copy">
                        <strong>{processStatus(events, running)}</strong>
                        <small>已提炼关键动作 · 点击查看搜索过程</small>
                      </span>
                      <span className="agent-process-chevron" aria-hidden="true">{processOpen ? "⌃" : "⌄"}</span>
                    </button>
                    {processOpen && (
                      <div className="agent-process-content">
                        <EventTimeline events={events} emptyText="Agent 正在准备搜索过程…" running={running} rootThreadId={threadId} />
                      </div>
                    )}
                  </section>
                )}

                {(finalAnswer || streamingAnswer) && (
                  <section className="chat-message assistant-message final-message">
                    <span className="agent-avatar" aria-hidden="true">SP</span>
                    <div className="chat-bubble chat-bubble-assistant final-answer-bubble">
                      <div className="final-answer-heading">
                        <div>
                          <strong>推荐结果</strong>
                          <span>{finalAnswer ? "根据你的预算和偏好整理完成" : "Agent 正在整理推荐"}</span>
                        </div>
                        {finalAnswer && <span className="result-status">已完成</span>}
                      </div>
                      {finalAnswer ? (
                        <MarkdownContent source={finalAnswer} />
                      ) : (
                        <div className="streaming-output" aria-live="polite">
                          <MarkdownContent source={streamingAnswer} />
                          <span className="streaming-caret" aria-hidden="true" />
                        </div>
                      )}
                    </div>
                  </section>
                )}
              </div>

              <section className="chat-composer">
                <div className="composer-input-row">
                  <textarea
                    aria-label="购物需求"
                    placeholder="例如：在 amazon、walmart、ebay 搜索预算 300 元以内的咖啡杯，优先耐用"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                  />
                  <button
                    className="composer-send-button"
                    onClick={running ? cancelTask : startTask}
                    type="button"
                    aria-label={running ? "取消任务" : "发送购物需求"}
                    title={running ? "取消任务" : "发送购物需求"}
                  >
                    {running ? "×" : "↑"}
                  </button>
                </div>
              </section>
            </section>
          )}
        </section>
      </div>
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
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>Agent 执行摘要</h2>
                <p className="panel-description">展示关键动作；底层事件仅在展开详情时查看。</p>
              </div>
              <span className="history-count">5 项动作</span>
            </div>
            <EventTimeline events={detail.events} emptyText="该历史任务没有保存事件流。" rootThreadId={detail.thread_id} />
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
  return <SearchPage initialThreadId={detailMatch ? decodeURIComponent(detailMatch[1]) : null} />;
}
