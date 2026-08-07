export type EvaluationCheck = {
  rule: string;
  title: string;
  section: string;
  severity: "error" | "warning" | "info";
  status: "pass" | "fail" | "skipped";
  passed: boolean;
  message: string;
  max_points: number;
  details?: Record<string, unknown>;
};

export type EvaluationSection = {
  key: string;
  label: string;
  score: number;
  max_score: number;
  passed_checks: number;
  failed_checks: number;
  skipped_checks: number;
};

export type JudgeDimension = {
  score: number;
  reason: string;
};

export type LlmJudgeResult = {
  status: "completed";
  judge_version: string;
  context_version?: string;
  stale?: boolean;
  model: string;
  evaluated_at: string;
  score: number;
  weights: Record<string, number>;
  dimensions: {
    planning_quality: JudgeDimension;
    tool_selection: JudgeDimension;
    trajectory_efficiency: JudgeDimension;
    final_answer_quality: JudgeDimension;
  };
  strengths: string[];
  issues: string[];
  suggestions: string[];
  verdict: "excellent" | "good" | "mixed" | "poor";
};

export type TrajectoryEvaluation = {
  schema_version: string;
  evaluator: string;
  score: number;
  passed: boolean;
  summary: {
    errors: number;
    warnings: number;
    passed: number;
    skipped: number;
    total: number;
  };
  metrics: {
    duration_ms?: number;
    model_steps?: number;
    model_calls_all_agents?: number;
    tool_calls?: number;
    tool_attempts?: number;
    tool_successes?: number;
    tool_failures?: number;
    tool_retries?: number;
    idempotency_hits?: number;
    fork_count?: number;
    fork_rejected?: number;
    actor_count?: number;
    picked_items?: number;
    terminal_tool?: string | null;
    duplicate_executions?: number;
    slow_tool_count?: number;
    longest_tool?: {
      tool_name?: string;
      duration_ms?: number;
      actor_thread_id?: string;
    } | null;
    [key: string]: unknown;
  };
  sections: Record<string, EvaluationSection>;
  checks: EvaluationCheck[];
  llm_judge: LlmJudgeResult | null;
};

function formatDuration(durationMs: number | undefined): string {
  if (durationMs == null || Number.isNaN(durationMs)) return "—";
  if (durationMs < 1000) return `${durationMs} ms`;
  if (durationMs < 60_000) return `${(durationMs / 1000).toFixed(1)} s`;
  const minutes = Math.floor(durationMs / 60_000);
  const seconds = Math.round((durationMs % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false });
}

function statusLabel(check: EvaluationCheck): string {
  if (check.status === "skipped") return "跳过";
  if (check.status === "pass") return "通过";
  return check.severity === "warning" ? "警告" : "异常";
}

function statusClass(check: EvaluationCheck): string {
  if (check.status === "skipped") return "check-skipped";
  if (check.status === "pass") return "check-pass";
  return check.severity === "warning" ? "check-warning" : "check-error";
}

function Metric({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="evaluation-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <small>{hint}</small>}
    </div>
  );
}

const judgeDimensionLabels: Record<keyof LlmJudgeResult["dimensions"], string> = {
  planning_quality: "规划质量",
  tool_selection: "工具选择",
  trajectory_efficiency: "轨迹效率",
  final_answer_quality: "最终答案",
};

const verdictLabels: Record<LlmJudgeResult["verdict"], string> = {
  excellent: "优秀",
  good: "良好",
  mixed: "有明显优化空间",
  poor: "质量较差",
};

function JudgeList({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="judge-list-block">
      <h4>{title}</h4>
      <ul>
        {items.map((item, index) => <li key={`${title}-${index}`}>{item}</li>)}
      </ul>
    </div>
  );
}

function JudgePanel({
  judge,
  running,
  error,
  onRun,
}: {
  judge: LlmJudgeResult | null;
  running: boolean;
  error: string;
  onRun?: (force: boolean) => void;
}) {
  if (!judge) {
    return (
      <div className="judge-placeholder judge-ready">
        <div>
          <strong>LLM-as-a-Judge</strong>
          <p>由独立 Judge 模型评价规划质量、工具选择、轨迹效率与最终答案；确定性错误仍由上面的 Rule-based Evaluator 负责。</p>
          {error && <p className="judge-error">{error}</p>}
        </div>
        <button disabled={running || !onRun} onClick={() => onRun?.(false)} type="button">
          {running ? "评审中…" : "运行 LLM Judge"}
        </button>
      </div>
    );
  }

  return (
    <div className="judge-panel">
      <div className="judge-hero">
        <div>
          <p className="history-detail-label">LLM-as-a-Judge</p>
          <h3>语义质量评审</h3>
          <p className="panel-description">
            {judge.model} · {judge.judge_version} · {formatDate(judge.evaluated_at)}
          </p>
          {judge.stale && (
            <p className="judge-stale">该结果来自旧版 Judge Rubric / Context，建议按当前 v2 规则重新评审。</p>
          )}
        </div>
        <div className="judge-score">
          <strong>{judge.score.toFixed(1)}</strong>
          <span>/ 100</span>
          <small>{verdictLabels[judge.verdict]}</small>
        </div>
      </div>

      <div className="judge-dimensions">
        {(Object.entries(judge.dimensions) as [keyof LlmJudgeResult["dimensions"], JudgeDimension][]).map(([key, dimension]) => (
          <article className="judge-dimension" key={key}>
            <div>
              <strong>{judgeDimensionLabels[key]}</strong>
              <span>{dimension.score} / 5</span>
            </div>
            <p>{dimension.reason}</p>
          </article>
        ))}
      </div>

      <div className="judge-lists">
        <JudgeList title="优点" items={judge.strengths} />
        <JudgeList title="发现的问题" items={judge.issues} />
        <JudgeList title="改进建议" items={judge.suggestions} />
      </div>

      {error && <p className="judge-error">{error}</p>}
      <div className="judge-actions">
        <button className="secondary-button" disabled={running || !onRun} onClick={() => onRun?.(true)} type="button">
          {running ? "重新评审中…" : judge.stale ? "按新版重新评审" : "重新运行 Judge"}
        </button>
        <span>已有结果会缓存到 evaluation.json；只有重新评审才再次调用模型。</span>
      </div>
    </div>
  );
}

export default function EvaluationPanel({
  evaluation,
  onRunJudge,
  judgeRunning = false,
  judgeError = "",
}: {
  evaluation: TrajectoryEvaluation;
  onRunJudge?: (force: boolean) => void;
  judgeRunning?: boolean;
  judgeError?: string;
}) {
  const orderedChecks = [...evaluation.checks].sort((left, right) => {
    const priority = (check: EvaluationCheck) => {
      if (check.status === "fail" && check.severity === "error") return 0;
      if (check.status === "fail" && check.severity === "warning") return 1;
      if (check.status === "skipped") return 3;
      return 2;
    };
    return priority(left) - priority(right);
  });
  const longest = evaluation.metrics.longest_tool;

  return (
    <section className="panel evaluation-panel">
      <div className="evaluation-hero">
        <div>
          <p className="history-detail-label">Rule-based Trajectory Evaluator</p>
          <h2>Agent 运行评估</h2>
          <p className="panel-description">
            基于 trace.json 与 result.json 的确定性规则检查；LLM Judge 作为独立的第二层语义评审。
          </p>
        </div>
        <div className={`evaluation-score ${evaluation.passed ? "score-pass" : "score-error"}`}>
          <strong>{evaluation.score.toFixed(1)}</strong>
          <span>/ 100</span>
          <small>{evaluation.passed ? "规则校验通过" : "存在确定性错误"}</small>
        </div>
      </div>

      <div className="evaluation-summary-strip">
        <span><strong>{evaluation.summary.errors}</strong> 个错误</span>
        <span><strong>{evaluation.summary.warnings}</strong> 个警告</span>
        <span><strong>{evaluation.summary.passed}</strong> 条通过</span>
        <span><strong>{evaluation.summary.skipped}</strong> 条跳过</span>
      </div>

      <div className="evaluation-sections">
        {Object.values(evaluation.sections).map((section) => (
          <div className="evaluation-section" key={section.key}>
            <div className="evaluation-section-heading">
              <span>{section.label}</span>
              <strong>{section.score.toFixed(1)} / {section.max_score.toFixed(0)}</strong>
            </div>
            <div className="evaluation-progress" aria-label={`${section.label} ${section.score}/${section.max_score}`}>
              <span style={{ width: `${Math.max(0, Math.min(100, section.score / section.max_score * 100))}%` }} />
            </div>
            <small>
              {section.passed_checks} 通过 · {section.failed_checks} 未通过
              {section.skipped_checks ? ` · ${section.skipped_checks} 跳过` : ""}
            </small>
          </div>
        ))}
      </div>

      <div className="evaluation-subheading">
        <div>
          <h3>运行指标</h3>
          <p>用于观察稳定性与执行效率，不把可恢复的降级直接当成答案错误。</p>
        </div>
      </div>
      <div className="evaluation-metrics">
        <Metric label="主 Agent 轮数" value={Number(evaluation.metrics.model_steps ?? 0)} />
        <Metric label="逻辑工具调用" value={Number(evaluation.metrics.tool_calls ?? 0)} />
        <Metric label="工具重试" value={Number(evaluation.metrics.tool_retries ?? 0)} />
        <Metric label="工具最终失败" value={Number(evaluation.metrics.tool_failures ?? 0)} />
        <Metric label="Fork 数" value={Number(evaluation.metrics.fork_count ?? 0)} />
        <Metric label="Fork 降级" value={Number(evaluation.metrics.fork_rejected ?? 0)} />
        <Metric label="总耗时" value={formatDuration(Number(evaluation.metrics.duration_ms ?? 0))} />
        <Metric
          label="最慢工具"
          value={longest?.tool_name || "—"}
          hint={longest?.duration_ms != null ? formatDuration(longest.duration_ms) : undefined}
        />
      </div>

      <div className="evaluation-subheading checks-heading">
        <div>
          <h3>规则检查</h3>
          <p>允许多条合法 Agent 路径，只校验生命周期、依赖、不变量和业务硬约束。</p>
        </div>
        <code>{evaluation.evaluator}</code>
      </div>
      <div className="evaluation-checks">
        {orderedChecks.map((check) => (
          <article className="evaluation-check" key={check.rule}>
            <span className={`evaluation-check-status ${statusClass(check)}`}>
              {statusLabel(check)}
            </span>
            <div>
              <div className="evaluation-check-title">
                <strong>{check.title}</strong>
                <code>{check.rule}</code>
              </div>
              <p>{check.message}</p>
            </div>
          </article>
        ))}
      </div>

      <JudgePanel
        judge={evaluation.llm_judge}
        running={judgeRunning}
        error={judgeError}
        onRun={onRunJudge}
      />
    </section>
  );
}
