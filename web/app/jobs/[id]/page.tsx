"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { RawTab, TabBar, type TabId } from "../../../components/JobTabs";
import { PreviewThumb } from "../../../components/PreviewThumb";

type Validation = { rule: string; expected: string | null; actual: string | null; passed: boolean };
type Output = {
  id: string;
  platform: string;
  ratio: string;
  kind: string;
  preview_key: string | null;
  speaker_on_screen_pct: number | null;
};
type Decision = {
  stage: string;
  choice: string;
  reason: string | null;
  confidence: number | null;
};
type OutputDetail = {
  output?: Output;
  master?: unknown;
  validations?: Validation[];
  decisions?: Decision[];
};
type JobData = {
  job: {
    id: string;
    status: string;
    stage: string | null;
    progress: number | null;
    error: string | null;
    master_title?: string | null;
  };
  outputs: Output[];
};

const iconProps = {
  width: 14,
  height: 14,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

function CheckIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <path d="M4 12.5l5 5L20 6.5" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg
      {...iconProps}
      aria-hidden="true"
      style={{ animation: "job-spin 1s linear infinite" }}
    >
      <path d="M12 3a9 9 0 1 1-9 9" />
    </svg>
  );
}

function ArrowUpRightIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <path d="M7 17L17 7M9 7h8v8" />
    </svg>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="ghost"
      style={{
        padding: "2px 4px",
        border: "none",
        background: "none",
        color: "var(--muted)",
        cursor: "pointer",
        display: "inline-flex",
        lineHeight: 0,
      }}
      aria-label="Copy id"
      title="Copy id"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        });
      }}
    >
      {copied ? (
        <CheckIcon />
      ) : (
        <svg {...iconProps} aria-hidden="true">
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15V5a2 2 0 0 1 2-2h10" />
        </svg>
      )}
    </button>
  );
}

function StatusBadge({ status }: { status: string }) {
  const failed = status === "failed";
  const done = status === "done" || status === "completed";
  return (
    <span className={`badge ${failed ? "fail" : done ? "pass" : "running"}`}>
      <span style={{ display: "inline-flex", verticalAlign: "-2px", marginRight: 5 }}>
        {failed ? <XIcon /> : done ? <CheckIcon /> : <SpinnerIcon />}
      </span>
      {status}
    </span>
  );
}

export default function JobPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [data, setData] = useState<JobData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("validation");

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Failed to load job (${res.status})`);
      setData(await res.json());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load job");
    }
  }, [id]);

  useEffect(() => {
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, [poll]);

  if (error && !data) return <div className="empty error-text">{error}</div>;
  if (!data) return <div className="empty">Loading…</div>;

  const { job, outputs } = data;
  const done = job.status === "done" || job.status === "completed";
  const failed = job.status === "failed";
  const running = !done && !failed;
  const progress = Math.round(job.progress ?? 0);

  return (
    <>
      <header className="page-head">
        <p className="micro" style={{ margin: "0 0 8px" }}>Job</p>
        <div className="row" style={{ gap: 12, justifyContent: "center" }}>
          <h1 style={{ margin: 0 }}>{job.master_title ?? "Job"}</h1>
          <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
            {job.id.slice(0, 8)}
          </span>
          <CopyButton text={job.id} />
        </div>
        <div className="row" style={{ gap: 12, justifyContent: "center", marginTop: 10 }}>
          <StatusBadge status={job.status} />
          {running && (
            <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
              {job.stage ?? "working"} · {progress}%
            </span>
          )}
        </div>
        {running && (
          <div style={{ maxWidth: 480, margin: "16px auto 0" }}>
            <div className="progress">
              <div style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}
        {failed && job.error && (
          <p className="error-text" style={{ fontSize: 14, margin: "16px 0 0" }}>{job.error}</p>
        )}
      </header>

      {done && (
        <section style={{ marginTop: 40 }}>
          <p className="micro" style={{ margin: "0 0 14px" }}>Outputs · {outputs.length}</p>
          {outputs.length === 0 ? (
            <div className="empty">No outputs recorded for this job.</div>
          ) : (
            <div className="grid">
              {outputs.map((o) => (
                <Link key={o.id} href={`/assets/${o.id}`} style={{ display: "block" }}>
                  <PreviewThumb
                    previewKey={o.preview_key}
                    alt={`${o.platform} ${o.ratio}`}
                    emptyLabel="Preview unavailable"
                    style={{
                      width: "100%",
                      borderRadius: 8,
                      border: "1px solid var(--border)",
                      display: "block",
                      marginBottom: 10,
                    }}
                  />
                  <div className="row" style={{ gap: 8 }}>
                    <span className="micro">{o.platform} · {o.ratio}</span>
                    <span
                      style={{ display: "inline-flex", marginLeft: "auto", color: "var(--muted)" }}
                      title="View asset"
                    >
                      <ArrowUpRightIcon />
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>
      )}

      <TabSection jobId={id} outputs={outputs} done={done} running={running} failed={failed} tab={tab} onTabChange={setTab} />

      <p style={{ marginTop: 48 }}>
        <Link className="btn ghost" href="/library">
          Go to library
        </Link>
      </p>
    </>
  );
}

function TabSection({
  jobId,
  outputs,
  done,
  running,
  failed,
  tab,
  onTabChange,
}: {
  jobId: string;
  outputs: Output[];
  done: boolean;
  running: boolean;
  failed: boolean;
  tab: TabId;
  onTabChange: (t: TabId) => void;
}) {
  const [details, setDetails] = useState<(OutputDetail | null)[] | null>(null);
  const [detailsDone, setDetailsDone] = useState(false);

  // Fetch per-output validations/decisions lazily and only once, when the job
  // finishes. Polling keeps running but must NOT refetch this every 2s.
  useEffect(() => {
    if (!done || !outputs.length) return;
    let cancelled = false;
    Promise.all(
      outputs.map((o) =>
        fetch(`/api/assets/${o.id}`, { cache: "no-store" })
          .then((r) => (r.ok ? r.json() : null))
          .catch(() => null)
      )
    ).then((details) => {
      if (cancelled) return;
      setDetails(details);
      setDetailsDone(true);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done, jobId]);

  const groups = outputs
    .map((o, i) => ({
      label: `${o.platform} · ${o.ratio}`,
      validations: details?.[i]?.validations ?? [],
    }))
    .filter((g) => g.validations.length > 0);

  const decisions: Decision[] = details?.flatMap((d) => d?.decisions ?? []) ?? [];
  const hasValidation = groups.length > 0;
  const hasDecisions = decisions.length > 0;

  const allTabs: { id: TabId; label: string }[] = [
    { id: "validation", label: "Validation" },
    { id: "decisions", label: "Decisions" },
    { id: "raw", label: "Raw" },
  ];
  // While the job runs and nothing has landed yet, only Raw is meaningful.
  const tabs =
    !hasValidation && !hasDecisions && (running || failed)
      ? allTabs.filter((t) => t.id === "raw")
      : allTabs;
  const activeTab: TabId = tabs.some((t) => t.id === tab) ? tab : "raw";

  return (
    <>
      <TabBar tabs={tabs} active={activeTab} onChange={onTabChange} />

      {activeTab === "validation" && (
        <section>
          <ValidationContent
            done={done}
            detailsDone={detailsDone}
            groups={groups}
          />
        </section>
      )}

      {activeTab === "decisions" && (
        <section>
          <DecisionsContent done={done} detailsDone={detailsDone} decisions={decisions} />
        </section>
      )}

      {activeTab === "raw" && <RawTab jobId={jobId} />}
    </>
  );
}

function ValidationContent({
  done,
  detailsDone,
  groups,
}: {
  done: boolean;
  detailsDone: boolean;
  groups: { label: string; validations: Validation[] }[];
}) {
  if (!done && !detailsDone) {
    return (
      <div className="empty">Waiting for first results — validations appear once the job finishes.</div>
    );
  }
  if (!detailsDone) {
    return (
      <div className="row" style={{ gap: 8, color: "var(--muted)" }}>
        <SpinnerIcon />
        <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>loading…</span>
      </div>
    );
  }
  if (!groups.length) {
    return <div className="empty">No validations recorded for this job.</div>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 36 }}>
      {groups.map((g, gi) => (
        <div key={gi}>
          <p className="micro" style={{ margin: "0 0 4px" }}>{g.label}</p>
          {g.validations.map((v, i) => (
            <div
              key={i}
              className="row"
              style={{
                justifyContent: "space-between",
                gap: 16,
                padding: "12px 0",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <span style={{ fontSize: 14, color: v.passed ? "inherit" : "var(--fail)" }}>{v.rule}</span>
              <span
                className="row"
                style={{ gap: 10, fontFamily: "var(--font-mono)", fontSize: 12 }}
              >
                <span className="muted">{v.expected ?? "—"}</span>
                <span className="muted">→</span>
                <span>{v.actual ?? "—"}</span>
              </span>
              <span
                style={{
                  display: "inline-flex",
                  flex: "none",
                  color: v.passed ? "var(--ok)" : "var(--fail)",
                }}
                title={v.passed ? "Passed" : "Failed"}
                aria-label={v.passed ? "Passed" : "Failed"}
              >
                {v.passed ? <CheckIcon /> : <XIcon />}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function DecisionsContent({
  done,
  detailsDone,
  decisions,
}: {
  done: boolean;
  detailsDone: boolean;
  decisions: Decision[];
}) {
  if (!done && !detailsDone) {
    return (
      <div className="empty">Waiting for first results — decisions appear once the job finishes.</div>
    );
  }
  if (!detailsDone) {
    return (
      <div className="row" style={{ gap: 8, color: "var(--muted)" }}>
        <SpinnerIcon />
        <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>loading…</span>
      </div>
    );
  }
  if (!decisions.length) {
    return <div className="empty">No decisions recorded for this job.</div>;
  }
  return (
    <div style={{ position: "relative", paddingLeft: 20 }}>
      {/* left hairline rail */}
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          left: 3,
          top: 8,
          bottom: 8,
          width: 1,
          background: "var(--border)",
        }}
      />
      {decisions.map((d, i) => (
        <div key={i} style={{ position: "relative", padding: "14px 0" }}>
          {/* node dot */}
          <span
            aria-hidden="true"
            style={{
              position: "absolute",
              left: -20.5,
              top: 22,
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: "var(--bg)",
              border: "1.5px solid var(--text)",
            }}
          />
          <div className="row" style={{ gap: 12 }}>
            <span className="micro" style={{ color: "var(--text)" }}>{d.stage}</span>
            {d.confidence != null && (
              <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
                conf {d.confidence.toFixed(2)}
              </span>
            )}
          </div>
          <p style={{ margin: "4px 0 0", fontSize: 14 }}>{d.choice}</p>
          {d.reason && (
            <p className="muted" style={{ margin: "4px 0 0", fontSize: 13 }}>{d.reason}</p>
          )}
        </div>
      ))}
    </div>
  );
}
