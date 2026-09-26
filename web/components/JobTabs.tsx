"use client";

import { useEffect, useState } from "react";

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

function ShieldIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <path d="M12 3l7 3v5c0 4.5-3 8.5-7 10-4-1.5-7-5.5-7-10V6l7-3z" />
      <path d="M9 12l2 2 4-4.5" />
    </svg>
  );
}

function ListIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <path d="M8 6h12M8 12h12M8 18h12" />
      <path d="M4 6h.01M4 12h.01M4 18h.01" />
    </svg>
  );
}

function BracesIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <path d="M8 4c-2 0-3 1-3 3v2c0 1.5-.8 2.5-2 3 1.2.5 2 1.5 2 3v2c0 2 1 3 3 3" />
      <path d="M16 4c2 0 3 1 3 3v2c0 1.5.8 2.5 2 3-1.2.5-2 1.5-2 3v2c0 2-1 3-3 3" />
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

export type TabId = "validation" | "decisions" | "raw";

export function TabBar({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: TabId; label: string }[];
  active: TabId;
  onChange: (id: TabId) => void;
}) {
  return (
    <div
      role="tablist"
      style={{
        display: "flex",
        gap: 28,
        borderBottom: "1px solid var(--border)",
        margin: "48px 0 32px",
      }}
    >
      {tabs.map((t) => {
        const isActive = t.id === active;
        return (
          <button
            key={t.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(t.id)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 7,
              background: "none",
              border: "none",
              borderBottom: `2px solid ${isActive ? "var(--text)" : "transparent"}`,
              color: isActive ? "var(--text)" : "var(--muted)",
              borderRadius: 0,
              padding: "8px 2px 10px",
              marginBottom: -1,
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              fontWeight: 500,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              transition: "color 0.15s, border-color 0.15s",
            }}
            onMouseEnter={(e) => {
              if (!isActive) e.currentTarget.style.color = "var(--text)";
            }}
            onMouseLeave={(e) => {
              if (!isActive) e.currentTarget.style.color = "var(--muted)";
            }}
          >
            {t.id === "validation" ? <ShieldIcon /> : t.id === "decisions" ? <ListIcon /> : <BracesIcon />}
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---------------- Raw artifacts ---------------- */

type RawArtifact = {
  kind: string;
  key: string;
  label?: string | null;
  bytes?: number | null;
};
type Manifest = {
  job_id?: string;
  generated_at?: string;
  artifacts?: RawArtifact[];
};

function fmtBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function serveUrl(key: string): string {
  return `/api/serve?key=${encodeURIComponent(key)}`;
}

function ArtifactContent({ artifact }: { artifact: RawArtifact }) {
  const [state, setState] = useState<
    { status: "loading" } | { status: "error"; error: string } | { status: "ready"; data: unknown }
  >({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    fetch(serveUrl(artifact.key), { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load artifact (${r.status})`);
        return r.json();
      })
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data });
      })
      .catch((e) => {
        if (!cancelled)
          setState({
            status: "error",
            error: e instanceof Error ? e.message : "Failed to load artifact",
          });
      });
    return () => {
      cancelled = true;
    };
  }, [artifact.key]);

  if (state.status === "loading") {
    return (
      <div className="row" style={{ gap: 8, padding: "14px 0", color: "var(--muted)" }}>
        <SpinnerIcon />
        <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
          loading…
        </span>
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <p className="error-text" style={{ fontSize: 13, padding: "14px 0" }}>
        {state.error}
      </p>
    );
  }

  if (artifact.kind === "vlm_calls") {
    return <VlmCallsView data={state.data} />;
  }

  return (
    <pre
      style={{
        margin: 0,
        padding: 16,
        background: "#141414",
        color: "#d7d3c8",
        borderRadius: "var(--radius)",
        border: "2px solid var(--border)",
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        lineHeight: 1.55,
        overflow: "auto",
        maxHeight: 420,
      }}
    >
      {JSON.stringify(state.data, null, 2)}
    </pre>
  );
}

function asText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

function VlmCallsView({ data }: { data: unknown }) {
  const calls: unknown[] = Array.isArray(data)
    ? data
    : data && typeof data === "object" && Array.isArray((data as { calls?: unknown[] }).calls)
      ? ((data as { calls: unknown[] }).calls)
      : [];

  if (!calls.length) {
    return (
      <div className="empty" style={{ padding: 20 }}>
        No VLM calls recorded.
      </div>
    );
  }

  return (
    <div>
      {calls.map((c, i) => (
        <VlmCallRow key={i} call={c as Record<string, unknown>} index={i} />
      ))}
    </div>
  );
}

function VlmCallRow({ call, index }: { call: Record<string, unknown>; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const provider = asText(call.provider) || "—";
  const model = asText(call.model) || "—";
  const latencyRaw = call.latency_ms ?? call.latency;
  const latency = typeof latencyRaw === "number" ? `${Math.round(latencyRaw)}ms` : asText(latencyRaw) || "—";
  const prompt = asText(call.prompt ?? call.prompt_text);
  const reply = asText(call.reply ?? call.response ?? call.output);
  const promptExcerpt = expanded ? prompt : prompt.slice(0, 120);
  const replyExcerpt = expanded ? reply : reply.slice(0, 120);

  return (
    <div style={{ borderBottom: "1px solid var(--border)", padding: "12px 0" }}>
      <div className="row" style={{ gap: 12 }}>
        <span className="micro" style={{ color: "var(--text)" }}>
          call {String(index + 1).padStart(2, "0")}
        </span>
        <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
          {provider} · {model}
        </span>
        {latency !== "—" && (
          <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11, marginLeft: "auto" }}>
            {latency}
          </span>
        )}
      </div>
      {prompt && (
        <p style={{ margin: "8px 0 0", fontSize: 13 }}>
          <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>prompt · </span>
          {promptExcerpt}
          {!expanded && prompt.length > 120 ? "…" : ""}
        </p>
      )}
      {reply && (
        <p style={{ margin: "4px 0 0", fontSize: 13 }}>
          <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>reply · </span>
          {replyExcerpt}
          {!expanded && reply.length > 120 ? "…" : ""}
        </p>
      )}
      {(prompt.length > 120 || reply.length > 120) && (
        <button
          className="ghost"
          onClick={() => setExpanded(!expanded)}
          style={{
            marginTop: 8,
            padding: "2px 8px",
            fontSize: 11,
            fontFamily: "var(--font-mono)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

function ArtifactExpander({ artifact }: { artifact: RawArtifact }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "none",
          border: "none",
          borderRadius: 0,
          padding: "14px 2px",
          color: "var(--text)",
          cursor: "pointer",
          textAlign: "left",
          font: "inherit",
        }}
      >
        <span
          style={{
            flex: "none",
            width: 0,
            height: 0,
            borderLeft: "6px solid var(--muted)",
            borderTop: "4px solid transparent",
            borderBottom: "4px solid transparent",
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.15s",
          }}
        />
        <span style={{ fontSize: 14, fontWeight: 500 }}>{artifact.label || artifact.kind}</span>
        <span className="micro" style={{ marginLeft: "auto" }}>
          {artifact.kind}
          {artifact.bytes != null ? ` · ${fmtBytes(artifact.bytes)}` : ""}
        </span>
      </button>
      {open && (
        <div style={{ padding: "0 2px 16px" }}>
          <ArtifactContent artifact={artifact} />
        </div>
      )}
    </div>
  );
}

export function RawTab({ jobId }: { jobId: string }) {
  const [state, setState] = useState<
    { status: "loading" } | { status: "missing" } | { status: "error"; error: string } | { status: "ready"; manifest: Manifest }
  >({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fetch(serveUrl(`jobs/${jobId}/raw/manifest.json`), { cache: "no-store" })
      .then((r) => {
        // S3 presigned GETs return 403 for missing keys (not 404).
        if (r.status === 404 || r.status === 403) throw new Error("missing");
        if (!r.ok) throw new Error(`Failed to load manifest (${r.status})`);
        return r.json();
      })
      .then((manifest) => {
        if (!cancelled) setState({ status: "ready", manifest });
      })
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof Error && e.message === "missing") setState({ status: "missing" });
        else
          setState({
            status: "error",
            error: e instanceof Error ? e.message : "Failed to load manifest",
          });
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (state.status === "loading") {
    return (
      <div className="empty">
        <div className="row" style={{ gap: 8, justifyContent: "center" }}>
          <SpinnerIcon />
          <span>Loading raw artifacts…</span>
        </div>
      </div>
    );
  }

  if (state.status === "missing") {
    return <div className="empty">No raw artifacts — this job ran before raw logging shipped.</div>;
  }

  if (state.status === "error") {
    return <div className="empty error-text">{state.error}</div>;
  }

  const artifacts = state.manifest.artifacts ?? [];
  const videos = artifacts.filter((a) => a.kind === "debug_video");
  const jsons = artifacts.filter((a) => a.kind !== "debug_video");

  if (!artifacts.length) {
    return <div className="empty">No raw artifacts — this job ran before raw logging shipped.</div>;
  }

  return (
    <div>
      {videos.map((a, i) => (
        <div key={a.key || i} style={{ maxWidth: 640, margin: "0 auto 32px" }}>
          <video
            controls
            src={serveUrl(a.key)}
            style={{
              width: "100%",
              background: "#141414",
              borderRadius: 10,
              border: "2px solid var(--border)",
              display: "block",
            }}
          />
          <p className="micro" style={{ margin: "8px 0 0", textAlign: "center" }}>
            {a.label || "debug_video"}
            {a.bytes != null ? ` · ${fmtBytes(a.bytes)}` : ""}
          </p>
        </div>
      ))}
      {jsons.map((a, i) => (
        <ArtifactExpander key={a.key || i} artifact={a} />
      ))}
    </div>
  );
}
