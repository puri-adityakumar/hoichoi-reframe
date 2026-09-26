import Link from "next/link";
import { getOutputDetail } from "@/db";
import { platformLabel } from "@/spec";

export const dynamic = "force-dynamic";

const iconProps = {
  width: 15,
  height: 15,
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

function ArrowUpRightIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <path d="M7 17L17 7M9 7h8v8" />
    </svg>
  );
}

function MicroRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="row"
      style={{ justifyContent: "space-between", gap: 16, padding: "10px 0", borderBottom: "1px solid var(--border)" }}
    >
      <span className="micro">{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{value}</span>
    </div>
  );
}

export default async function AssetPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const detail = await getOutputDetail(id).catch(() => null);

  if (!detail) {
    return (
      <>
        <p className="micro" style={{ margin: "0 0 8px" }}>Asset</p>
        <h1>Asset not found</h1>
        <p style={{ marginTop: 24 }}>
          <Link className="btn ghost" href="/library">
            Back to library
          </Link>
        </p>
      </>
    );
  }

  const { output, master, validations, decisions } = detail;
  const allPassed = validations.length > 0 && validations.every((v) => v.passed);
  const isVideo = output.kind === "reel" || output.s3_key.endsWith(".mp4");
  const label = await platformLabel(output.platform);
  const isVertical = output.ratio === "9:16";

  return (
    <>
      <header className="page-head">
        <p className="micro" style={{ margin: "0 0 8px" }}>Asset</p>
        <div className="row" style={{ gap: 12 }}>
          <h1 style={{ margin: 0 }}>{label}</h1>
          <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
            {output.id.slice(0, 8)}
          </span>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <span className={`badge ${allPassed ? "pass" : "fail"}`}>
            <span style={{ display: "inline-flex", verticalAlign: "-2px", marginRight: 5 }}>
              {allPassed ? <CheckIcon /> : <XIcon />}
            </span>
            {allPassed ? "all checks passed" : "validation failures"}
          </span>
        </div>
      </header>

      <section style={{ margin: "48px 0", display: "flex", justifyContent: "center" }}>
        <div
          style={{
            width: "100%",
            maxWidth: isVideo && isVertical ? 420 : 896,
            background: "#141414",
            borderRadius: "var(--radius)",
            border: "2px solid var(--border)",
            overflow: "hidden",
            lineHeight: 0,
          }}
        >
          {isVideo ? (
            <video
              controls
              src={`/api/serve?key=************************************`}
              style={{ width: "100%", display: "block" }}
            />
          ) : (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={`/api/serve?key=************************************`}
              alt={label}
              style={{ width: "100%", display: "block" }}
            />
          )}
        </div>
      </section>

      {output.speaker_on_screen_pct != null && (
        <section style={{ marginBottom: 48 }}>
          <p className="micro" style={{ margin: "0 0 10px" }}>Speaker on screen</p>
          <div className="progress">
            <div style={{ width: `${Math.min(100, output.speaker_on_screen_pct)}%` }} />
          </div>
          <p className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12, margin: "8px 0 0" }}>
            {output.speaker_on_screen_pct.toFixed(1)}% of the time, the active speaker stays in frame.
          </p>
        </section>
      )}

      <section style={{ marginBottom: 48 }}>
        <p className="micro" style={{ margin: "0 0 6px" }}>Details</p>
        <MicroRow label="Ratio" value={output.ratio} />
        <MicroRow label="Platform" value={label} />
        <MicroRow label="Kind" value={output.kind} />
        {master && <MicroRow label="Master" value={master.title} />}
        <MicroRow label="Created" value={new Date(output.created_at).toLocaleString()} />
      </section>

      <section style={{ marginBottom: 48 }}>
        <p className="micro" style={{ margin: "0 0 6px" }}>Spec checks</p>
        {validations.length === 0 ? (
          <div className="empty">No validations recorded.</div>
        ) : (
          <div>
            {validations.map((v, i) => (
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
                <span className="row" style={{ gap: 10, fontFamily: "var(--font-mono)", fontSize: 12 }}>
                  <span className="muted">{v.expected ?? "—"}</span>
                  <span className="muted">→</span>
                  <span>{v.actual ?? "—"}</span>
                </span>
                <span
                  style={{ display: "inline-flex", flex: "none", color: v.passed ? "var(--ok)" : "var(--fail)" }}
                  title={v.passed ? "Passed" : "Failed"}
                  aria-label={v.passed ? "Passed" : "Failed"}
                >
                  {v.passed ? <CheckIcon /> : <XIcon />}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      {decisions.length > 0 && (
        <section style={{ marginBottom: 48 }}>
          <p className="micro" style={{ margin: "0 0 6px" }}>Decisions</p>
          <div>
            {decisions.map((d, i) => (
              <div key={i} style={{ padding: "14px 0", borderBottom: "1px solid var(--border)" }}>
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
        </section>
      )}

      <p style={{ marginTop: 40 }}>
        <Link
          href={`/masters/${output.master_id}`}
          className="group-link"
          style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
        >
          {master ? `Back to ${master.title}` : "Back to master"}
          <ArrowUpRightIcon />
        </Link>
      </p>
    </>
  );
}
