import Link from "next/link";
import { PreviewThumb } from "../../../components/PreviewThumb";
import { getMaster, listJobs, listOutputsByMaster } from "@/db";
import { platformLabel } from "@/spec";

export const dynamic = "force-dynamic";

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

function ClockIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 3" />
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
      style={{
        justifyContent: "space-between",
        gap: 16,
        padding: "10px 0",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <span className="micro">{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{value}</span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const failed = status === "failed";
  const done = status === "done" || status === "completed";
  return (
    <span className={`badge ${failed ? "fail" : done ? "pass" : "muted-badge"}`}>
      <span style={{ display: "inline-flex", verticalAlign: "-2px", marginRight: 5 }}>
        {failed ? <XIcon /> : done ? <CheckIcon /> : <ClockIcon />}
      </span>
      {status}
    </span>
  );
}

export default async function MasterPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const master = await getMaster(id).catch(() => null);
  const outputs = await listOutputsByMaster(id).catch(() => []);
  const jobs = await listJobs(200).catch(() => []);
  const masterJobs = jobs.filter((j) => j.master_id === id);

  if (!master) {
    return (
      <>
        <p className="micro" style={{ margin: "0 0 8px" }}>Master</p>
        <h1>Master not found</h1>
        <p style={{ marginTop: 24 }}>
          <Link className="btn ghost" href="/">
            Back to upload
          </Link>
        </p>
      </>
    );
  }

  return (
    <>
      <header className="page-head">
        <p className="micro" style={{ margin: "0 0 8px" }}>Master</p>
        <div className="row" style={{ gap: 12 }}>
          <h1 style={{ margin: 0 }}>{master.title}</h1>
          <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
            {master.id.slice(0, 8)}
          </span>
        </div>
      </header>

      <section style={{ margin: "24px 0 48px", maxWidth: 520 }}>
        <MicroRow label="Kind" value={master.kind} />
        {master.width && master.height && (
          <MicroRow label="Dimensions" value={`${master.width} × ${master.height}`} />
        )}
        {master.duration_sec != null && (
          <MicroRow label="Duration" value={`${master.duration_sec.toFixed(1)}s`} />
        )}
        <MicroRow label="Created" value={new Date(master.created_at).toLocaleDateString()} />
      </section>

      <section style={{ marginBottom: 48 }}>
        <p className="micro" style={{ margin: "0 0 6px" }}>Jobs</p>
        {masterJobs.length === 0 ? (
          <div className="empty">No jobs run on this master yet.</div>
        ) : (
          <div>
            {masterJobs.map((j) => (
              <Link
                key={j.id}
                href={`/jobs/${j.id}`}
                className="row"
                style={{
                  justifyContent: "space-between",
                  gap: 16,
                  padding: "12px 0",
                  borderBottom: "1px solid var(--border)",
                }}
              >
                <span style={{ fontSize: 14 }}>{j.id.slice(0, 8)}</span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                  <StatusBadge status={j.status} />
                  <span style={{ display: "inline-flex", color: "var(--muted)" }} title="Open job">
                    <ArrowUpRightIcon />
                  </span>
                </span>
              </Link>
            ))}
          </div>
        )}
      </section>

      <section>
        <p className="micro" style={{ margin: "0 0 14px" }}>Outputs · {outputs.length}</p>
        {outputs.length === 0 ? (
          <div className="empty">No outputs yet for this master.</div>
        ) : (
          <div className="grid">
            {outputs.map(async (o) => (
              <Link key={o.id} href={`/assets/${o.id}`} style={{ display: "block" }}>
                <PreviewThumb
                  previewKey={o.preview_key}
                  alt={`${await platformLabel(o.platform)} ${o.ratio}`}
                  emptyLabel="Preview unavailable"
                  style={{
                    width: "100%",
                    borderRadius: 8,
                    border: "2px solid var(--border)",
                    display: "block",
                    marginBottom: 10,
                  }}
                />
                <div className="row" style={{ gap: 8 }}>
                  <span className="micro">{await platformLabel(o.platform)} · {o.ratio}</span>
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
    </>
  );
}
