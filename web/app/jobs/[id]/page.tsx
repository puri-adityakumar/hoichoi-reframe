"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

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
type JobData = {
  job: {
    id: string;
    status: string;
    stage: string | null;
    progress: number | null;
    error: string | null;
  };
  outputs: Output[];
};

export default function JobPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [data, setData] = useState<JobData | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <>
      <h1>Job {job.id.slice(0, 8)}</h1>
      <p className="sub">
        Status: <strong>{job.status}</strong>
        {job.stage ? ` · ${job.stage}` : ""}
      </p>

      {!done && !failed && (
        <div className="card">
          <div className="progress">
            <div style={{ width: `${Math.round(job.progress ?? 0)}%` }} />
          </div>
          <p className="muted">{job.stage ?? "Working"}… {Math.round(job.progress ?? 0)}%</p>
        </div>
      )}

      {failed && (
        <div className="card">
          <p className="error-text">{job.error ?? "Job failed."}</p>
        </div>
      )}

      {done && (
        <>
          <h2>Outputs ({outputs.length})</h2>
          {outputs.length === 0 ? (
            <div className="empty">No outputs recorded for this job.</div>
          ) : (
            <div className="grid">
              {outputs.map((o) => (
                <Link className="card" key={o.id} href={`/assets/${o.id}`}>
                  {o.preview_key && (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      src={`/api/serve?key=${encodeURIComponent(o.preview_key)}`}
                      alt={o.platform}
                      style={{ width: "100%", borderRadius: 8, marginBottom: 10 }}
                    />
                  )}
                  <div className="row">
                    <span className="badge">{o.platform}</span>
                    <span className="badge">{o.ratio}</span>
                  </div>
                </Link>
              ))}
            </div>
          )}
          <p style={{ marginTop: 20 }}>
            <Link className="btn ghost" href="/library">
              Go to library
            </Link>
          </p>
        </>
      )}
    </>
  );
}
