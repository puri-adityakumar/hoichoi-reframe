"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { JobListRow } from "@/db";

function UploadIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M8 10.5V2.5" />
      <path d="M4.5 6 8 2.5 11.5 6" />
      <path d="M2.5 11v2.5h11V11" />
    </svg>
  );
}

const PAGE_SIZE = 10;

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<JobListRow[]>([]);
  const [page, setPage] = useState(1);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const r = await fetch("/api/jobs");
        const d = r.ok ? await r.json() : { jobs: [] };
        if (!cancelled) setJobs(d.jobs ?? []);
        return d.jobs ?? [];
      } catch {
        if (!cancelled) setJobs([]);
        return [];
      }
    }
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const rows: JobListRow[] = await load();
      if (cancelled) return;
      const anyActive = rows.some(
        (j) => j.status === "queued" || j.status === "running"
      );
      // Poll every 4s while any job is queued/running; once none are active,
      // fall back to a slow 30s refresh so the list stays fresh.
      const delay = anyActive ? 4000 : 30000;
      timer = setTimeout(tick, delay);
    };
    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  async function readDims(f: File) {
    if (f.type.startsWith("image/")) {
      const url = URL.createObjectURL(f);
      try {
        const img = new Image();
        const dims = await new Promise<{ width: number; height: number }>(
          (res, rej) => {
            img.onload = () => res({ width: img.naturalWidth, height: img.naturalHeight });
            img.onerror = rej;
            img.src = url;
          }
        );
        return dims;
      } finally {
        URL.revokeObjectURL(url);
      }
    }
    return {};
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!file) {
      setError("Pick a file first.");
      return;
    }
    setBusy(true);
    try {
      const dims = await readDims(file).catch(() => ({}));
      const res = await fetch("/api/uploads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: title.trim() || file.name,
          filename: file.name,
          size: file.size,
          contentType: file.type,
          ...dims,
        }),
      });
      if (!res.ok) throw new Error(`Upload init failed (${res.status})`);
      const { masterId, uploadUrl } = (await res.json()) as {
        masterId: string;
        uploadUrl: string;
      };
      const put = await fetch(uploadUrl, {
        method: "PUT",
        body: file,
        headers: { "Content-Type": file.type },
      });
      if (!put.ok) throw new Error(`File upload failed (${put.status})`);
      const jobRes = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ masterId }),
      });
      if (!jobRes.ok) throw new Error(`Job creation failed (${jobRes.status})`);
      const { jobId } = (await jobRes.json()) as { jobId: string };
      router.push(`/jobs/${jobId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setBusy(false);
    }
  }

  const activeCount = jobs.filter(
    (j) => j.status === "queued" || j.status === "running"
  ).length;
  const totalPages = Math.max(1, Math.ceil(jobs.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pageRows = jobs.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  return (
    <>
      <section>
        <div
          style={{
            background: "var(--panel-2)",
            border: "2px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: 24,
            marginBottom: 8,
          }}
        >
          <p className="card-label">Upload a video / image for the processing pipeline</p>
          <form onSubmit={submit}>
          <div className="row">
            <label className="field" style={{ flex: "1 1 220px", marginBottom: 0 }}>
              <span>Title</span>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Diwali promo scene 3"
              />
            </label>
            <label className="field file-field" style={{ marginBottom: 0 }}>
              <span>Master file — image or video</span>
              <input
                type="file"
                accept="image/*,video/*"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>
          </div>
          <div style={{ marginTop: 20 }}>
            <button type="submit" disabled={busy || !file}>
              {busy ? (
                "Working…"
              ) : (
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <UploadIcon />
                  Upload &amp; run
                </span>
              )}
            </button>
          </div>
          {error && (
            <p className="error-text" style={{ marginTop: 12 }}>
              {error}
            </p>
          )}
          </form>
        </div>
      </section>

      <section>
        <div className="row" style={{ margin: "40px 0 12px" }}>
          <p className="micro" style={{ margin: 0 }}>Jobs</p>
          <span className="muted" style={{ marginLeft: "auto", fontFamily: "var(--font-mono)", fontSize: 11 }}>
            {jobs.length === 0 ? "0" : `${jobs.length}${activeCount > 0 ? ` · ${activeCount} active` : ""}`}
          </span>
        </div>
        {jobs.length === 0 ? (
          <div className="empty">No jobs yet. Upload a master to run one.</div>
        ) : (
          <>
            <div className="job-grid" style={{ borderTop: "1px solid var(--border)" }}>
              {pageRows.map((j, i) => {
                const dt = new Date(j.created_at);
                return (
                  <Link href={`/jobs/${j.id}`} className="job-grid-row" key={j.id}>
                    <span className="job-slno">
                      {(safePage - 1) * PAGE_SIZE + i + 1}
                    </span>
                    <span style={{ minWidth: 0 }}>
                      <strong className="job-title" style={{ display: "block" }}>
                        {j.master_title ?? "Untitled master"}
                      </strong>
                      <span
                        className="muted"
                        style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
                      >
                        {j.id.slice(0, 8)} · {j.master_kind ?? "master"}
                        {j.status === "failed" && j.error
                          ? ` · ${j.error.length > 60 ? `${j.error.slice(0, 60)}…` : j.error}`
                          : ""}
                      </span>
                    </span>
                    <span className={`job-status ${j.status}`}>
                      {j.status}
                      {j.status === "running" && (
                        <span className="muted" style={{ display: "block", fontSize: 11 }}>
                          {j.stage ?? "working"} · {j.progress ?? 0}%
                        </span>
                      )}
                    </span>
                    <span
                      className="muted"
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: 12,
                        textAlign: "right",
                      }}
                    >
                      {dt.toLocaleDateString(undefined, { month: "short", day: "numeric" })}{" "}
                      {dt.toLocaleTimeString(undefined, {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </Link>
                );
              })}
            </div>
            {totalPages > 1 && (
              <div className="job-pages">
                <button disabled={safePage <= 1} onClick={() => setPage(safePage - 1)}>
                  ‹ prev
                </button>
                <span>
                  page {safePage} / {totalPages}
                </span>
                <button
                  disabled={safePage >= totalPages}
                  onClick={() => setPage(safePage + 1)}
                >
                  next ›
                </button>
              </div>
            )}
          </>
        )}
      </section>
    </>
  );
}
