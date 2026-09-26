"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { Master } from "@/db";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [masters, setMasters] = useState<Master[]>([]);

  useEffect(() => {
    fetch("/api/masters")
      .then((r) => (r.ok ? r.json() : { masters: [] }))
      .then((d) => setMasters(d.masters ?? []))
      .catch(() => setMasters([]));
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

  async function runJob(masterId: string) {
    setError(null);
    setBusy(true);
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ masterId }),
      });
      if (!res.ok) throw new Error(`Job creation failed (${res.status})`);
      const { jobId } = (await res.json()) as { jobId: string };
      router.push(`/jobs/${jobId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setBusy(false);
    }
  }

  return (
    <>
      <h1>Upload a master</h1>
      <p className="sub">
        One image or video in — every platform version comes out, validated and traceable.
      </p>

      <form className="card" onSubmit={submit}>
        <label className="field">
          <span>Title</span>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Diwali promo scene 3"
          />
        </label>
        <label className="field">
          <span>Master file (image or video)</span>
          <input
            type="file"
            accept="image/*,video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <button type="submit" disabled={busy || !file}>
          {busy ? "Working…" : "Upload & run"}
        </button>
        {error && <p className="error-text">{error}</p>}
      </form>

      <h2>Sample masters</h2>
      {masters.length === 0 ? (
        <div className="empty">No masters yet. Upload one above.</div>
      ) : (
        <div className="grid">
          {masters.map((m) => (
            <div className="card" key={m.id}>
              <div className="row">
                <strong>{m.title}</strong>
                <span className="badge">{m.kind}</span>
              </div>
              <p className="muted">
                {new Date(m.created_at).toLocaleDateString()} ·{" "}
                {m.width && m.height ? `${m.width}×${m.height}` : "size unknown"}
              </p>
              <button className="ghost" disabled={busy} onClick={() => runJob(m.id)}>
                Run job
              </button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
