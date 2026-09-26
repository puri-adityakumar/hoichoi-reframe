import Link from "next/link";
import { listLibrary } from "@/db";
import { PreviewThumb } from "../../components/PreviewThumb";
import { platformLabel } from "@/spec";

export const dynamic = "force-dynamic";

function CheckIcon() {
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
      aria-hidden
      style={{ color: "var(--ok)" }}
    >
      <path d="M3 8.5l3.5 3.5L13 4.5" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      aria-hidden
      style={{ color: "var(--fail)" }}
    >
      <path d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

function ArrowUpRightIcon() {
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
      aria-hidden
    >
      <path d="M4.5 11.5l7-7M6 4.5h5.5V10" />
    </svg>
  );
}

export default async function LibraryPage({
  searchParams,
}: {
  searchParams: Promise<{ platform?: string; masterId?: string; kind?: string; page?: string }>;
}) {
  const sp = await searchParams;
  let rows: Awaited<ReturnType<typeof listLibrary>>;
  let error: string | null = null;
  try {
    rows = await listLibrary({
      platform: sp.platform || undefined,
      masterId: sp.masterId || undefined,
      kind: sp.kind || undefined,
    });
  } catch {
    rows = [];
    error = "Could not load the library. Check DATABASE_URL.";
  }

  const labels = await Promise.all(
    rows.map(async (r) => ({ ...r, label: await platformLabel(r.platform) }))
  );

  // Group outputs by job. Rows are already ordered created_at desc, so Map
  // insertion order gives most recent job first, and within-group card order
  // is preserved.
  const groups: { jobId: string; outputs: typeof labels }[] = [];
  const byJob = new Map<string, typeof labels>();
  for (const o of labels) {
    const group = byJob.get(o.job_id);
    if (group) {
      group.push(o);
    } else {
      const fresh: typeof labels = [o];
      byJob.set(o.job_id, fresh);
      groups.push({ jobId: o.job_id, outputs: fresh });
    }
  }

  // Gallery cards: 10 job cards per page, paginated server-side via ?page=.
  const PAGE_SIZE = 10;
  const totalPages = Math.max(1, Math.ceil(groups.length / PAGE_SIZE));
  const pageNum = Math.min(
    Math.max(parseInt(sp.page ?? "1", 10) || 1, 1),
    totalPages
  );
  const pageGroups = groups.slice((pageNum - 1) * PAGE_SIZE, pageNum * PAGE_SIZE);
  const pageHref = (p: number) => {
    const q = new URLSearchParams();
    if (sp.kind) q.set("kind", sp.kind);
    if (sp.platform) q.set("platform", sp.platform);
    if (sp.masterId) q.set("masterId", sp.masterId);
    q.set("page", String(p));
    return `/library?${q.toString()}`;
  };

  return (
    <>
      <form className="filters" method="get">
        <select name="kind" defaultValue={sp.kind ?? ""} aria-label="Type">
          <option value="">All types</option>
          <option value="image">Image</option>
          <option value="video">Video</option>
        </select>
        <select name="platform" defaultValue={sp.platform ?? ""} aria-label="Platform">
          <option value="">All platforms</option>
          <option value="instagram_reel">Instagram Reels / Shorts</option>
          <option value="instagram_feed">Instagram Feed</option>
          <option value="youtube_square">Square social post</option>
          <option value="youtube_landscape">YouTube landscape</option>
          <option value="youtube_thumbnail">YouTube thumbnail</option>
          <option value="feed_image">Feed image</option>
          <option value="story_image">Story image</option>
          <option value="square_image">Square image</option>
          <option value="landscape_image">Landscape image</option>
        </select>
        <select name="masterId" defaultValue={sp.masterId ?? ""} aria-label="Master">
          <option value="">All masters</option>
          {rows
            .filter((r, i, a) => a.findIndex((x) => x.master_id === r.master_id) === i)
            .map((r) => (
              <option key={r.master_id} value={r.master_id}>
                {r.master_title ?? r.master_id.slice(0, 8)}
              </option>
            ))}
        </select>
        <button className="ghost" type="submit">
          Filter
        </button>
        <Link className="btn ghost" href="/library">
          Clear
        </Link>
      </form>

      {error && <div className="empty error-text">{error}</div>}
      {!error && labels.length === 0 && (
        <div className="empty">No outputs yet. Run a job to populate the library.</div>
      )}

      <div className="lib-grid">
        {pageGroups.map((g) => {
          // One lead thumbnail per job — the first output — rather than a strip
          // of every ratio. The job page still lists them all.
          const first = g.outputs[0];
          const passed = g.outputs.filter((o) => o.all_passed).length;
          return (
            <article className="lib-card" key={g.jobId}>
              <div className="lib-card-head">
                <h2>{first.master_title ?? "Untitled master"}</h2>
                <span className="badge">{first.master_kind ?? first.kind}</span>
                <span className="badge muted-badge">{g.jobId.slice(0, 8)}</span>
                <span className="badge muted-badge">
                  {passed}/{g.outputs.length} pass
                  {passed < g.outputs.length ? ` · ${g.outputs.length - passed} fail` : ""}
                </span>
              </div>

              <Link
                className="lib-thumb"
                href={`/assets/${first.id}`}
                title={`${first.label} · ${first.ratio} · ${first.all_passed ? "pass" : "fail"}`}
              >
                <PreviewThumb
                  previewKey={first.preview_key}
                  alt={`${first.platform} ${first.ratio}`}
                />
              </Link>

              <div className="lib-card-foot">
                <span className="micro lib-grow">
                  {first.label} · {first.ratio}
                </span>
                <span
                  className="muted"
                  aria-label={first.all_passed ? "Validation passed" : "Validation failed"}
                >
                  {first.all_passed ? <CheckIcon /> : <XIcon />}
                </span>
              </div>

              <div className="lib-card-meta">
                <span className="muted">
                  {new Date(first.created_at).toLocaleDateString()} · {g.outputs.length}{" "}
                  {g.outputs.length === 1 ? "output" : "outputs"}
                </span>
                <Link className="group-link" href={`/jobs/${g.jobId}`}>
                  <ArrowUpRightIcon />
                  View job
                </Link>
              </div>
            </article>
          );
        })}
      </div>

      {totalPages > 1 && (
        <div className="job-pages">
          {pageNum > 1 ? (
            <Link href={pageHref(pageNum - 1)} style={{ textDecoration: "none" }}>
              ‹ prev
            </Link>
          ) : (
            <span style={{ opacity: 0.4 }}>‹ prev</span>
          )}
          <span>
            page {pageNum} / {totalPages}
          </span>
          {pageNum < totalPages ? (
            <Link href={pageHref(pageNum + 1)} style={{ textDecoration: "none" }}>
              next ›
            </Link>
          ) : (
            <span style={{ opacity: 0.4 }}>next ›</span>
          )}
        </div>
      )}
    </>
  );
}
