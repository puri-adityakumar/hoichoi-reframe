import Link from "next/link";
import { listLibrary } from "@/db";
import { platformLabel } from "@/spec";

export const dynamic = "force-dynamic";

export default async function LibraryPage({
  searchParams,
}: {
  searchParams: Promise<{ platform?: string; masterId?: string }>;
}) {
  const sp = await searchParams;
  let rows: Awaited<ReturnType<typeof listLibrary>>;
  let error: string | null = null;
  try {
    rows = await listLibrary({
      platform: sp.platform || undefined,
      masterId: sp.masterId || undefined,
    });
  } catch {
    rows = [];
    error = "Could not load the library. Check DATABASE_URL.";
  }

  const labels = await Promise.all(
    rows.map(async (r) => ({ ...r, label: await platformLabel(r.platform) }))
  );

  return (
    <>
      <h1>Library</h1>
      <p className="sub">Every output is validated and traceable to its master.</p>

      <form className="filters" method="get">
        <select name="platform" defaultValue={sp.platform ?? ""}>
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
        <select name="masterId" defaultValue={sp.masterId ?? ""}>
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

      <div className="grid">
        {labels.map((o) => (
          <Link className="card" key={o.id} href={`/assets/${o.id}`}>
            {o.preview_key ? (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={`/api/serve?key=${encodeURIComponent(o.preview_key)}`}
                alt={o.platform}
                style={{ width: "100%", borderRadius: 8, marginBottom: 10 }}
              />
            ) : (
              <div className="preview">
                <img src="/no-preview.png" alt="" style={{ width: "100%", borderRadius: 8, marginBottom: 10, display: "none" }} />
              </div>
            )}
            <div className="row">
              <span className="badge">{o.label}</span>
              <span className="badge">{o.ratio}</span>
              <span className={`badge ${o.all_passed ? "pass" : "fail"}`}>
                {o.all_passed ? "pass" : "fail"}
              </span>
            </div>
            <p className="muted" style={{ marginTop: 8 }}>
              {o.master_title ?? "Untitled master"} · {o.kind} ·{" "}
              {new Date(o.created_at).toLocaleDateString()}
            </p>
          </Link>
        ))}
      </div>
    </>
  );
}
