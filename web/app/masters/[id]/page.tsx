import Link from "next/link";
import { getMaster, listOutputsByMaster } from "@/db";
import { platformLabel } from "@/spec";

export const dynamic = "force-dynamic";

export default async function MasterPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const master = await getMaster(id).catch(() => null);
  const outputs = await listOutputsByMaster(id).catch(() => []);

  if (!master) {
    return (
      <>
        <h1>Master not found</h1>
        <Link className="btn ghost" href="/">
          Back to upload
        </Link>
      </>
    );
  }

  return (
    <>
      <h1>{master.title}</h1>
      <p className="sub">
        <span className="badge">{master.kind}</span>
        {master.width && master.height && (
          <span className="badge">
            {master.width}×{master.height}
          </span>
        )}
        {master.duration_sec != null && (
          <span className="badge">{master.duration_sec.toFixed(1)}s</span>
        )}
        <span className="badge">{new Date(master.created_at).toLocaleDateString()}</span>
      </p>

      <h2>Outputs ({outputs.length})</h2>
      {outputs.length === 0 ? (
        <div className="empty">No outputs yet for this master.</div>
      ) : (
        <div className="grid">
          {outputs.map(async (o) => (
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
                <span className="badge">{await platformLabel(o.platform)}</span>
                <span className="badge">{o.ratio}</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </>
  );
}
