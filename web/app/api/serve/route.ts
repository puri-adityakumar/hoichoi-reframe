import { NextResponse } from "next/server";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { bucket, s3 } from "@/s3";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const key = new URL(req.url).searchParams.get("key");
  if (!key) {
    return NextResponse.json({ error: "key query param required" }, { status: 400 });
  }
  // Path-traversal guard: only serve objects under known prefixes.
  // jobs/<uuid>/raw/ is the worker's QA-artifact area (analysis JSON, VLM
  // call log, tracking debug video); nothing else under jobs/ is servable.
  const rawOk = /^jobs\/[0-9a-f-]{36}\/raw\//.test(key);
  if (!rawOk && !key.startsWith("masters/") && !key.startsWith("outputs/")) {
    return NextResponse.json({ error: "forbidden key" }, { status: 403 });
  }
  // No ContentType override here on purpose: this endpoint points the browser
  // straight at object storage, and Neon ignores `response-content-type` on a
  // presigned GET, so the header it returns is whatever the object was stored
  // with. The object must therefore be written with the right ContentType --
  // see storage.upload_file and scripts/backfill_content_types.py.
  try {
    const url = await getSignedUrl(
      s3(),
      new GetObjectCommand({ Bucket: bucket(), Key: key }),
      { expiresIn: 600 }
    );
    return NextResponse.redirect(url, 302);
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "S3 error" },
      { status: 500 }
    );
  }
}
