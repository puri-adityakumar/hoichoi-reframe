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
  if (!key.startsWith("masters/") && !key.startsWith("outputs/")) {
    return NextResponse.json({ error: "forbidden key" }, { status: 403 });
  }
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
