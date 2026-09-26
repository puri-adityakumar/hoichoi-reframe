import { NextResponse } from "next/server";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import { createMaster } from "@/db";
import { presignPut } from "@/s3";

const bodySchema = z.object({
  title: z.string().min(1).max(200),
  filename: z.string().min(1).max(300),
  size: z.number().int().positive(),
  contentType: z.string().min(1).max(150),
  width: z.number().int().positive().optional().nullable(),
  height: z.number().int().positive().optional().nullable(),
  durationSec: z.number().positive().optional().nullable(),
});

export async function POST(req: Request) {
  let parsedBody: unknown;
  try {
    parsedBody = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const parsed = bodySchema.safeParse(parsedBody);
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
  }
  const b = parsed.data;

  const kind = b.contentType.startsWith("video/") ? "video" : "image";
  const key = `masters/${randomUUID()}/${b.filename.replace(/[^\w.\-]+/g, "_")}`;

  let master;
  try {
    master = await createMaster({
      title: b.title,
      kind,
      s3Key: key,
      sizeBytes: b.size,
      width: b.width ?? null,
      height: b.height ?? null,
      durationSec: b.durationSec ?? null,
      fps: null,
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "DB error" },
      { status: 500 }
    );
  }

  try {
    const uploadUrl = await presignPut(key, 3600);
    return NextResponse.json({ masterId: master.id, uploadUrl, s3Key: key });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "S3 presign error" },
      { status: 500 }
    );
  }
}
