import { NextResponse } from "next/server";
import { listLibrary } from "@/db";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const url = new URL(req.url);
  try {
    const outputs = await listLibrary({
      platform: url.searchParams.get("platform") || undefined,
      masterId: url.searchParams.get("masterId") || undefined,
      kind: url.searchParams.get("kind") || undefined,
    });
    return NextResponse.json({ outputs });
  } catch (e) {
    return NextResponse.json(
      { outputs: [], error: e instanceof Error ? e.message : "DB error" },
      { status: 200 }
    );
  }
}
