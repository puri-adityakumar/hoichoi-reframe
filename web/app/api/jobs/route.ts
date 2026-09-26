import { NextResponse } from "next/server";
import { createJob } from "@/db";
import { triggerReframeWorker } from "@/blaxel";

export async function POST(req: Request) {
  let masterId: string | undefined;
  try {
    const body = await req.json();
    masterId = body?.masterId ?? body?.master_id;
  } catch {
    /* fall through */
  }
  if (!masterId || typeof masterId !== "string") {
    return NextResponse.json({ error: "masterId is required" }, { status: 400 });
  }

  let job;
  try {
    job = await createJob(masterId);
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "DB error" },
      { status: 500 }
    );
  }

  // Best-effort trigger; demo jobs are precomputed so failure keeps the job queued.
  await triggerReframeWorker(job.id, masterId);

  return NextResponse.json({ jobId: job.id, status: job.status });
}
