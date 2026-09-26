import { NextResponse } from "next/server";
import { getJob, getMaster, listOutputsByJob } from "@/db";

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ id: string }> }
) {
  const { id } = await ctx.params;
  const job = await getJob(id).catch(() => null);
  if (!job) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }
  try {
    const [outputs, master] = await Promise.all([
      listOutputsByJob(id),
      getMaster(job.master_id).catch(() => null),
    ]);
    return NextResponse.json({
      job: { ...job, master_title: master?.title ?? null },
      outputs,
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "DB error" },
      { status: 500 }
    );
  }
}
