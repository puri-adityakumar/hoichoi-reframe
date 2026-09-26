/**
 * Best-effort trigger for the Blaxel reframe worker.
 * Failures are swallowed on purpose: demo jobs are precomputed, so an
 * unreachable/unauthorized Blaxel must never fail the request.
 */
export async function triggerReframeWorker(jobId: string, masterId: string): Promise<boolean> {
  const workspace = process.env.BL_WORKSPACE;
  const apiKey = process.env.BL_API_KEY;
  if (!workspace || !apiKey) return false;
  try {
    const res = await fetch(
      "https://api.blaxel.ai/v0/jobs/reframe-worker/executions",
      {
        method: "POST",
        headers: {
          // Blaxel executions API requires X-Blaxel-Authorization (not Authorization)
          "X-Blaxel-Authorization": `Bearer ${apiKey}`,
          "X-Blaxel-Workspace": workspace,
          "Content-Type": "application/json",
        },
        // tasks array: each task's fields arrive as handler kwargs in the container
        body: JSON.stringify({ tasks: [{ job_id: jobId, master_id: masterId }] }),
        signal: AbortSignal.timeout(10_000),
      }
    );
    return res.ok;
  } catch {
    return false;
  }
}
