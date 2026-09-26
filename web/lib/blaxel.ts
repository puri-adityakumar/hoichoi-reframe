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
          Authorization: `Bearer ${apiKey}`,
          "X-Blaxel-Workspace": workspace,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: { job_id: jobId, master_id: masterId } }),
        signal: AbortSignal.timeout(10_000),
      }
    );
    return res.ok;
  } catch {
    return false;
  }
}
