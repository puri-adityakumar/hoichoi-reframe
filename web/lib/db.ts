import { Pool, type PoolClient } from "pg";

declare global {
  // eslint-disable-next-line no-var
  var __pgPool: Pool | undefined;
}

function pool(): Pool {
  if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is not set");
  }
  if (!global.__pgPool) {
    global.__pgPool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 5,
    });
  }
  return global.__pgPool;
}

export type Master = {
  id: string;
  title: string;
  kind: string;
  s3_key: string;
  size_bytes: string | number | null;
  width: number | null;
  height: number | null;
  duration_sec: number | null;
  fps: number | null;
  created_at: string;
};

export type Job = {
  id: string;
  master_id: string;
  status: string;
  stage: string | null;
  progress: number | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
};

export type Output = {
  id: string;
  job_id: string;
  master_id: string;
  platform: string;
  ratio: string;
  kind: string;
  s3_key: string;
  preview_key: string | null;
  speaker_on_screen_pct: number | null;
  created_at: string;
};

export type Validation = {
  output_id: string;
  rule: string;
  expected: string | null;
  actual: string | null;
  passed: boolean;
};

export type Decision = {
  job_id: string | null;
  output_id: string | null;
  stage: string;
  choice: string;
  reason: string | null;
  confidence: number | null;
  created_at: string;
};

async function q<T>(sql: string, params: unknown[] = []): Promise<T[]> {
  const client: PoolClient = await pool().connect();
  try {
    const res = await client.query(sql, params as never[]);
    return res.rows as T[];
  } finally {
    client.release();
  }
}

export async function createMaster(m: {
  title: string;
  kind: string;
  s3Key: string;
  sizeBytes: number | null;
  width: number | null;
  height: number | null;
  durationSec: number | null;
  fps: number | null;
}): Promise<Master> {
  const rows = await q<Master>(
    `insert into masters (title, kind, s3_key, size_bytes, width, height, duration_sec, fps)
     values ($1,$2,$3,$4,$5,$6,$7,$8)
     returning id, title, kind, s3_key, size_bytes, width, height, duration_sec, fps, created_at`,
    [m.title, m.kind, m.s3Key, m.sizeBytes, m.width, m.height, m.durationSec, m.fps]
  );
  return rows[0];
}

export async function listMasters(limit = 50): Promise<Master[]> {
  return q<Master>(
    `select id, title, kind, s3_key, size_bytes, width, height, duration_sec, fps, created_at
     from masters order by created_at desc limit $1`,
    [limit]
  );
}

export async function getMaster(id: string): Promise<Master | null> {
  const rows = await q<Master>(
    `select id, title, kind, s3_key, size_bytes, width, height, duration_sec, fps, created_at
     from masters where id = $1`,
    [id]
  );
  return rows[0] ?? null;
}

export async function createJob(masterId: string): Promise<Job> {
  const rows = await q<Job>(
    `insert into jobs (master_id, status, progress) values ($1, 'queued', 0)
     returning id, master_id, status, stage, progress, error, created_at, finished_at`,
    [masterId]
  );
  return rows[0];
}

export async function getJob(id: string): Promise<Job | null> {
  const rows = await q<Job>(
    `select id, master_id, status, stage, progress, error, created_at, finished_at
     from jobs where id = $1`,
    [id]
  );
  return rows[0] ?? null;
}

export type JobListRow = Job & {
  master_title: string | null;
  master_kind: string | null;
  output_count: number;
  passed_count: number;
};

export async function listJobs(limit = 20): Promise<JobListRow[]> {
  return q<JobListRow>(
    `select j.id, j.master_id, j.status, j.stage, j.progress, j.error, j.created_at, j.finished_at,
            m.title as master_title, m.kind as master_kind,
            coalesce(o.output_count, 0) as output_count,
            coalesce(o.passed_count, 0) as passed_count
     from jobs j
     left join masters m on m.id = j.master_id
     left join (
       select job_id,
              count(*) as output_count,
              count(*) filter (where coalesce(v.all_passed, true)) as passed_count
       from outputs
       left join (
         select output_id, bool_and(passed) as all_passed from validations group by output_id
       ) v on v.output_id = outputs.id
       group by job_id
     ) o on o.job_id = j.id
     order by j.created_at desc
     limit $1`,
    [limit]
  );
}

export async function listOutputsByJob(jobId: string): Promise<Output[]> {
  return q<Output>(
    `select id, job_id, master_id, platform, ratio, kind, s3_key, preview_key,
            speaker_on_screen_pct, created_at
     from outputs where job_id = $1 order by created_at asc`,
    [jobId]
  );
}

export async function listOutputsByMaster(masterId: string): Promise<Output[]> {
  return q<Output>(
    `select id, job_id, master_id, platform, ratio, kind, s3_key, preview_key,
            speaker_on_screen_pct, created_at
     from outputs where master_id = $1 order by created_at asc`,
    [masterId]
  );
}

export type LibraryRow = Output & {
  master_title: string | null;
  master_kind: string | null;
  all_passed: boolean;
};

export async function listLibrary(filter?: {
  platform?: string;
  masterId?: string;
  kind?: string;
}): Promise<LibraryRow[]> {
  const where: string[] = [];
  const params: unknown[] = [];
  if (filter?.platform) {
    params.push(filter.platform);
    where.push(`o.platform = $${params.length}`);
  }
  if (filter?.masterId) {
    params.push(filter.masterId);
    where.push(`o.master_id = $${params.length}`);
  }
  if (filter?.kind) {
    params.push(filter.kind);
    where.push(`m.kind = $${params.length}`);
  }
  const whereSql = where.length ? `where ${where.join(" and ")}` : "";
  return q<LibraryRow>(
    `select o.id, o.job_id, o.master_id, o.platform, o.ratio, o.kind, o.s3_key,
            o.preview_key, o.speaker_on_screen_pct, o.created_at,
            m.title as master_title,
            m.kind as master_kind,
            coalesce(v.all_passed, true) as all_passed
     from outputs o
     left join masters m on m.id = o.master_id
     left join (
       select output_id, bool_and(passed) as all_passed from validations group by output_id
     ) v on v.output_id = o.id
     ${whereSql}
     order by o.created_at desc
     limit 200`,
    params
  );
}

export type OutputDetail = {
  output: Output;
  master: Master | null;
  validations: Validation[];
  decisions: Decision[];
};

export async function getOutputDetail(id: string): Promise<OutputDetail | null> {
  const outs = await q<Output>(
    `select id, job_id, master_id, platform, ratio, kind, s3_key, preview_key,
            speaker_on_screen_pct, created_at
     from outputs where id = $1`,
    [id]
  );
  const output = outs[0];
  if (!output) return null;
  const [master, validations, decisions] = await Promise.all([
    getMaster(output.master_id),
    q<Validation>(
      `select output_id, rule, expected, actual, passed from validations
       where output_id = $1 order by id asc`,
      [id]
    ),
    q<Decision>(
      `select job_id, output_id, stage, choice, reason, confidence, created_at
       from decisions where output_id = $1 or (output_id is null and job_id = $2)
       order by created_at asc`,
      [id, output.job_id]
    ),
  ]);
  return { output, master, validations, decisions };
}
