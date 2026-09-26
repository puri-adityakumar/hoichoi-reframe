import { readFile } from "node:fs/promises";
import path from "node:path";

export type PlatformSpec = {
  label: string;
  kind: string;
  ratio: string;
  width: number;
  height: number;
  max_duration_sec?: number;
  max_size_mb?: number;
  codec?: string;
  container?: string;
  min_fps?: number;
};

export type Spec = { version: string; platforms: Record<string, PlatformSpec> };

let cached: Spec | null = null;

/**
 * Loaded at runtime from ../spec/spec.json relative to the project cwd.
 * For Vercel, `outputFileTracingIncludes` in next.config.ts ships the file
 * with the traced bundle, so the relative path resolves in serverless too.
 */
export async function loadSpec(): Promise<Spec> {
  if (cached) return cached;
  const file = path.join(process.cwd(), "..", "spec", "spec.json");
  cached = JSON.parse(await readFile(file, "utf8")) as Spec;
  return cached;
}

export async function platformLabel(platform: string): Promise<string> {
  try {
    const spec = await loadSpec();
    return spec.platforms[platform]?.label ?? platform;
  } catch {
    return platform;
  }
}
