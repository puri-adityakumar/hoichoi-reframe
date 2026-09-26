import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  serverExternalPackages: ["pg"],
  // Ship spec.json into traced server bundles (works on Vercel too).
  outputFileTracingIncludes: {
    "/**": [path.join(__dirname, "..", "spec", "spec.json")],
  },
};

export default nextConfig;
