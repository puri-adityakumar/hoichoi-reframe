"""Load P4/.env.local into os.environ + a dict. Never log secret values."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_P4_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = Path(os.environ.get("P4_ENV_FILE", _P4_ROOT / ".env.local"))


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def load_config(env_path: Path | None = None, apply_env: bool = True) -> dict[str, str]:
    """Parse env file; optionally export into os.environ (existing env wins)."""
    vals = parse_env_file(env_path or ENV_PATH)
    if apply_env:
        for k, v in vals.items():
            os.environ.setdefault(k, v)
    return dict(vals)


def masked(cfg: dict[str, Any]) -> dict[str, Any]:
    """Safe-to-print view: hide values of keys that look secret."""
    secrets = ("KEY", "TOKEN", "SECRET", "PASSWORD", "DSN", "URI", "URL")
    return {
        k: ("***" if any(s in k.upper() for s in secrets) and v else v)
        for k, v in cfg.items()
    }
