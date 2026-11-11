"""
Atomic Builder, Synchronizer & Zero-Manual QA Validator for the `autonomous-research v2.0` Antigravity Plugin.

Synchronizes:
  Source repo: `<repo_root>/{plugin_bundle, dream_rsi_scripts, prompts}`
  Target plugin: `~/.gemini/config/plugins/autonomous-research` (or `$AUTONOMOUS_RESEARCH_PLUGIN_DIR`)

Usage:
    python sync_plugin.py            # sync, purge .bak clutter, validate, and run _selftest.py
    python sync_plugin.py --verify   # read-only SHA-256 + markdown + frontmatter + budget audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(REPO_ROOT, "plugin_bundle")
LOCAL_SCRIPTS_DIR = os.path.join(REPO_ROOT, "dream_rsi_scripts")
PROMPTS_DIR = os.path.join(REPO_ROOT, "prompts")
TARGET_PLUGIN_DIR = os.environ.get(
    "AUTONOMOUS_RESEARCH_PLUGIN_DIR",
    os.path.expanduser(r"~\.gemini\config\plugins\autonomous-research"),
)

CORE_SCRIPT_FILES = (
    "engine.py",
    "sleep_dag_rsi.py",
    "cartridge.py",
    "dreamrsi.py",
    "loop.py",
    "simulator.py",
    "_selftest.py",
)

MAX_SKILL_BYTES = (
    10_240  # Enforce compact Progressive Disclosure (< 10 KB per SKILL.md)
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sync_manifest() -> List[Tuple[str, str]]:
    """Returns `(src_path, dst_path)` pairs for every managed artifact in the plugin."""
    pairs: List[Tuple[str, str]] = [
        (
            os.path.join(BUNDLE_DIR, "plugin.json"),
            os.path.join(TARGET_PLUGIN_DIR, "plugin.json"),
        ),
        # Skills
        (
            os.path.join(BUNDLE_DIR, "skills", "dream-rsi", "SKILL.md"),
            os.path.join(TARGET_PLUGIN_DIR, "skills", "dream-rsi", "SKILL.md"),
        ),
        (
            os.path.join(BUNDLE_DIR, "skills", "neural-sleep", "SKILL.md"),
            os.path.join(TARGET_PLUGIN_DIR, "skills", "neural-sleep", "SKILL.md"),
        ),
        (
            os.path.join(BUNDLE_DIR, "skills", "epistemic-research", "SKILL.md"),
            os.path.join(TARGET_PLUGIN_DIR, "skills", "epistemic-research", "SKILL.md"),
        ),
        # Disclosed domain references
        (
            os.path.join(
                BUNDLE_DIR, "skills", "dream-rsi", "references", "extremal-math-kkt.md"
            ),
            os.path.join(
                TARGET_PLUGIN_DIR,
                "skills",
                "dream-rsi",
                "references",
                "extremal-math-kkt.md",
            ),
        ),
        (
            os.path.join(
                BUNDLE_DIR,
                "skills",
                "dream-rsi",
                "references",
                "timeseries-trading.md",
            ),
            os.path.join(
                TARGET_PLUGIN_DIR,
                "skills",
                "dream-rsi",
                "references",
                "timeseries-trading.md",
            ),
        ),
        (
            os.path.join(
                BUNDLE_DIR, "skills", "dream-rsi", "references", "engine-v8-api.md"
            ),
            os.path.join(
                TARGET_PLUGIN_DIR,
                "skills",
                "dream-rsi",
                "references",
                "engine-v8-api.md",
            ),
        ),
        # Agents
        (
            os.path.join(BUNDLE_DIR, "agents", "rsi-branch-worker.md"),
            os.path.join(TARGET_PLUGIN_DIR, "agents", "rsi-branch-worker.md"),
        ),
        (
            os.path.join(BUNDLE_DIR, "agents", "rsi-blind-auditor.md"),
            os.path.join(TARGET_PLUGIN_DIR, "agents", "rsi-blind-auditor.md"),
        ),
        # Prompt template
        (
            os.path.join(PROMPTS_DIR, "policy_improvement.md"),
            os.path.join(
                TARGET_PLUGIN_DIR,
                "skills",
                "dream-rsi",
                "prompts",
                "policy_improvement.md",
            ),
        ),
    ]
    for script_name in CORE_SCRIPT_FILES:
        pairs.append(
            (
                os.path.join(LOCAL_SCRIPTS_DIR, script_name),
                os.path.join(
                    TARGET_PLUGIN_DIR, "skills", "dream-rsi", "scripts", script_name
                ),
            )
        )
    return pairs


def validate_markdown_and_frontmatter(
    path: str, require_frontmatter: bool, max_bytes: int | None = None
) -> List[str]:
    errs: List[str] = []
    raw = open(path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        errs.append(f"{path}: UTF-8 BOM detected")
    if max_bytes is not None and len(raw) > max_bytes:
        errs.append(f"{path}: size {len(raw)}B exceeds budget {max_bytes}B")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"{path}: invalid UTF-8 ({exc})"]

    if "\ufffd" in text:
        errs.append(f"{path}: replacement character U+FFFD detected")

    fence_count = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("```"))
    if fence_count % 2 != 0:
        errs.append(f"{path}: unbalanced code fences (count={fence_count})")

    if require_frontmatter:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            errs.append(f"{path}: missing opening YAML frontmatter '---'")
        else:
            end_idx = None
            for idx in range(1, min(len(lines), 30)):
                if lines[idx].strip() == "---":
                    end_idx = idx
                    break
            if end_idx is None:
                errs.append(f"{path}: missing closing YAML frontmatter '---'")
            else:
                fm_block = "\n".join(lines[1:end_idx])
                for req_key in ("name:", "description:"):
                    if req_key not in fm_block:
                        errs.append(f"{path}: frontmatter missing '{req_key}'")
    return errs


def sync_all(verify_only: bool = False) -> Dict[str, object]:
    # Keep local mirror copies consistent first
    if not verify_only:
        shutil.copyfile(
            os.path.join(BUNDLE_DIR, "skills", "dream-rsi", "SKILL.md"),
            os.path.join(LOCAL_SCRIPTS_DIR, "SKILL.md"),
        )
        shutil.copyfile(
            os.path.join(REPO_ROOT, "sleep_dag_rsi.py"),
            os.path.join(LOCAL_SCRIPTS_DIR, "sleep_dag_rsi.py"),
        )

    manifest = build_sync_manifest()
    removed_bak: List[str] = []
    if not verify_only:
        for root, _dirs, files in os.walk(TARGET_PLUGIN_DIR):
            for fn in files:
                if fn.endswith(".bak"):
                    p = os.path.join(root, fn)
                    os.remove(p)
                    removed_bak.append(p)
        for src, dst in manifest:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

    # Audit SHA-256 parity, plugin.json schema, markdown integrity, and size budgets
    errors: List[str] = []
    synced_sizes: Dict[str, int] = {}
    for src, dst in manifest:
        if not os.path.isfile(src):
            errors.append(f"Missing source: {src}")
            continue
        if not os.path.isfile(dst):
            errors.append(f"Missing target: {dst}")
            continue
        if sha256_file(src) != sha256_file(dst):
            errors.append(f"SHA-256 mismatch: {src} != {dst}")
        rel = os.path.relpath(dst, TARGET_PLUGIN_DIR).replace("\\", "/")
        synced_sizes[rel] = os.path.getsize(dst)

        if dst.endswith("SKILL.md"):
            errors.extend(
                validate_markdown_and_frontmatter(
                    dst, require_frontmatter=True, max_bytes=MAX_SKILL_BYTES
                )
            )
        elif "/agents/" in rel and dst.endswith(".md"):
            errors.extend(
                validate_markdown_and_frontmatter(
                    dst, require_frontmatter=True, max_bytes=MAX_SKILL_BYTES
                )
            )
        elif dst.endswith(".md"):
            errors.extend(
                validate_markdown_and_frontmatter(dst, require_frontmatter=False)
            )

    # Check root sleep_dag_rsi.py parity
    if sha256_file(os.path.join(REPO_ROOT, "sleep_dag_rsi.py")) != sha256_file(
        os.path.join(LOCAL_SCRIPTS_DIR, "sleep_dag_rsi.py")
    ):
        errors.append(
            "Root sleep_dag_rsi.py out of sync with dream_rsi_scripts/sleep_dag_rsi.py"
        )

    # Check zero leftover .bak files in target plugin
    for root, _dirs, files in os.walk(TARGET_PLUGIN_DIR):
        for fn in files:
            if fn.endswith(".bak"):
                errors.append(
                    f"Leftover .bak file in target plugin: {os.path.join(root, fn)}"
                )

    # Validate plugin.json
    pjson_path = os.path.join(TARGET_PLUGIN_DIR, "plugin.json")
    with open(pjson_path, "r", encoding="utf-8") as fh:
        pjson = json.load(fh)
    allowed_fields = {
        "name",
        "description",
        "logo",
        "suggestedPrompts",
        "displayName",
        "version",
        "author",
        "homepage",
        "repository",
        "disabled",
    }
    unknown = set(pjson.keys()) - allowed_fields
    if unknown:
        errors.append(f"plugin.json contains unsupported fields: {sorted(unknown)}")
    if len(pjson.get("suggestedPrompts", [])) != 3:
        errors.append("plugin.json should provide 3 suggestedPrompts")

    return {
        "valid": len(errors) == 0,
        "managed_file_count": len(manifest),
        "removed_bak_files": removed_bak,
        "skill_bytes": {
            k: v for k, v in synced_sizes.items() if k.endswith("SKILL.md")
        },
        "agent_bytes": {
            k: v for k, v in synced_sizes.items() if k.startswith("agents/")
        },
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="verify parity and budgets without copying files",
    )
    ap.add_argument(
        "--skip-selftest",
        action="store_true",
        help="skip running installed _selftest.py",
    )
    args = ap.parse_args()

    rep = sync_all(verify_only=args.verify)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not rep["valid"]:
        return 1

    if not args.skip_selftest:
        selftest_path = os.path.join(
            TARGET_PLUGIN_DIR, "skills", "dream-rsi", "scripts", "_selftest.py"
        )
        print(f"\nRunning installed plugin self-test: {selftest_path}")
        proc = subprocess.run(
            [sys.executable, selftest_path],
            capture_output=True,
            text=True,
            check=False,
        )
        tail = "\n".join(proc.stdout.strip().splitlines()[-12:])
        print(tail)
        if proc.returncode != 0 or "RESULT: all checks passed" not in proc.stdout:
            print(proc.stderr, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
