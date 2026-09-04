#!/usr/bin/env python3
"""Developer-only llama.cpp patch migration tool.

IMPORTANT: this tool is intentionally NOT part of CI. Checked-in patch files
are immutable CI inputs. This migration helper may regenerate them only when a
developer explicitly opts in with ALLOW_LLAMA_PATCH_REGEN=1.

Hunk counts and offsets are never manually derived from SHA values. Git alone
creates canonical unified diffs after semantic migration against the pinned
source revision.
"""

import os

if os.environ.get("CI", "").lower() == "true" and os.environ.get("ALLOW_LLAMA_PATCH_REGEN") != "1":
    raise SystemExit(
        "Refusing to regenerate llama.cpp patches in CI. "
        "Patch files are immutable CI inputs; run this migration tool locally "
        "with ALLOW_LLAMA_PATCH_REGEN=1 only when intentionally porting patches."
    )

from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
PATCH_DIR = ROOT / "core/runtime-android/src/main/cpp/patches"
TAG = "c5fc7e34885ba31217e330809437afa993d27745"
PATCHES = [
    "llama-model-loading-memory.patch",
    "llama-backend-vulkan-diagnostics.patch",
    "llama-vulkan-micro-diagnostics.patch",
]


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def parse_patch(text):
    files = []
    current = None
    hunk = None
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current is not None:
                files.append(current)
            match = re.match(r"diff --git a/(.*?) b/(.*)", line)
            if not match:
                raise RuntimeError(f"Malformed diff header: {line.rstrip()}")
            current = {"path": match.group(2), "hunks": []}
            hunk = None
        elif current is not None and line.startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)\n?$", line)
            if not match:
                raise RuntimeError(f"Malformed hunk header: {line.rstrip()}")
            hunk = {"old_start": int(match.group(1)), "new_start": int(match.group(3)), "suffix": match.group(5), "lines": []}
            current["hunks"].append(hunk)
        elif current is not None and hunk is not None:
            if line.startswith((" ", "+", "-")) or line.startswith("\\ No newline"):
                hunk["lines"].append(line)
    if current is not None:
        files.append(current)
    if not files:
        raise RuntimeError("Patch contains no file diffs")
    return files


def preimage_postimage(hunk):
    old_lines, new_lines = [], []
    for line in hunk["lines"]:
        if line.startswith("\\ No newline"):
            continue
        prefix, body = line[0], line[1:]
        if prefix in " -":
            old_lines.append(body)
        if prefix in " +":
            new_lines.append(body)
    return old_lines, new_lines


def find_exact(lines, preimage, hint, path):
    if not preimage:
        return max(0, min(hint - 1, len(lines)))
    matches = [i for i in range(len(lines) - len(preimage) + 1) if lines[i:i + len(preimage)] == preimage]
    if not matches:
        raise RuntimeError(f"{path}: exact preimage not found near old line {hint}")
    if len(matches) > 1:
        return min(matches, key=lambda i: abs((i + 1) - hint))
    return matches[0]


def apply_context_fallback(lines, hunk, path):
    entries = [line for line in hunk["lines"] if not line.startswith("\\ No newline")]
    context = [line[1:] for line in entries if line.startswith(" ")]
    if not context:
        raise RuntimeError(f"{path}: no exact context anchor near old line {hunk['old_start']}")
    matches = [i for i in range(len(lines) - len(context) + 1) if lines[i:i + len(context)] == context]
    if not matches:
        raise RuntimeError(f"{path}: exact context anchor not found near old line {hunk['old_start']}")
    pos = min(matches, key=lambda i: abs((i + 1) - hunk["old_start"]))
    first_context = next(i for i, line in enumerate(entries) if line.startswith(" "))
    last_context = len(entries) - 1 - next(i for i, line in enumerate(reversed(entries)) if line.startswith(" "))
    changed = entries[first_context + 1:last_context] if first_context < last_context else entries[first_context + 1:]
    if any(line.startswith(" ") for line in changed):
        raise RuntimeError(f"{path}: non-contiguous fallback hunk near old line {hunk['old_start']}")
    additions = [line[1:] for line in changed if line.startswith("+")]
    if not additions:
        raise RuntimeError(f"{path}: fallback hunk has no additions near old line {hunk['old_start']}")
    lines[pos + first_context + 1:pos + first_context + 1] = additions
    return lines


def apply_semantic_hunk(lines, hunk, path):
    preimage, postimage = preimage_postimage(hunk)
    try:
        pos = find_exact(lines, preimage, hunk["old_start"], path)
    except RuntimeError:
        return apply_context_fallback(lines, hunk, path)
    lines[pos:pos + len(preimage)] = postimage
    return lines


def apply_semantic_patch(repo, patch_file):
    for spec in parse_patch(Path(patch_file).read_text()):
        target = repo / spec["path"]
        if not target.is_file():
            raise RuntimeError(f"{spec['path']}: target does not exist")
        lines = target.read_text().splitlines(keepends=True)
        for hunk in spec["hunks"]:
            lines = apply_semantic_hunk(lines, hunk, spec["path"])
        target.write_text("".join(lines))


def reset(repo):
    run("git", "-C", str(repo), "reset", "--hard", "--quiet", TAG)
    run("git", "-C", str(repo), "clean", "-fd", "-q")


def snapshot(repo):
    run("git", "-C", str(repo), "add", "-A")
    run("git", "-C", str(repo), "commit", "--quiet", "--no-verify", "-m", "generator snapshot")


def diff_for(repo, paths):
    return run("git", "-C", str(repo), "diff", "--no-ext-diff", "--no-color", "--", *paths)


def validate(upstream, generated, work_dir):
    reset(upstream)
    for index, (name, diff) in enumerate(zip(PATCHES, generated)):
        patch_file = work_dir / f"generated-{index}-{name}"
        patch_file.write_text(diff)
        run("git", "-C", str(upstream), "apply", "--check", str(patch_file))
        run("git", "-C", str(upstream), "apply", str(patch_file))
        print(f"VALIDATED: {name}")


with tempfile.TemporaryDirectory(prefix="ai-chat-llama-regen-") as temp_dir:
    temp = Path(temp_dir)
    upstream = temp / "llama.cpp"
    run("git", "clone", "--quiet", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(upstream))
    run("git", "-C", str(upstream), "checkout", "--quiet", "--detach", TAG)
    run("git", "-C", str(upstream), "config", "user.name", "patch-generator")
    run("git", "-C", str(upstream), "config", "user.email", "patch-generator@localhost")

    generated = []
    for index, name in enumerate(PATCHES):
        reset(upstream)
        for previous in PATCHES[:index]:
            apply_semantic_patch(upstream, PATCH_DIR / previous)
            snapshot(upstream)
        spec = parse_patch((PATCH_DIR / name).read_text())
        apply_semantic_patch(upstream, PATCH_DIR / name)
        diff = diff_for(upstream, [file_spec["path"] for file_spec in spec])
        if not diff.strip():
            raise RuntimeError(f"{name}: generated empty diff")
        generated.append(diff)

    validate(upstream, generated, temp)
    for name, diff in zip(PATCHES, generated):
        (PATCH_DIR / name).write_text(diff)

print(f"Regenerated and validated {len(PATCHES)} llama.cpp patches from {TAG}")
