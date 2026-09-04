#!/usr/bin/env python3
"""Regenerate downstream llama.cpp patches from the pinned revision.

The checked-in patches are semantic change specifications. Hunk counts and
line offsets are never edited or trusted. Each hunk's exact preimage is found
in the pinned source, the semantic change is applied directly to that source,
and Git alone generates the canonical unified diff.
"""

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
            hunk = {
                "old_start": int(match.group(1)),
                "new_start": int(match.group(3)),
                "suffix": match.group(5),
                "lines": [],
            }
            current["hunks"].append(hunk)
        elif current is not None and hunk is not None:
            if line.startswith((" ", "+", "-")):
                hunk["lines"].append(line)
            elif line.startswith("\\ No newline at end of file"):
                hunk["lines"].append(line)
    if current is not None:
        files.append(current)
    if not files:
        raise RuntimeError("Patch contains no file diffs")
    return files


def preimage_postimage(hunk):
    old_lines = []
    new_lines = []
    for line in hunk["lines"]:
        if line.startswith("\\ No newline at end of file"):
            continue
        prefix = line[0]
        body = line[1:]
        if prefix in " -":
            old_lines.append(body)
        if prefix in " +":
            new_lines.append(body)
    return old_lines, new_lines


def find_exact(lines, preimage, hint, path):
    if not preimage:
        return max(0, min(hint - 1, len(lines)))
    width = len(preimage)
    matches = [i for i in range(len(lines) - width + 1) if lines[i:i + width] == preimage]
    if not matches:
        raise RuntimeError(
            f"{path}: cannot resolve hunk near old line {hint}: exact preimage not found"
        )
    return min(matches, key=lambda i: abs((i + 1) - hint))


def apply_semantic_patch(repo, patch_file):
    """Apply a patch specification without constructing/applying a unified diff."""
    specs = parse_patch(Path(patch_file).read_text())
    for spec in specs:
        path = spec["path"]
        target = repo / path
        if not target.is_file():
            raise RuntimeError(f"{path}: target does not exist in pinned revision")

        lines = target.read_text().splitlines(keepends=True)
        # Resolve each hunk against the source state produced by earlier hunks.
        # The old_start is only a search hint; exact preimage equality is required.
        for hunk in spec["hunks"]:
            preimage, postimage = preimage_postimage(hunk)
            pos = find_exact(lines, preimage, hunk["old_start"], path)
            if preimage:
                lines[pos:pos + len(preimage)] = postimage
            else:
                lines[pos:pos] = postimage
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

        # Rebuild the same cumulative state that the downstream patch stack
        # represents. Every transformation is exact; Git generates the output.
        for previous in PATCHES[:index]:
            apply_semantic_patch(upstream, PATCH_DIR / previous)
            snapshot(upstream)

        spec = parse_patch((PATCH_DIR / name).read_text())
        apply_semantic_patch(upstream, PATCH_DIR / name)
        paths = [file_spec["path"] for file_spec in spec]
        diff = diff_for(upstream, paths)
        if not diff.strip():
            raise RuntimeError(f"{name}: generated empty diff")
        generated.append(diff)

    validate(upstream, generated, temp)

    for name, diff in zip(PATCHES, generated):
        (PATCH_DIR / name).write_text(diff)

print(f"Regenerated and validated {len(PATCHES)} llama.cpp patches from {TAG}")
