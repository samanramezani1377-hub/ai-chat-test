#!/usr/bin/env python3
"""Regenerate downstream llama.cpp patches from the pinned revision.

Patch files are semantic change specifications. Neither hunk line counts nor
hunk start offsets are trusted: the generator resolves each hunk's preimage
against the actual pinned source, then Git produces the canonical final diff.
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


def check(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def run(*args, cwd=None):
    subprocess.check_call(args, cwd=cwd)


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
                "header": line,
                "old_start": int(match.group(1)),
                "new_start": int(match.group(3)),
                "lines": [],
                "suffix": match.group(5),
            }
            current["hunks"].append(hunk)
        elif current is not None and hunk is not None and line[:1] in " +-":
            hunk["lines"].append(line)
    if current is not None:
        files.append(current)
    return files


def _preimage_and_postimage(hunk):
    old_lines = []
    new_lines = []
    for line in hunk["lines"]:
        prefix = line[:1]
        body = line[1:]
        if prefix in " -":
            old_lines.append(body)
        if prefix in " +":
            new_lines.append(body)
    return old_lines, new_lines


def _find_preimage(lines, preimage, hint):
    if not preimage:
        return max(0, min(hint - 1, len(lines)))
    matches = []
    width = len(preimage)
    for start in range(0, len(lines) - width + 1):
        if lines[start:start + width] == preimage:
            matches.append(start)
    if not matches:
        raise RuntimeError(
            f"cannot resolve patch hunk near old line {hint}: exact preimage not found"
        )
    return min(matches, key=lambda pos: abs((pos + 1) - hint))


def canonicalize_patch(repo, patch_file, work_dir):
    """Resolve stale hunk locations against the real pinned source.

    This is deliberately not a fuzzy application. Every context/deletion line
    must match exactly. Only the temporary hunk locations are rewritten; the
    repository patch is later replaced by a fresh `git diff`, which generates
    both hunk counts and offsets itself.
    """
    text = Path(patch_file).read_text()
    specs = parse_patch(text)
    lines_by_path = {}
    out = []
    cursor = 0
    all_lines = text.splitlines(keepends=True)

    for spec in specs:
        path = spec["path"]
        target = repo / path
        if not target.is_file():
            raise RuntimeError(f"patch target does not exist in pinned source: {path}")
        lines_by_path[path] = target.read_text().splitlines(keepends=True)

    # Rebuild the patch while resolving each hunk against the current
    # preimage. This also accounts for earlier hunks in the same file.
    for line in all_lines:
        if not line.startswith("@@ "):
            out.append(line)
            continue

        # Find the corresponding parsed hunk in source order.
        spec = next(s for s in specs if any(h["header"] == line for h in s["hunks"]))
        hunk = next(h for h in spec["hunks"] if h["header"] == line)
        path = spec["path"]
        current_lines = lines_by_path[path]
        preimage, postimage = _preimage_and_postimage(hunk)
        pos = _find_preimage(current_lines, preimage, hunk["old_start"])

        old_start = pos + 1 if preimage else pos + 1
        new_start = old_start
        suffix = hunk["suffix"]
        out.append(f"@@ -{old_start} +{new_start}{suffix}\n")

        if preimage:
            current_lines[pos:pos + len(preimage)] = postimage
        else:
            current_lines[pos:pos] = postimage

    canonical = work_dir / f"canonical-{Path(patch_file).name}"
    canonical.write_text("".join(out))
    return canonical, specs


def apply_patch(repo, patch_file, work_dir):
    canonical, _ = canonicalize_patch(repo, patch_file, work_dir)
    run("git", "-C", str(repo), "apply", "--recount", "--check", str(canonical))
    run("git", "-C", str(repo), "apply", "--recount", str(canonical))


def git_diff(repo, paths):
    return check("git", "-C", str(repo), "diff", "--no-ext-diff", "--", *paths)


def git_snapshot(repo):
    run("git", "-C", str(repo), "add", "-A")
    run("git", "-C", str(repo), "commit", "--quiet", "--no-verify", "-m", "generator snapshot")


def validate_generated(upstream, generated, work_dir):
    run("git", "-C", str(upstream), "reset", "--hard", "--quiet", TAG)
    run("git", "-C", str(upstream), "clean", "-fd", "-q")
    for name, diff in zip(PATCHES, generated):
        patch_file = work_dir / f"generated-{name}"
        patch_file.write_text(diff)
        run("git", "-C", str(upstream), "apply", "--recount", "--check", str(patch_file))
        run("git", "-C", str(upstream), "apply", "--recount", str(patch_file))
        print(f"VALIDATED: {name}")


with tempfile.TemporaryDirectory(prefix="ai-chat-llama-regen-") as temp_dir:
    temp = Path(temp_dir)
    upstream = temp / "llama.cpp"
    run(
        "git", "clone", "--quiet", "--filter=blob:none",
        "https://github.com/ggml-org/llama.cpp.git", str(upstream)
    )
    run("git", "-C", str(upstream), "checkout", "--quiet", "--detach", TAG)
    run("git", "-C", str(upstream), "config", "user.name", "patch-generator")
    run("git", "-C", str(upstream), "config", "user.email", "patch-generator@localhost")

    specs = [parse_patch((PATCH_DIR / name).read_text()) for name in PATCHES]
    generated = []

    for index, (name, spec) in enumerate(zip(PATCHES, specs)):
        run("git", "-C", str(upstream), "reset", "--hard", "--quiet", TAG)
        run("git", "-C", str(upstream), "clean", "-fd", "-q")

        for previous in PATCHES[:index]:
            apply_patch(upstream, PATCH_DIR / previous, temp)
            git_snapshot(upstream)

        apply_patch(upstream, PATCH_DIR / name, temp)
        diff = git_diff(upstream, [file_spec["path"] for file_spec in spec])
        if not diff.strip():
            raise RuntimeError(f"{name}: generated empty diff")
        generated.append(diff)

    validate_generated(upstream, generated, temp)

    for name, diff in zip(PATCHES, generated):
        (PATCH_DIR / name).write_text(diff)

print(f"Regenerated and validated {len(PATCHES)} llama.cpp patches from {TAG}")
