#!/usr/bin/env python3
"""Regenerate downstream llama.cpp patches from the pinned revision.

Patch files are treated as semantic change specifications. Unified-diff hunk
line counts are never trusted or edited manually: Git generates the final
headers from the resulting file state.

Application is delegated to Git itself with --recount. This is important:
--recount makes Git infer hunk sizes from the actual patch body instead of
trusting stale @@ counts, while normal Git patch matching still provides the
safety boundary. Ambiguous or genuinely stale preimages fail instead of being
guessed.
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
            hunk = {"lines": []}
            current["hunks"].append(hunk)
        elif current is not None and hunk is not None and line[:1] in " +-":
            hunk["lines"].append(line)
    if current is not None:
        files.append(current)
    return files


def apply_patch(repo, patch_file):
    """Apply a patch without trusting manually edited hunk line counts."""
    patch_file = Path(patch_file).resolve()
    run("git", "-C", str(repo), "apply", "--recount", "--check", str(patch_file))
    run("git", "-C", str(repo), "apply", "--recount", str(patch_file))


def git_diff(repo, paths):
    return check("git", "-C", str(repo), "diff", "--no-ext-diff", "--", *paths)


def git_snapshot(repo):
    run("git", "-C", str(repo), "add", "-A")
    run("git", "-C", str(repo), "commit", "--quiet", "--no-verify", "-m", "generator snapshot")


def validate_generated(upstream, generated):
    run("git", "-C", str(upstream), "reset", "--hard", "--quiet", TAG)
    run("git", "-C", str(upstream), "clean", "-fd", "-q")
    for name, diff in zip(PATCHES, generated):
        patch_file = upstream / f".generated-{name}"
        patch_file.write_text(diff)
        try:
            run("git", "-C", str(upstream), "apply", "--recount", "--check", str(patch_file))
            run("git", "-C", str(upstream), "apply", "--recount", str(patch_file))
        finally:
            patch_file.unlink(missing_ok=True)
        print(f"VALIDATED: {name}")


with tempfile.TemporaryDirectory(prefix="ai-chat-llama-regen-") as temp_dir:
    upstream = Path(temp_dir) / "llama.cpp"
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
            apply_patch(upstream, PATCH_DIR / previous)
            git_snapshot(upstream)

        apply_patch(upstream, PATCH_DIR / name)
        diff = git_diff(upstream, [file_spec["path"] for file_spec in spec])
        if not diff.strip():
            raise RuntimeError(f"{name}: generated empty diff")
        generated.append(diff)

    validate_generated(upstream, generated)

    for name, diff in zip(PATCHES, generated):
        (PATCH_DIR / name).write_text(diff)

print(f"Regenerated and validated {len(PATCHES)} llama.cpp patches from {TAG}")
