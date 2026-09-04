#!/usr/bin/env python3
"""Verify the checked-in llama.cpp patch stack against the exact pinned revision.

This is CI verification only. It never rewrites patch files, never changes hunk
headers, and never commits or pushes anything. The patch files are the single
source of truth; Git's native patch engine decides whether they apply.
"""

from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
LLAMA_SHA = "c5fc7e34885ba31217e330809437afa993d27745"
PATCH_DIR = ROOT / "core/runtime-android/src/main/cpp/patches"
PATCHES = [
    "llama-model-loading-memory.patch",
    "llama-backend-vulkan-diagnostics.patch",
    "llama-vulkan-micro-diagnostics.patch",
]


def run(*args, cwd=None):
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message, output=""):
    if output:
        print(output.rstrip(), file=sys.stderr)
    raise RuntimeError(message)


print(f"LLAMA REVISION: {LLAMA_SHA}")
print("PATCH STACK:")
for index, name in enumerate(PATCHES, 1):
    path = PATCH_DIR / name
    if not path.is_file():
        fail(f"PATCH {index}: MISSING {name}")
    print(f"  {index}. {name} sha256={sha256(path)}")

with tempfile.TemporaryDirectory(prefix="ai-chat-llama-verify-") as temp_dir:
    upstream = Path(temp_dir) / "llama.cpp"
    try:
        run(
            "git", "clone", "--quiet", "--filter=blob:none",
            "https://github.com/ggml-org/llama.cpp.git", str(upstream),
        )
        run("git", "-C", str(upstream), "checkout", "--quiet", "--detach", LLAMA_SHA)
    except subprocess.CalledProcessError as exc:
        fail("FAILED TO CHECK OUT PINNED LLAMA REVISION", exc.stdout)

    actual = run("git", "-C", str(upstream), "rev-parse", "HEAD").strip()
    if actual != LLAMA_SHA:
        fail(f"PINNED REVISION MISMATCH: expected {LLAMA_SHA}, got {actual}")

    for index, name in enumerate(PATCHES, 1):
        patch = PATCH_DIR / name
        try:
            check = run("git", "-C", str(upstream), "apply", "--check", str(patch))
        except subprocess.CalledProcessError as exc:
            print(f"PATCH {index}: FAIL")
            fail(f"PATCH {index}: {name} does not apply to {LLAMA_SHA}", exc.stdout)
        if check.strip():
            print(check.rstrip())
        run("git", "-C", str(upstream), "apply", str(patch))
        print(f"PATCH {index}: PASS")

    diff_check = subprocess.run(
        ["git", "-C", str(upstream), "diff", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if diff_check.returncode != 0:
        fail("PATCH STACK: whitespace/error check failed", diff_check.stdout)

    # Prove the resulting tree is exactly the cumulative checked-in patch stack.
    # No generated patch is written back to the repository.
    print("PATCH STACK: PASS")
    print(f"PATCHED TREE: {run('git', '-C', str(upstream), 'rev-parse', 'HEAD').strip()} + {len(PATCHES)} checked-in patches")
