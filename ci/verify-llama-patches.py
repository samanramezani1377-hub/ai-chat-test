#!/usr/bin/env python3
"""Verify the checked-in llama.cpp patch stack against the exact pinned revision.

CI verification only: patch files are immutable source-of-truth. --recount lets
Git infer hunk lengths from their bodies; it never relocates hunks or changes
context, so semantic drift still fails normally.
"""

from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
LLAMA_SHA = "c5fc7e34885ba31217e330809437afa993d27745"
PATCH_DIR = ROOT / "core/runtime-android/src/main/cpp/patches"
PATCHES = ["llama-model-loading-memory.patch", "llama-backend-vulkan-diagnostics.patch", "llama-vulkan-micro-diagnostics.patch", "llama-vulkan-bda-android-safety.patch"]


def run(*args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        run("git", "clone", "--quiet", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(upstream))
        run("git", "-C", str(upstream), "checkout", "--quiet", "--detach", LLAMA_SHA)
    except subprocess.CalledProcessError as exc:
        fail("FAILED TO CHECK OUT PINNED LLAMA REVISION", exc.stdout)

    actual = run("git", "-C", str(upstream), "rev-parse", "HEAD").strip()
    if actual != LLAMA_SHA:
        fail(f"PINNED REVISION MISMATCH: expected {LLAMA_SHA}, got {actual}")

    for index, name in enumerate(PATCHES, 1):
        patch = PATCH_DIR / name
        try:
            run("git", "-C", str(upstream), "apply", "--recount", "--check", str(patch))
            run("git", "-C", str(upstream), "apply", "--recount", str(patch))
        except subprocess.CalledProcessError as exc:
            print(f"PATCH {index}: FAIL")
            fail(f"PATCH {index}: {name} does not apply to {LLAMA_SHA}", exc.stdout)
        print(f"PATCH {index}: PASS")

    diff_check = subprocess.run(["git", "-C", str(upstream), "diff", "--check"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if diff_check.returncode != 0:
        fail("PATCH STACK: whitespace/error check failed", diff_check.stdout)

    print("PATCH STACK: PASS")
