#!/usr/bin/env python3
"""Regenerate downstream llama.cpp patches from the pinned revision.

The existing patch files are used only as semantic change specifications. Their
unified-diff line counts are deliberately ignored. Each change is located by its
exact preimage in a clean checkout, then Git itself emits the final patch with
`git diff --no-ext-diff`.
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
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)


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


def apply_hunks(repo, spec):
    for file_spec in spec:
        path = repo / file_spec["path"]
        lines = path.read_text().splitlines(keepends=True)
        for hunk in file_spec["hunks"]:
            old = [line for line in hunk["lines"] if not line.startswith("+")]
            new = [line[1:] for line in hunk["lines"] if not line.startswith("-")]

            matches = [
                index
                for index in range(0, len(lines) - len(old) + 1)
                if lines[index : index + len(old)] == old
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"{path}: expected exactly one preimage match, found {len(matches)}"
                )
            index = matches[0]
            lines[index : index + len(old)] = new
        path.write_text("".join(lines))


def git_diff(repo, paths):
    return subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--no-ext-diff", "--", *paths],
        text=True,
    )


with tempfile.TemporaryDirectory(prefix="ai-chat-llama-regen-") as temp_dir:
    upstream = Path(temp_dir) / "llama.cpp"
    subprocess.check_call(
        [
            "git",
            "clone",
            "--quiet",
            "--filter=blob:none",
            "https://github.com/ggml-org/llama.cpp.git",
            str(upstream),
        ]
    )
    subprocess.check_call(["git", "-C", str(upstream), "checkout", "--quiet", "--detach", TAG])

    generated = []
    for index, name in enumerate(PATCHES):
        if index == 0:
            subprocess.check_call(["git", "-C", str(upstream), "reset", "--hard", "--quiet", TAG])

        spec = parse_patch((PATCH_DIR / name).read_text())
        apply_hunks(upstream, spec)
        generated.append(git_diff(upstream, [file_spec["path"] for file_spec in spec]))

    for name, diff in zip(PATCHES, generated):
        if not diff.strip():
            raise RuntimeError(f"{name}: generated empty diff")
        (PATCH_DIR / name).write_text(diff)

print(f"Regenerated {len(PATCHES)} llama.cpp patches from {TAG}")
