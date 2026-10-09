#!/usr/bin/env python3
"""Compare device benchmark JSON against a same-device baseline.

This is intentionally opt-in until a reproducible physical-device benchmark
producer and a per-device baseline are available. It never fabricates metrics.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

IDENTITY_FIELDS = ("device", "gpu_renderer", "model_sha256", "quantization", "context")
LOWER_IS_BETTER = ("load_ms", "ttft_ms", "peak_pss_kib")
HIGHER_IS_BETTER = ("prefill_tokens_per_sec", "decode_tokens_per_sec")
METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER


def compare(baseline: dict, current: dict, max_regression_percent: float) -> list[str]:
    failures = []
    for field in IDENTITY_FIELDS:
        if not baseline.get(field) or current.get(field) != baseline.get(field):
            failures.append(f"benchmark identity mismatch: {field}")
    for metric in METRICS:
        try:
            before = float(baseline[metric])
            after = float(current[metric])
        except (KeyError, TypeError, ValueError):
            failures.append(f"missing/non-numeric metric: {metric}")
            continue
        if not math.isfinite(before) or not math.isfinite(after) or before <= 0 or after <= 0:
            failures.append(f"metric must be finite and > 0: {metric}")
            continue
        ratio = after / before
        if metric in LOWER_IS_BETTER and ratio > 1 + max_regression_percent / 100:
            failures.append(f"{metric} regressed {((ratio - 1) * 100):.1f}%")
        if metric in HIGHER_IS_BETTER and ratio < 1 - max_regression_percent / 100:
            failures.append(f"{metric} throughput regressed {((1 - ratio) * 100):.1f}%")
    return failures


def self_test() -> int:
    baseline = {
        "device": "fixture-device", "gpu_renderer": "fixture-gpu",
        "model_sha256": "abc123", "quantization": "Q6_K", "context": 4096,
        "load_ms": 1000, "ttft_ms": 900, "peak_pss_kib": 500000,
        "prefill_tokens_per_sec": 20, "decode_tokens_per_sec": 8,
    }
    good = dict(baseline, load_ms=1050, ttft_ms=950, peak_pss_kib=520000,
                prefill_tokens_per_sec=19, decode_tokens_per_sec=7.5)
    bad = dict(baseline, decode_tokens_per_sec=5)
    mismatch = dict(good, gpu_renderer="different-gpu")
    cases = [
        ("accepts metrics inside tolerance", not compare(baseline, good, 15)),
        ("rejects throughput regression", any("decode_tokens_per_sec" in e for e in compare(baseline, bad, 15))),
        ("rejects mismatched device identity", any("gpu_renderer" in e for e in compare(baseline, mismatch, 15))),
    ]
    failures = []
    for label, ok in cases:
        print(("PASS: " if ok else "FAIL: ") + label)
        if not ok:
            failures.append(label)
    print(f"Performance comparator self-tests: {len(cases) - len(failures)}/{len(cases)} passed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--max-regression-percent", type=float, default=15.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.baseline or not args.current:
        parser.error("provide --baseline and --current, or use --self-test")
    if not math.isfinite(args.max_regression_percent) or args.max_regression_percent < 0:
        parser.error("--max-regression-percent must be finite and >= 0")
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        current = json.loads(args.current.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: unable to read benchmark JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(baseline, dict) or not isinstance(current, dict):
        print("FAIL: benchmark JSON root must be an object", file=sys.stderr)
        return 2
    failures = compare(baseline, current, args.max_regression_percent)
    if failures:
        print("FAIL: benchmark regression or identity mismatch")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print(f"PASS: benchmark metrics within {args.max_regression_percent:g}% tolerance for matching device/model/context")
    for metric in METRICS:
        print(f"  {metric}: baseline={baseline[metric]} current={current[metric]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
