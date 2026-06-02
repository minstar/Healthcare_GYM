#!/usr/bin/env python
"""Environment-correctness verification suite for Healthcare AI GYM.

Codifies the bugs found during the paper<->code audit as regression tests so
the harness can auto-verify the environment on every tick. Each check returns
a (name, passed, detail) tuple. Exits 0 iff all checks pass.

Run:
    .venv/bin/python scripts/research_harness/verify_env.py [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Individual checks. Each returns (passed: bool, detail: str).
# --------------------------------------------------------------------------- #
def check_terminated_on_submit():
    """submit_answer / submit_report must set the gym terminated flag.

    Regression test for the bug where state.terminated was only ever set in
    ehr_benchmark_eval.py, so a raw gym.Env consumer never saw an episode end
    on submission (contradicting paper Appendix A.1).
    """
    import bioagents.environment.environment as E

    class _FakeTools:
        def get_db_hash(self):
            return "fake"

        def use_tool(self, name, **kwargs):
            return {"ok": True, "tool": name, "args": kwargs}

        def has_tool(self, name):
            return True

        def get_tool_definitions_dict(self):
            return []

        def get_db_hash(self):  # noqa: F811 (kept explicit for clarity)
            return "fake"

    env = E.Environment(domain_name="test", policy="", tools=_FakeTools(), max_turns=20)
    env.reset()

    # A normal tool call should NOT terminate.
    _, _, term_mid, trunc_mid, _ = env.step(json.dumps({"name": "get_patient_info", "arguments": {}}))
    if term_mid:
        return False, "non-terminal tool wrongly set terminated=True"

    # submit_answer SHOULD terminate.
    _, _, term, trunc, _ = env.step(json.dumps({"name": "submit_answer", "arguments": {"answer": "A"}}))
    if not term:
        return False, "submit_answer did NOT set terminated=True (Bug A regression)"
    return True, "submit_answer terminates; non-terminal tool does not"


def check_reward_default_sums_to_one():
    """The canonical default 6D reward weights must sum to 1.0."""
    import inspect

    from bioagents.evaluation import rewards

    src = inspect.getsource(rewards.compute_composite_reward)
    # The default dict is built when weights is None; reproduce it by calling.
    res = rewards.compute_composite_reward("Answer: A", correct_answer="A")
    w = res["weights"]
    total_w = round(sum(w.values()), 6)
    if abs(total_w - 1.0) > 1e-6:
        return False, f"default weights sum to {total_w}, expected 1.0 (weights={w})"
    return True, f"default 6D weights sum to 1.0 ({sorted(w)})"


def check_reward_partial_dict_uses_only_listed_dims():
    """A partial weights dict means 'use exactly these dims'; omitted dims must
    contribute 0 (symmetric fallback). Otherwise unrequested per-sample signals
    (e.g. coherence) leak into the GRPO advantage. grpo_trainer passes a partial
    {accuracy,format,process} dict, so this is the live-training contract.
    """
    from bioagents.evaluation.rewards import compute_composite_reward

    partial = {"accuracy": 0.25, "format": 0.10, "process": 0.20}
    r = compute_composite_reward("Answer: A", correct_answer="A", weights=partial)
    expected = sum(partial[d] * r[d] for d in partial)  # omitted dims weighted 0
    if abs(r["total"] - expected) > 1e-9:
        return False, (f"partial-dict total {r['total']:.4f} != listed-dims total {expected:.4f} "
                       "— omitted dims leak nonzero weight into the reward")
    return True, f"partial dict uses only listed dims (total={r['total']:.4f})"


def check_tool_count(expected=171):
    """Detect drift in the total decorated-tool count (paper claims 135)."""
    out = subprocess.run(
        ["bash", "-lc", f"grep -rc '@is_tool' {ROOT}/bioagents/domains/*/tools.py | "
                        "awk -F: '{s+=$2} END{print s}'"],
        capture_output=True, text=True,
    )
    try:
        count = int(out.stdout.strip() or "0")
    except ValueError:
        return False, f"could not count tools: {out.stdout!r} {out.stderr!r}"
    if count != expected:
        return False, f"@is_tool count = {count}, expected {expected} (drift — update paper/this check)"
    return True, f"@is_tool count = {count}"


def check_domain_count(expected=10):
    domains_dir = ROOT / "bioagents" / "domains"
    domains = [p.name for p in domains_dir.iterdir()
               if p.is_dir() and not p.name.startswith("__")]
    if len(domains) != expected:
        return False, f"{len(domains)} domains, expected {expected}: {sorted(domains)}"
    return True, f"{len(domains)} domains: {sorted(domains)}"


CHECKS = {
    "terminated_on_submit": check_terminated_on_submit,
    "reward_default_sums_to_one": check_reward_default_sums_to_one,
    "reward_partial_dict_uses_only_listed_dims": check_reward_partial_dict_uses_only_listed_dims,
    "tool_count": check_tool_count,
    "domain_count": check_domain_count,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write JSON report to this path")
    args = ap.parse_args()

    results = []
    for name, fn in CHECKS.items():
        try:
            passed, detail = fn()
        except Exception as e:  # noqa: BLE001
            passed, detail = False, f"EXCEPTION: {type(e).__name__}: {e}"
        results.append({"check": name, "passed": passed, "detail": detail})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: {detail}")

    n_pass = sum(r["passed"] for r in results)
    n = len(results)
    print(f"\n{n_pass}/{n} checks passed")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"passed": n_pass == n, "n_pass": n_pass, "n_total": n, "results": results},
            indent=2))

    sys.exit(0 if n_pass == n else 1)


if __name__ == "__main__":
    main()
