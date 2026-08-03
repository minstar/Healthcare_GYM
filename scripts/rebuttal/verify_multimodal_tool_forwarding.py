#!/usr/bin/env python
"""Verify the HCGYM_MULTIMODAL_TOOLS switch changes exactly one thing.

Rebuttal, 2026-07-29. ``_generate_sglang_multimodal`` used to accept a ``tools``
argument and never forward it, so on the IMAGE benchmarks the Base+AR condition
(prompt_mode default / strong_tool) measured a TOOLLESS agent under a heading
that says tool-using. That call is the only path by which the catalog reaches
those modes on an image input; prompt_mode="react" renders its own catalog into
the system prompt and never depended on it.

The fix is in, and ``HCGYM_MULTIMODAL_TOOLS=0`` reproduces the pre-fix
behaviour from the SAME binary. Both arms of the comparison therefore differ in
one runtime flag rather than in two code states — which is the only way the
difference can be attributed to tool forwarding at all.

The interesting checks here are DIFFERENTIAL, not unit: every one of them drives
the real eval loop twice, once per arm, against a stubbed sglang client that
records the exact request payload, and diffs what the server would have seen.

Checks
  1. payload_diff_vqa
        Real vqa_rad tasks, real images, prompt_mode=default, through
        eval_benchmark_multiturn.run_benchmark_multiturn. Every request payload
        is captured under both arms and diffed. The ONLY permitted difference
        is presence/absence of the "tools" key.
  2. system_prompt_identical
        The rendered system prompt the model receives is byte-identical between
        the two arms (sha256 reported), for default and strong_tool.
  3. text_only_unaffected
        Real medqa tasks under both arms produce byte-identical requests — same
        endpoint (completions, not chat.completions), same rendered prompt
        string. The results artifact records multimodal_requests=0, so "this
        switch cannot touch a text benchmark" is a recorded number.
  4. react_unaffected
        Same VQA tasks under prompt_mode=react: payloads byte-identical across
        arms, with no "tools" key in either, because react arrives at the
        multimodal call with tools=None regardless of the flag.
  5. artifact_records_arm
        Every results JSON carries "multimodal_tool_forwarding" at the top
        level, with the arm, the env value, and how many image requests
        actually carried the catalog. A number cannot be read without it.
  6. default_is_correct_behaviour
        With the variable unset the catalog IS forwarded, an unparseable value
        raises rather than guessing, and only the toolless arm needs an
        explicit opt-in.

Run:
    PYTHONPATH=<repo> .venv/bin/python \
        scripts/rebuttal/verify_multimodal_tool_forwarding.py -v
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from bioagents.evaluation.agent_runner import (  # noqa: E402
    MULTIMODAL_TOOLS_ENV,
    AgentRunner,
    RunConfig,
    multimodal_tool_forwarding,
    multimodal_tools_arm,
    multimodal_tools_enabled,
)

TOKENIZER_PATH = "/data/project/private/minstar/workspace/hcgym_rebuttal/models/Qwen3.5-9B"

# Two turns, so the diff covers a follow-up request carrying a tool observation
# and not just the opening one. Identical in both arms by construction: the stub
# replays this script, so the model's behaviour cannot itself differ and confound
# the payload comparison.
SCRIPT = [
    '{"name": "think", "arguments": {"thought": "Inspect the study before answering."}}',
    '{"name": "submit_answer", "arguments": {"answer": "yes", "reasoning": "grounded"}}',
]


# --------------------------------------------------------------------------- #
#  Stub sglang client — records the exact payload, returns a scripted output
# --------------------------------------------------------------------------- #
class _Choice:
    def __init__(self, text):
        self.text = text
        self.message = type("M", (), {"content": text})()


class _Response:
    def __init__(self, text):
        self.choices = [_Choice(text)]


class _Endpoint:
    def __init__(self, runner, name):
        self._runner = runner
        self._name = name

    def create(self, **kwargs):
        self._runner.captured.append(
            {"endpoint": self._name, "payload": copy.deepcopy(kwargs)}
        )
        out = self._runner._script[min(self._runner._i, len(self._runner._script) - 1)]
        self._runner._i += 1
        return _Response(out)


class _StubClient:
    def __init__(self, runner):
        self.chat = type("C", (), {"completions": _Endpoint(runner, "chat.completions")})()
        self.completions = _Endpoint(runner, "completions")


class StubRunner(AgentRunner):
    """AgentRunner on the real sglang code path, with a recording stub client.

    Nothing about request construction is reimplemented: `_generate_sglang` and
    `_generate_sglang_multimodal` run unmodified, and `captured` is exactly what
    they handed to the OpenAI client.
    """

    def __init__(self, config: RunConfig, script=SCRIPT):
        super().__init__(config)
        self._script = list(script)
        self._i = 0
        self.captured: list[dict] = []
        self._sglang_client = _StubClient(self)
        self._sglang_model_name = "hcgym"
        self._is_vl_model = False
        self.processor = None
        self.model = None
        self.tokenizer = _TOKENIZER

    def load_model(self):  # no GPU, no weights, no server
        pass


_TOKENIZER = None


def _load_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return _TOKENIZER


# --------------------------------------------------------------------------- #
#  Arm harness
# --------------------------------------------------------------------------- #
class arm:
    """Context manager that pins HCGYM_MULTIMODAL_TOOLS for one run."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self._prev = os.environ.get(MULTIMODAL_TOOLS_ENV)
        if self.value is None:
            os.environ.pop(MULTIMODAL_TOOLS_ENV, None)
        else:
            os.environ[MULTIMODAL_TOOLS_ENV] = self.value
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop(MULTIMODAL_TOOLS_ENV, None)
        else:
            os.environ[MULTIMODAL_TOOLS_ENV] = self._prev
        return False


def _tasks(benchmark: str, n: int) -> list[dict]:
    import eval_benchmark_multiturn as EB

    if benchmark in {"vqa_rad", "slake"}:
        items = EB.load_vqa_benchmark(benchmark)
    else:
        items = EB.load_textqa_benchmark(benchmark)
    return [copy.deepcopy(t) for t in items[:n]]


def run_arm(benchmark: str, prompt_mode: str, env_value, n_tasks: int = 2):
    """Drive the REAL benchmark loop under one arm. Returns (captured, artifact)."""
    import eval_benchmark_multiturn as EB

    _load_tokenizer()
    tasks = _tasks(benchmark, n_tasks)
    domain = EB.BENCHMARK_DOMAIN[benchmark]

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        cfg = RunConfig(
            model_name_or_path="stub-model",
            backend="sglang",
            server_url="http://127.0.0.1:0",
            domain=domain,
            max_turns=3,
            log_dir=str(Path(td) / "logs"),
            no_think=True,
            prompt_mode=prompt_mode,
        )
        runner = StubRunner(cfg)
        with arm(env_value):
            EB.run_benchmark_multiturn(
                benchmark_name=benchmark,
                tasks=tasks,
                runner=runner,
                domain=domain,
                max_turns=3,
                output_dir=out_dir,
            )
        written = sorted(out_dir.glob(f"{benchmark}_multiturn_*.json"))
        artifact = json.loads(written[-1].read_text()) if written else {}
    return runner.captured, artifact


# --------------------------------------------------------------------------- #
#  Payload rendering / diffing
# --------------------------------------------------------------------------- #
def _elide_images(obj):
    """Replace base64 image payloads with a stable digest so a diff is readable.

    Applied ONLY for display. Equality is always asserted on the full payload.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image/"):
                head, _, b64 = v.partition(";base64,")
                out[k] = (f"{head};base64,<{len(b64)} b64 chars, "
                          f"sha256:{hashlib.sha256(b64.encode()).hexdigest()[:16]}>")
            else:
                out[k] = _elide_images(v)
        return out
    if isinstance(obj, list):
        return [_elide_images(x) for x in obj]
    return obj


def _render(entry: dict, tools_detail: bool = False) -> str:
    payload = _elide_images(entry["payload"])
    if not tools_detail and "tools" in payload:
        names = [t.get("function", t).get("name") for t in payload["tools"]]
        payload["tools"] = (f"<{len(payload['tools'])} tool specs: "
                            f"{', '.join(names[:4])}, ...>")
    return f"POST {entry['endpoint']}\n" + json.dumps(
        payload, indent=2, ensure_ascii=False, sort_keys=True, default=str
    )


def _key(entry: dict) -> str:
    return json.dumps(
        {"endpoint": entry["endpoint"], "payload": entry["payload"]},
        sort_keys=True, ensure_ascii=False, default=str,
    )


def payload_diff(a: list[dict], b: list[dict], label_a: str, label_b: str) -> str:
    lines_a, lines_b = [], []
    for entry in a:
        lines_a.extend(_render(entry).splitlines())
    for entry in b:
        lines_b.extend(_render(entry).splitlines())
    return "\n".join(difflib.unified_diff(
        lines_a, lines_b, fromfile=label_a, tofile=label_b, lineterm="", n=2,
    ))


def _differing_top_level_keys(a: list[dict], b: list[dict]) -> set:
    """Top-level payload keys whose value differs between the two runs."""
    keys = set()
    for ea, eb in zip(a, b):
        if ea["endpoint"] != eb["endpoint"]:
            keys.add("__endpoint__")
        for k in set(ea["payload"]) | set(eb["payload"]):
            va = json.dumps(ea["payload"].get(k, "__ABSENT__"), sort_keys=True, default=str)
            vb = json.dumps(eb["payload"].get(k, "__ABSENT__"), sort_keys=True, default=str)
            if va != vb:
                keys.add(k)
    return keys


def _system_prompt(entry: dict) -> str:
    for m in entry["payload"].get("messages", []):
        if m.get("role") == "system":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c, sort_keys=True)
    # text-only path renders the whole conversation into a single prompt string
    return entry["payload"].get("prompt", "")


# --------------------------------------------------------------------------- #
#  Checks
# --------------------------------------------------------------------------- #
def check_payload_diff_vqa(verbose: bool = False):
    """The two arms must differ in the "tools" key and in nothing else."""
    fwd, art_f = run_arm("vqa_rad", "default", None)
    wit, art_w = run_arm("vqa_rad", "default", "0")

    if not fwd or not wit:
        return False, "no requests captured — the VQA path did not run"
    if len(fwd) != len(wit):
        return False, (f"different number of requests: forward={len(fwd)} "
                       f"withheld={len(wit)} — the loop itself diverged")
    if any(e["endpoint"] != "chat.completions" for e in fwd):
        return False, f"VQA did not take the multimodal path: {[e['endpoint'] for e in fwd]}"

    differing = _differing_top_level_keys(fwd, wit)
    if differing != {"tools"}:
        return False, (f"arms differ in {sorted(differing)}, expected exactly "
                       f"{{'tools'}}")
    if any("tools" not in e["payload"] for e in fwd):
        return False, "forward arm sent a request WITHOUT the tool catalog"
    if any("tools" in e["payload"] for e in wit):
        return False, "withheld arm sent a request WITH the tool catalog"

    n_tools = len(fwd[0]["payload"]["tools"])
    if verbose:
        print("\n--- request payload diff: forward vs withheld "
              f"(vqa_rad, prompt_mode=default, {len(fwd)} requests) ---")
        print(payload_diff(fwd, wit, "HCGYM_MULTIMODAL_TOOLS unset (forward)",
                           "HCGYM_MULTIMODAL_TOOLS=0 (withheld)"))
        print(f"\n(tool spec list elided above; it carries {n_tools} tools, "
              f"first: "
              f"{[t.get('function', t).get('name') for t in fwd[0]['payload']['tools']][:5]})")

    return True, (f"{len(fwd)} multimodal requests over {art_f['total']} vqa_rad "
                  f"tasks; sole payload difference is the 'tools' key "
                  f"({n_tools} specs present vs absent); messages, model, "
                  f"max_tokens, temperature and top_p byte-identical")


def check_system_prompt_identical(verbose: bool = False):
    """The rendered system prompt must not move between arms."""
    rows = []
    for mode in ("default", "strong_tool"):
        fwd, _ = run_arm("vqa_rad", mode, None)
        wit, _ = run_arm("vqa_rad", mode, "0")
        for i, (a, b) in enumerate(zip(fwd, wit)):
            sa, sb = _system_prompt(a), _system_prompt(b)
            if sa != sb:
                return False, (f"prompt_mode={mode} request {i}: system prompt "
                               f"CHANGED between arms ({len(sa)} vs {len(sb)} chars)")
            if not sa:
                return False, f"prompt_mode={mode} request {i}: no system prompt found"
        h = hashlib.sha256(_system_prompt(fwd[0]).encode()).hexdigest()
        rows.append((mode, len(_system_prompt(fwd[0])), h, len(fwd)))

    # Whole message list, not just the system turn: images, task ticket, tool
    # observations and the last-turn nudge all have to be identical too.
    fwd, _ = run_arm("vqa_rad", "default", None)
    wit, _ = run_arm("vqa_rad", "default", "0")
    for i, (a, b) in enumerate(zip(fwd, wit)):
        ma = json.dumps(a["payload"]["messages"], sort_keys=True, default=str)
        mb = json.dumps(b["payload"]["messages"], sort_keys=True, default=str)
        if ma != mb:
            return False, f"request {i}: the message list differs between arms"

    if verbose:
        print(f"\n{'prompt_mode':14} {'sys-prompt chars':>18} {'requests':>10}  sha256 "
              f"(identical in both arms)")
        for mode, n, h, k in rows:
            print(f"{mode:14} {n:18d} {k:10d}  {h}")

    return True, "; ".join(
        f"{mode}: {n} chars, sha256 {h[:16]}… identical across arms"
        for mode, n, h, _ in rows
    )


def check_text_only_unaffected(verbose: bool = False):
    """A text benchmark must produce byte-identical requests in both settings."""
    fwd, art_f = run_arm("medqa", "default", None)
    wit, art_w = run_arm("medqa", "default", "0")

    if not fwd:
        return False, "no requests captured for medqa"
    if any(e["endpoint"] != "completions" for e in fwd):
        return False, (f"medqa took the multimodal path: "
                       f"{sorted({e['endpoint'] for e in fwd})}")
    if len(fwd) != len(wit):
        return False, f"request count differs: {len(fwd)} vs {len(wit)}"
    for i, (a, b) in enumerate(zip(fwd, wit)):
        if _key(a) != _key(b):
            return False, f"medqa request {i} DIFFERS between arms"

    for name, art in (("forward", art_f), ("withheld", art_w)):
        mm = art.get("multimodal_tool_forwarding", {})
        if mm.get("multimodal_requests") != 0:
            return False, (f"{name}: medqa artifact reports "
                           f"multimodal_requests={mm.get('multimodal_requests')}, "
                           f"expected 0")

    h = hashlib.sha256(
        "".join(_key(e) for e in fwd).encode()
    ).hexdigest()
    if verbose:
        print(f"\n--- medqa (text-only), {len(fwd)} requests, both arms ---")
        print(f"endpoints           : {sorted({e['endpoint'] for e in fwd})}")
        print(f"'tools' key present : "
              f"{any('tools' in e['payload'] for e in fwd + wit)}")
        print(f"sha256 of all payloads, forward arm : {h}")
        print(f"sha256 of all payloads, withheld arm: "
              f"{hashlib.sha256(''.join(_key(e) for e in wit).encode()).hexdigest()}")
        print(f"artifact multimodal_requests        : "
              f"{art_f['multimodal_tool_forwarding']['multimodal_requests']} "
              f"(forward) / "
              f"{art_w['multimodal_tool_forwarding']['multimodal_requests']} (withheld)")
        print(f"\nfirst 400 chars of the rendered prompt (identical in both arms):\n"
              f"{fwd[0]['payload']['prompt'][:400]}")

    return True, (f"{len(fwd)} medqa requests byte-identical across arms "
                  f"(sha256 {h[:16]}…), all on the text 'completions' endpoint; "
                  f"artifact records multimodal_requests=0 in both arms")


def check_react_unaffected(verbose: bool = False):
    """prompt_mode=react carries its own catalog and must ignore the flag."""
    fwd, _ = run_arm("vqa_rad", "react", None)
    wit, _ = run_arm("vqa_rad", "react", "0")

    if not fwd:
        return False, "no requests captured for react"
    if len(fwd) != len(wit):
        return False, f"request count differs: {len(fwd)} vs {len(wit)}"
    for i, (a, b) in enumerate(zip(fwd, wit)):
        if _key(a) != _key(b):
            return False, f"react request {i} DIFFERS between arms"
    if any("tools" in e["payload"] for e in fwd + wit):
        return False, "react sent a native 'tools' key — its catalog is in-prompt"

    # ...and the catalog really is in the prompt, in both arms, so "unaffected"
    # does not quietly mean "toolless in both".
    from bioagents.evaluation.agent_runner import render_tool_catalog

    sysp = _system_prompt(fwd[0])
    if "<tools>" not in sysp:
        return False, "react system prompt carries no <tools> catalog"

    h = hashlib.sha256("".join(_key(e) for e in fwd).encode()).hexdigest()
    if verbose:
        print(f"\n--- vqa_rad, prompt_mode=react, {len(fwd)} requests ---")
        print(f"native 'tools' key sent : "
              f"{any('tools' in e['payload'] for e in fwd + wit)}")
        print(f"'<tools>' in system prompt: {'<tools>' in sysp}")
        print(f"sha256 forward  : {h}")
        print(f"sha256 withheld : "
              f"{hashlib.sha256(''.join(_key(e) for e in wit).encode()).hexdigest()}")

    return True, (f"{len(fwd)} react requests byte-identical across arms "
                  f"(sha256 {h[:16]}…); no native 'tools' key in either, and the "
                  f"<tools> catalog is present in the system prompt regardless "
                  f"of the flag")


def check_artifact_records_arm(verbose: bool = False):
    """The arm must be in the results file, not only in a log line."""
    seen = {}
    for env_value, expect in ((None, "forward"), ("1", "forward"), ("0", "withheld")):
        _cap, art = run_arm("vqa_rad", "default", env_value)
        if "multimodal_tool_forwarding" not in art:
            return False, (f"{MULTIMODAL_TOOLS_ENV}={env_value!r}: "
                           f"multimodal_tool_forwarding missing from the results JSON")
        mm = art["multimodal_tool_forwarding"]
        if mm.get("arm") != expect:
            return False, (f"{MULTIMODAL_TOOLS_ENV}={env_value!r}: artifact says "
                           f"arm={mm.get('arm')!r}, expected {expect!r}")
        if mm.get("env_value") != env_value:
            return False, (f"artifact env_value={mm.get('env_value')!r}, "
                           f"expected {env_value!r}")
        want_with_tools = mm["multimodal_requests"] if expect == "forward" else 0
        if mm.get("multimodal_requests_with_tools") != want_with_tools:
            return False, (f"{expect}: multimodal_requests_with_tools="
                           f"{mm.get('multimodal_requests_with_tools')}, "
                           f"expected {want_with_tools}")
        if mm["multimodal_requests"] == 0:
            return False, f"{expect}: artifact recorded 0 image requests on vqa_rad"
        seen[str(env_value)] = mm
        if verbose:
            print(f"\nHCGYM_MULTIMODAL_TOOLS={env_value!r} -> results JSON "
                  f'["multimodal_tool_forwarding"]:')
            print(json.dumps(mm, indent=2, ensure_ascii=False))

    return True, ("results JSON carries multimodal_tool_forwarding at the top "
                  "level: " + ", ".join(
                      f"{k}->{v['arm']} "
                      f"({v['multimodal_requests_with_tools']}/{v['multimodal_requests']} "
                      f"image requests with catalog)"
                      for k, v in seen.items()))


def check_default_is_correct_behaviour(verbose: bool = False):
    """Unset must mean forward, and a typo must raise rather than pick an arm."""
    with arm(None):
        if not multimodal_tools_enabled():
            return False, "unset does not default to forwarding the catalog"
        if multimodal_tools_arm() != "forward":
            return False, f"unset arm is {multimodal_tools_arm()!r}"
        if multimodal_tool_forwarding()["default_when_unset"] is not True:
            return False, "artifact block does not declare the default"
    for good in ("1", "true", "TRUE", "yes", "on", "forward", " 1 ", ""):
        with arm(good):
            if not multimodal_tools_enabled():
                return False, f"{good!r} should forward"
    for bad_arm in ("0", "false", "no", "off", "withhold", " 0 "):
        with arm(bad_arm):
            if multimodal_tools_enabled():
                return False, f"{bad_arm!r} should withhold"
    raised = []
    for typo in ("2", "ture", "none", "tools", "-1", "forwrd"):
        with arm(typo):
            try:
                multimodal_tools_enabled()
            except ValueError:
                raised.append(typo)
            else:
                return False, (f"{typo!r} was silently accepted — a typo must not "
                               f"select an arm")
    if verbose:
        print(f"\nunset -> {multimodal_tools_arm()!r} (correct behaviour is the default)")
        print(f"values that raise instead of guessing: {raised}")
    return True, ("unset/1/true/yes/on/forward -> forward (the fixed behaviour); "
                  "0/false/no/off/withhold -> withheld (explicit opt-in only); "
                  f"{len(raised)} malformed values raise ValueError")


CHECKS = {
    "payload_diff_vqa": check_payload_diff_vqa,
    "system_prompt_identical": check_system_prompt_identical,
    "text_only_unaffected": check_text_only_unaffected,
    "react_unaffected": check_react_unaffected,
    "artifact_records_arm": check_artifact_records_arm,
    "default_is_correct_behaviour": check_default_is_correct_behaviour,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write JSON report to this path")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the captured payloads and the diff")
    args = ap.parse_args()

    results = []
    for name, fn in CHECKS.items():
        try:
            passed, detail = fn(verbose=args.verbose)
        except Exception as e:  # noqa: BLE001
            import traceback
            passed, detail = False, (f"EXCEPTION: {type(e).__name__}: {e}\n"
                                     + traceback.format_exc())
        results.append({"check": name, "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    n_pass = sum(r["passed"] for r in results)
    print(f"\n{n_pass}/{len(results)} checks passed")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"passed": n_pass == len(results), "n_pass": n_pass,
             "n_total": len(results), "results": results}, indent=2))

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
