#!/usr/bin/env python3
"""Check whether TT-OPD's outcome-conditioned privileged hints actually execute.

The paper names outcome-conditioned privileged hints as one of TT-OPD's three
mechanisms. This script traces, reproducibly, whether the code that builds and
injects them is reachable under the configurations the runs actually used.

Four gates. Hints fire only if a run clears ALL of them.

  GATE 1  streaming teacher path forwards hints
          agent_loop.py sets stream_teacher_with_rollout from
          distillation.teacher_model.enable_resource_pool. When that is True, teacher
          logprobs come from _compute_teacher_logprobs; if that call site does not
          forward hint_token_ids, hints cannot fire, because the batch path that
          builds them is gated on `not stream_teacher_with_rollout`.

  GATE 2  a tokenizer reaches the hint builder
          The injected turn is rendered with the model's chat template, so the
          builder needs a tokenizer. AsyncTeacherLLMServerManager used to take
          tokenizer=None by default while neither construction site passed one, and
          the documented self.config.tokenizer fallback does not resolve on a
          DictConfig. Passes when the parameter has no silent None default AND every
          construction site passes a tokenizer. Parsed with ast, not regex: a regex
          over the raw call text stops at the first ')' inside a comment or a nested
          call and reports false negatives.

  GATE 3  the outcome signal is available where the hint is chosen
          An outcome-conditioned hint needs THIS trajectory's correctness. On the
          streaming path that means the reward must already be on the output object
          when the teacher call is made, and the hint branch must actually read it.
          Passes when _compute_score is awaited before _compute_teacher_logprobs in
          the same function, the streaming hint branch consumes reward_score, and a
          missing score maps to an explicit suppress-injection outcome.

  GATE 4  the resolver really produces distinct hints (live execution)
          Not a grep: imports the working tree, enables HINT_OPD_ENABLED, and runs
          the injection resolver over synthetic correct / incorrect / missing-score /
          multimodal samples, asserting that correct and incorrect get DIFFERENT
          token spans and that a missing score suppresses injection. Against --rev
          the module cannot be imported, so the gate falls back to asserting the
          resolver exists in that revision at all.

  SCOPE   multimodal samples are skipped by design in both paths: inserting tokens
          shifts image placeholder positions and breaks VLM processing. That is a
          documented restriction, not a defect, so it is reported as scope together
          with the MEASURED multimodal fraction of the configured training data. It
          is promoted to a blocking gate only if it would suppress every sample,
          which is checked rather than assumed.

Usage:
    python scripts/rebuttal/verify_hint_injection.py --verl /path/to/verl_ttopd
    python scripts/rebuttal/verify_hint_injection.py --verl <path> --rev e14d6a58

--rev inspects a git revision instead of the working tree, so the shipped code can be
checked rather than whatever is currently being edited.
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

AGENT_LOOP = "verl/experimental/agent_loop/agent_loop.py"
TEACHER_MANAGER = "verl/experimental/teacher_loop/teacher_manager.py"
TEACHER_MODEL = "verl/experimental/teacher_loop/teacher_model.py"

# Training data the arms are configured against; used to measure how many samples
# the multimodal restriction actually suppresses.
DEFAULT_DATA = (
    "/data/project/private/minstar/workspace/hcgym_rebuttal/data/verl_parquet/full_4modality_clean/train.parquet"
)


def read(verl: Path, rel: str, rev: str | None) -> str:
    if rev:
        out = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=verl, capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"[fatal] cannot read {rel} at {rev}: {out.stderr.strip()}")
        return out.stdout
    return (verl / rel).read_text()


def _parse(verl: Path, rel: str, rev: str | None) -> ast.Module:
    return ast.parse(read(verl, rel, rev))


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    """Every call to ``name``, whether bare (f(...)) or attribute (self.f(...))."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if called == name:
                out.append(node)
    return out


def _kwarg_names(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


# ── GATE 1 ────────────────────────────────────────────────────────────────────


def gate1_teacher_path(verl: Path, rev: str | None) -> tuple[bool, list[str]]:
    notes = []
    src = read(verl, AGENT_LOOP, rev)
    tree = ast.parse(src)

    m = re.search(r"self\.stream_teacher_with_rollout\s*=\s*(.+)", src)
    notes.append(f"stream_teacher_with_rollout <- {m.group(1).strip() if m else '???'}")

    calls = _calls_named(tree, "compute_teacher_logprobs_single")
    forwards = [("hint_token_ids" in _kwarg_names(c)) for c in calls]
    notes.append(f"streaming call sites: {len(calls)}; forwarding hint_token_ids: {sum(forwards)}")
    for call, fwd in zip(calls, forwards, strict=True):
        notes.append(f"  {'FORWARDS' if fwd else '    none'}  line {call.lineno}: kwargs={sorted(_kwarg_names(call))}")

    gated = bool(re.search(r"if\s+self\.distillation_enabled\s+and\s+not\s+self\.stream_teacher_with_rollout", src))
    notes.append("batch (hint-building) path gated on `not stream_teacher_with_rollout`: " + ("yes" if gated else "no"))
    return bool(calls) and all(forwards), notes


# ── GATE 2 ────────────────────────────────────────────────────────────────────


def _tokenizer_default(tree: ast.Module) -> object:
    """None-default status of AsyncTeacherLLMServerManager.__init__'s tokenizer arg.

    Returns True (silent None default), False (required, no default), "absent", or
    "no-init" if the class or method cannot be found.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "AsyncTeacherLLMServerManager"):
            continue
        for item in node.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
                continue
            positional = [a.arg for a in item.args.args[1:]]
            pos_defaults = {}
            if item.args.defaults:
                pos_defaults = dict(zip(positional[-len(item.args.defaults) :], item.args.defaults, strict=True))
            kw_defaults = dict(zip([a.arg for a in item.args.kwonlyargs], item.args.kw_defaults, strict=True))
            if "tokenizer" not in positional and "tokenizer" not in kw_defaults:
                return "absent"
            default = pos_defaults.get("tokenizer", kw_defaults.get("tokenizer"))
            return bool(default is not None and isinstance(default, ast.Constant) and default.value is None)
    return "no-init"


def gate2_tokenizer(verl: Path, rev: str | None) -> tuple[bool, list[str]]:
    notes = []
    src_mgr = read(verl, TEACHER_MANAGER, rev)
    silent_default = _tokenizer_default(ast.parse(src_mgr))
    notes.append(f"AsyncTeacherLLMServerManager.__init__ tokenizer has a silent None default: {silent_default}")

    sites = []
    for rel in (AGENT_LOOP, TEACHER_MODEL):
        for call in _calls_named(_parse(verl, rel, rev), "AsyncTeacherLLMServerManager"):
            sites.append((rel, call.lineno, "tokenizer" in _kwarg_names(call), sorted(_kwarg_names(call))))
    all_pass = bool(sites) and all(p for _, _, p, _ in sites)
    notes.append(f"construction sites found: {len(sites)}; ALL pass tokenizer=: {all_pass}")
    for rel, lineno, passes, kws in sites:
        notes.append(f"  {'PASSES' if passes else '  none'}  {rel}:{lineno} kwargs={kws}")

    if re.search(r"hasattr\(self\.config,\s*['\"]tokenizer['\"]\)", src_mgr):
        from omegaconf import OmegaConf

        probe = OmegaConf.create({"name": "sglang", "multi_turn": {"enable": True}})
        notes.append(
            f"documented fallback self.config.tokenizer resolves on a DictConfig: {hasattr(probe, 'tokenizer')}"
        )

    refuses = "tokenizer=None" in src_mgr and re.search(r"raise ValueError\(\s*\n?\s*message", src_mgr) is not None
    notes.append(f"construction refuses to start when conditioning is on but tokenizer is None: {bool(refuses)}")
    return all_pass and silent_default is False, notes


# ── GATE 3 ────────────────────────────────────────────────────────────────────


def gate3_outcome_signal(verl: Path, rev: str | None) -> tuple[bool, list[str]]:
    notes = []
    tree = _parse(verl, AGENT_LOOP, rev)

    ordered = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        score = [c.lineno for c in _calls_named(node, "_compute_score")]
        teach = [c.lineno for c in _calls_named(node, "_compute_teacher_logprobs")]
        if score and teach:
            ordered = max(score) < min(teach)
            notes.append(
                f"{node.name}(): _compute_score at line {score[0]}, "
                f"_compute_teacher_logprobs at line {teach[0]} -> score computed first: {ordered}"
            )

    consumes = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_compute_teacher_logprobs":
            dumped = ast.dump(node)
            consumes = "'reward_score'" in dumped and "'resolve_privileged_injection'" in dumped
            notes.append(f"streaming hint branch passes reward_score into the injection resolver: {consumes}")
    if not consumes:
        notes.append("streaming hint branch does NOT consume a per-trajectory outcome signal")

    src = read(verl, TEACHER_MANAGER, rev)
    suppresses = "SKIP_SCORE_MISSING" in src
    notes.append(f"a missing score maps to an explicit suppress-injection reason: {suppresses}")
    return ordered and consumes and suppresses, notes


# ── GATE 4 ────────────────────────────────────────────────────────────────────


class _StubTokenizer:
    """1 token per whitespace-separated word; deterministic ids."""

    bos_token = "<bos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return f"<bos><user> {messages[0]['content']} </user>"

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(tok, len(self.vocab) + 1) for tok in text.split()]


def gate4_live(verl: Path, rev: str | None) -> tuple[bool, list[str]]:
    if rev:
        src = read(verl, TEACHER_MANAGER, rev)
        present = "def resolve_privileged_injection" in src
        return present, [
            "a git revision cannot be imported; static presence check only",
            f"resolve_privileged_injection defined at this revision: {present}",
        ]

    notes = []
    sys.path.insert(0, str(verl))
    import verl.experimental.teacher_loop.teacher_manager as tm

    if Path(tm.__file__).resolve() != (verl / TEACHER_MANAGER).resolve():
        return False, [f"imported {tm.__file__}, expected {verl / TEACHER_MANAGER}"]

    saved = (tm._HINT_OPD_ENABLED, tm._OPSD_GOLD_CONDITIONING, dict(tm._HINT_TOKEN_CACHE))
    try:
        tm._HINT_OPD_ENABLED, tm._OPSD_GOLD_CONDITIONING = True, False
        tm._HINT_TOKEN_CACHE.clear()
        # __new__, not __init__: the resolver only needs the two token builders, and
        # __init__ would demand live ray server handles.
        mgr = tm.AsyncTeacherLLMServerManager.__new__(tm.AsyncTeacherLLMServerManager)
        tok = _StubTokenizer()

        right, r_reason = tm.resolve_privileged_injection(mgr, tok, has_multimodal=False, reward_score=1.1)
        wrong, w_reason = tm.resolve_privileged_injection(mgr, tok, has_multimodal=False, reward_score=0.0)
        none_, n_reason = tm.resolve_privileged_injection(mgr, tok, has_multimodal=False, reward_score=None)
        mm, m_reason = tm.resolve_privileged_injection(mgr, tok, has_multimodal=True, reward_score=1.1)

        notes.append(f"reward=1.1  -> {r_reason}, {len(right or [])} tokens")
        notes.append(f"reward=0.0  -> {w_reason}, {len(wrong or [])} tokens")
        notes.append(f"reward=None -> {n_reason}, injected={none_ is not None}")
        notes.append(f"multimodal  -> {m_reason}, injected={mm is not None}")
        distinct = bool(right and wrong and right != wrong)
        notes.append(f"correct and incorrect hints are DIFFERENT token spans: {distinct}")

        ok = (
            r_reason == tm.INJECT_HINT_CORRECT
            and w_reason == tm.INJECT_HINT_INCORRECT
            and distinct
            and none_ is None
            and n_reason == tm.SKIP_SCORE_MISSING
            and mm is None
            and m_reason == tm.SKIP_MULTIMODAL
        )

        # Precedence: with both signals enabled, gold must win and hints must not fire.
        tm._OPSD_GOLD_CONDITIONING = True
        _, p_reason = tm.resolve_privileged_injection(
            mgr, tok, has_multimodal=False, gold_text="the answer is C", reward_score=1.1
        )
        notes.append(f"gold+hint both enabled -> {p_reason} (gold must take precedence)")
        return bool(ok and p_reason == tm.INJECT_GOLD), notes
    finally:
        tm._HINT_OPD_ENABLED, tm._OPSD_GOLD_CONDITIONING, cache = saved
        tm._HINT_TOKEN_CACHE.clear()
        tm._HINT_TOKEN_CACHE.update(cache)


# ── SCOPE ─────────────────────────────────────────────────────────────────────


def scope_multimodal(verl: Path, rev: str | None, data: str) -> tuple[bool, list[str]]:
    """Multimodal samples are skipped by design. Report how many that actually is."""
    notes = []
    src = read(verl, TEACHER_MANAGER, rev)
    per_sample = "has_multimodal" in src
    notes.append(f"both teacher paths skip injection per-sample for image/video samples: {per_sample}")

    total, mm = None, None
    try:
        import pandas as pd

        df = pd.read_parquet(data)
        total = len(df)
        columns = set(df.columns)

        def _mm(row) -> bool:
            # Explicit None checks throughout: prompt/images come back as numpy
            # object arrays, whose truthiness raises rather than being falsy-empty.
            if "images" in columns:
                images = row["images"]
                if images is not None and len(images) > 0:
                    return True
            prompt = row["prompt"] if "prompt" in columns else None
            for msg in [] if prompt is None else list(prompt):
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, (list, tuple)):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("image", "image_url", "video"):
                            return True
            return False

        mm = int(df.apply(_mm, axis=1).sum())
        notes.append(f"configured training data {data}")
        notes.append(f"  {mm}/{total} rows multimodal ({100.0 * mm / total if total else 0.0:.1f}%)")
        notes.append(f"  samples the multimodal restriction suppresses in these runs: {mm}")
    except Exception as exc:  # data not staged on this host
        notes.append(f"could not measure multimodal fraction ({type(exc).__name__}: {exc})")

    blocks_everything = total is not None and total > 0 and mm == total
    if blocks_everything:
        notes.append("the restriction suppresses EVERY sample; promoting it to a blocking gate")
    return (per_sample and not blocks_everything), notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verl", default="/data/project/private/minstar/workspace/verl_ttopd")
    ap.add_argument("--rev", default=None, help="git revision to inspect (default: working tree)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="training parquet, for the multimodal scope measurement")
    args = ap.parse_args()

    verl = Path(args.verl)
    if not (verl / ".git").exists():
        print(f"[fatal] {verl} is not a git checkout", file=sys.stderr)
        return 2

    print(f"inspecting {verl} @ {args.rev or 'working tree'}\n")

    results = []
    for name, fn in (
        ("GATE 1  streaming teacher path forwards hints", gate1_teacher_path),
        ("GATE 2  a tokenizer reaches the hint builder", gate2_tokenizer),
        ("GATE 3  the outcome signal is available where the hint is chosen", gate3_outcome_signal),
        ("GATE 4  the resolver produces distinct hints (live)", gate4_live),
    ):
        passed, notes = fn(verl, args.rev)
        results.append((name, passed))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        for n in notes:
            print(f"        {n}")
        print()

    scope_ok, scope_notes = scope_multimodal(verl, args.rev, args.data)
    print(f"[{'SCOPE' if scope_ok else 'FAIL '}] multimodal samples are skipped by design")
    for n in scope_notes:
        print(f"        {n}")
    print()
    if not scope_ok:
        results.append(("SCOPE  multimodal restriction suppresses every sample", False))

    print("=" * 72)
    print("With distillation.teacher_model.enable_resource_pool=True (v26-v32 run scripts),")
    print("the streaming path is taken, so GATE 1 decides. Without it (v15-v25), the batch")
    print("path is taken and GATE 2 decides. GATE 3 and GATE 4 apply to both.")
    print()
    if all(p for _, p in results):
        print("VERDICT: outcome-conditioned privileged hints CAN fire.")
        return 0
    print("VERDICT: hints CANNOT fire. Failing gates:")
    for n, p in results:
        if not p:
            print(f"  - {n}")
    print()
    print("TT-OPD as actually executed = GRPO + cosine length reward + EMA teacher")
    print("+ turn-level truncation, WITHOUT outcome-conditioned privileged hints.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
