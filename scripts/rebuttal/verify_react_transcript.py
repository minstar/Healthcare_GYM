#!/usr/bin/env python
"""Verify ReAct transcript fidelity + format-adherence instrumentation.

Rebuttal ("ReAct-style prompting with the same tools"). An arm
that is not actually ReAct cannot be reported as evidence, so this suite drives
the REAL multi-turn loops with a scripted model and checks what the model
actually sees.

Checks
  1. react_transcript_agent_runner
        AgentRunner.run_task under prompt_mode="react" replays the assistant's
        turn verbatim as ReAct and returns tool results prefixed "Observation:".
  2. react_transcript_eval_script
        Same for scripts/eval_benchmark_multiturn._run_single_task_multiturn,
        which is the loop the benchmark eval actually runs.
  3. hallucinated_observation_not_replayed
        A ReAct model that self-continues past its own Action Input (writing a
        fabricated "Observation:" and a further Action) must not have that tail
        replayed into the context — verbatim replay would otherwise let the
        model feed itself invented tool output.
  4. default_transcript_unchanged
        default and strong_tool transcripts are byte-for-byte identical to the
        anchors'. Each anchor's agent_runner.py is loaded straight out of git
        (PRE_COMMIT, the tree this change landed on, and BASE_COMMIT, the
        commit that shipped the Base+AR row) as live modules and driven over
        the same task with the same scripted model, so this is a real diff
        against the shipped baseline, not against a re-implementation of it.
        The same comparison asserts the react transcript DID change — against
        each anchor in the half that anchor's fix landed in. Also sweeps every
        domain x tool set x task mode and requires the default / strong_tool
        SYSTEM PROMPTS to be byte-identical to both anchors.
  5. parse_tool_call_unchanged
        The current parse_tool_call returns exactly what the anchors' return,
        over every format it accepts plus 20k randomized non-tool-call outputs
        (the refactor added a format label; it must not have changed a single
        parse). Compared against their real functions, loaded from git. Covers
        the training path too, since grpo_trainer delegates to this function.
  6. adherence_metric_in_output
        The adherence metric is computed and lands in the written results JSON,
        for react (1.00) and default (0.00) alike, and a model that IGNORES the
        react format is reported as react_rate 0.00 rather than silently
        scored as a ReAct result.
  7. react_tool_catalog
        The react system prompt carries the FULL tool specification — every
        tool name, every argument name, every argument type that the other
        modes receive via apply_chat_template(tools=...) — byte-for-byte
        identical to the <tools> block the real chat template would inject,
        across all 10 domains. Not a summary, not a subset.
  8. react_native_contract_suppressed
        Rendered through the REAL Qwen3.5 chat template: the react prompt
        contains no native tool-call contract (no "<tool_call>" / "<function="
        instruction block), while default's still does — and default's fully
        rendered prompt string is byte-identical to PRE_COMMIT's.

Run:
    PYTHONPATH=<repo> .venv/bin/python scripts/rebuttal/verify_react_transcript.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import string
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioagents.evaluation.agent_runner import (  # noqa: E402
    AgentRunner,
    RunConfig,
    parse_tool_call,
    parse_tool_call_with_format,
    summarize_format_adherence,
)

DOMAIN = "cross_domain"

# Byte-identity anchors for the modes that must not move.
#   BASE_COMMIT — the commit that shipped the Base+AR row of the results table.
#   PRE_COMMIT  — the tree the sole-contract change was applied to, i.e.
#                 everything merged between the two.
# Both are checked, so "default did not move" holds against the number in the
# paper AND against the tree this change landed on.
#
# These are pinned hashes, never the symbolic "HEAD": once the change is
# committed, HEAD is this change, and every comparison against it would pass
# vacuously. An anchor has to name a tree that predates what it is guarding.
BASE_COMMIT = "dd962b9"
PRE_COMMIT = "10fdc57"
ANCHORS = (PRE_COMMIT, BASE_COMMIT)

# Tokenizer/template only — no weights are loaded, no GPU is used. Used to
# render the prompt exactly as the eval backend does.
TOKENIZER_PATH = "/data/project/private/minstar/workspace/hcgym_rebuttal/models/Qwen3.5-9B"

_ANCHOR_MODULES: dict = {}


def head_module(ref: str = PRE_COMMIT):
    """Load a git ref's bioagents/evaluation/agent_runner.py as a live module.

    Imported alongside the current one, under a distinct module name, so the
    two implementations can be driven over the same task and diffed. Nothing is
    stubbed or re-implemented — this is the code that shipped.
    """
    if ref in _ANCHOR_MODULES:
        return _ANCHOR_MODULES[ref]
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         f"{ref}:bioagents/evaluation/agent_runner.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    name = "_agent_runner_" + "".join(c if c.isalnum() else "_" for c in ref)
    tmp = Path(tempfile.mkdtemp()) / f"{name}.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _ANCHOR_MODULES[ref] = mod
    return mod

# A scripted model that emits canonical ReAct on every turn.
REACT_SCRIPT = [
    'Thought: I need evidence on the sepsis bundle before I can answer.\n'
    'Action: search_evidence\n'
    'Action Input: {"query": "sepsis bundle", "max_results": 2}',

    'Thought: I should also check the guideline text itself.\n'
    'Action: search_guidelines\n'
    'Action Input: {"condition": "sepsis"}',

    'Thought: I now have enough evidence to answer.\n'
    'Action: submit_answer\n'
    'Action Input: {"answer": "A", "reasoning": "One-hour bundle complete."}',
]

# The same three actions expressed as bare JSON — what a default-mode model
# emits. Used to prove the default transcript did not move.
JSON_SCRIPT = [
    '{"name": "search_evidence", "arguments": {"query": "sepsis bundle", "max_results": 2}}',
    '{"name": "search_guidelines", "arguments": {"condition": "sepsis"}}',
    '{"name": "submit_answer", "arguments": {"answer": "A", "reasoning": "One-hour bundle complete."}}',
]


class ScriptedRunner(AgentRunner):
    """AgentRunner whose generate() replays a fixed script and records inputs.

    ``self.seen[i]`` is a deep copy of the message list passed to the i-th
    generate() call, i.e. exactly the transcript the model saw.
    """

    def __init__(self, config: RunConfig, script: list[str]):
        super().__init__(config)
        self._script = list(script)
        self._i = 0
        self.seen: list[list[dict]] = []

    def load_model(self):  # no GPU, no weights
        pass

    def generate(self, messages, tools=None):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        out = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return out


def _mk(prompt_mode: str, script: list[str], log_dir: str) -> ScriptedRunner:
    cfg = RunConfig(
        model_name_or_path="scripted-mock",
        domain=DOMAIN,
        max_turns=6,
        log_dir=log_dir,
        prompt_mode=prompt_mode,
    )
    return ScriptedRunner(cfg, script)


def _env():
    from bioagents.gym.agent_env import BioAgentGymEnv

    return BioAgentGymEnv(domain=DOMAIN, max_turns=6)


def _first_task(env):
    return env._tasks[0]


def _conversation(messages: list[dict]) -> list[dict]:
    """Messages excluding the system prompt — what the LOOP put in the context.

    The system prompt is excluded on purpose: the react system prompt contains
    the literal string "Observation:" inside its worked example, so including
    it would make the "did the environment emit an Observation?" check pass
    vacuously.
    """
    return [m for m in messages if m.get("role") != "system"]


def _system(messages: list[dict]) -> str:
    """The system prompt the loop built, or "" if there is none."""
    for m in messages:
        if m.get("role") == "system":
            return _text(m)
    return ""


def _text(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def render_transcript(messages: list[dict], width: int = 300) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?").upper()
        c = _text(m)
        if role == "SYSTEM":
            out.append(f"[{role}] <{len(c)} chars of system prompt, elided>")
            continue
        out.append(f"[{role}] {c if len(c) <= width else c[:width] + ' [...]'}")
    return "\n".join(out)


def transcript_facts(messages: list[dict]) -> dict:
    """The three yes/no facts the defect report was stated in."""
    conv = _conversation(messages)
    asst = [_text(m) for m in conv if m["role"] == "assistant"]
    joined = "\n".join(_text(m) for m in conv)
    return {
        "n_assistant_turns": len(asst),
        "react_text": bool(asst) and all("Thought:" in a and "Action:" in a for a in asst),
        "bare_json": bool(asst) and all(a.strip().startswith('{"name"') for a in asst),
        "observation_present": "Observation:" in joined,
        "tool_result_prefix_present": "Tool result for " in joined,
    }


def run_head_loop(prompt_mode: str, script: list[str],
                  ref: str = PRE_COMMIT) -> list[dict]:
    """Drive an anchor's AgentRunner.run_task over the same task and return the
    transcript its model saw on the final generate() call."""
    H = head_module(ref)

    class HeadScripted(H.AgentRunner):
        def __init__(self, config, sc):
            super().__init__(config)
            self._sc = list(sc)
            self._i = 0
            self.seen = []

        def load_model(self):
            pass

        def generate(self, messages, tools=None):
            self.seen.append(json.loads(json.dumps(messages, default=str)))
            out = self._sc[min(self._i, len(self._sc) - 1)]
            self._i += 1
            return out

    with tempfile.TemporaryDirectory() as td:
        cfg = H.RunConfig(
            model_name_or_path="scripted-mock",
            domain=DOMAIN,
            max_turns=6,
            log_dir=td,
            prompt_mode=prompt_mode,
        )
        runner = HeadScripted(cfg, script)
        env = _env()
        runner.run_task(_first_task(env), env)
        return runner.seen[-1]


# --------------------------------------------------------------------------- #
#  Checks
# --------------------------------------------------------------------------- #
def check_react_transcript_agent_runner(verbose: bool = False):
    with tempfile.TemporaryDirectory() as td:
        runner = _mk("react", REACT_SCRIPT, td)
        env = _env()
        runner.run_task(_first_task(env), env)
        final = runner.seen[-1]

    if verbose:
        print("\n--- AgentRunner.run_task, prompt_mode=react: transcript on the "
              "FINAL generate() call ---")
        print(render_transcript(final))

    f = transcript_facts(final)
    if verbose:
        print(f"\nassistant turns replayed as ReAct text ? {f['react_text']}")
        print(f"assistant turns replayed as bare JSON  ? {f['bare_json']}")
        print(f"string 'Observation:' ever appears     ? {f['observation_present']}")

    if f["n_assistant_turns"] < 2:
        return False, f"expected >=2 assistant turns, got {f['n_assistant_turns']}"
    if not f["react_text"]:
        return False, "assistant turns were NOT replayed as ReAct text"
    if f["bare_json"]:
        return False, "assistant turns were rewritten into bare JSON"
    if not f["observation_present"]:
        return False, "'Observation:' never appears in the conversation"
    if f["tool_result_prefix_present"]:
        return False, "react transcript still uses the 'Tool result for' prefix"
    return True, (f"{f['n_assistant_turns']} assistant turns replayed verbatim as "
                  "ReAct; tool results returned as 'Observation:'")


def check_react_transcript_eval_script(verbose: bool = False):
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_benchmark_multiturn import _run_single_task_multiturn_debug

    with tempfile.TemporaryDirectory() as td:
        runner = _mk("react", REACT_SCRIPT, td)
        env = _env()
        task = _first_task(env)
        env._task_map[task["id"]] = task
        (_turns, _sub, _lat, _log), final = _run_single_task_multiturn_debug(
            runner, task, env, 6
        )

    if verbose:
        print("\n--- eval_benchmark_multiturn._run_single_task_multiturn, "
              "prompt_mode=react: transcript on the FINAL generate() call ---")
        print(render_transcript(final))

    f = transcript_facts(final)
    if verbose:
        print(f"\nassistant turns replayed as ReAct text ? {f['react_text']}")
        print(f"assistant turns replayed as bare JSON  ? {f['bare_json']}")
        print(f"string 'Observation:' ever appears     ? {f['observation_present']}")

    if not f["react_text"] or f["bare_json"]:
        return False, "eval-script loop did not replay assistant turns as ReAct"
    if not f["observation_present"] or f["tool_result_prefix_present"]:
        return False, "eval-script loop did not return results as 'Observation:'"
    return True, (f"{f['n_assistant_turns']} assistant turns replayed verbatim as "
                  "ReAct; tool results returned as 'Observation:'")


def check_default_transcript_unchanged(verbose: bool = False):
    """default / strong_tool must be byte-for-byte what the anchors produced.

    Diffed against the anchors' real AgentRunner (loaded from git: PRE_COMMIT,
    the tree this change landed on, and BASE_COMMIT, the commit that shipped
    the Base+AR row), driven over the
    same task with the same scripted model — including the system prompt, so
    this also re-verifies that the default (Base+AR) prompt did not move.
    react is asserted to DIFFER from each anchor in the half that anchor's fix
    landed in.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_benchmark_multiturn import _run_single_task_multiturn_debug

    unchanged, changed = [], []
    for mode, script, must_match in (
        ("default", JSON_SCRIPT, True),
        ("strong_tool", JSON_SCRIPT, True),
        ("react", REACT_SCRIPT, False),
    ):
        with tempfile.TemporaryDirectory() as td:
            runner = _mk(mode, script, td)
            env = _env()
            runner.run_task(_first_task(env), env)
            now = runner.seen[-1]

        for ref in ANCHORS:
            head = run_head_loop(mode, script, ref=ref)
            # must_match arms compare the WHOLE message list (system prompt
            # included, so a Base+AR prompt drift would also be caught).
            same = json.dumps(head, sort_keys=True) == json.dumps(now, sort_keys=True)
            conv_same = (json.dumps(_conversation(head), sort_keys=True)
                         == json.dumps(_conversation(now), sort_keys=True))
            sys_same = _system(head) == _system(now)
            if must_match and not same:
                diff = [
                    (i, (h.get("role"), _text(h)[:120]), (n.get("role"), _text(n)[:120]))
                    for i, (h, n) in enumerate(zip(head, now))
                    if json.dumps(h, sort_keys=True) != json.dumps(n, sort_keys=True)
                ]
                return False, (f"prompt_mode={mode} transcript CHANGED vs {ref}; "
                               f"first diffs: {diff[:2]}")
            # The react arm must differ from BOTH anchors, but in the place each
            # fix actually landed. Asserting the wrong half against the wrong
            # anchor is how a reverted fix hides behind an unrelated diff:
            #   vs BASE_COMMIT — the CONVERSATION must differ (verbatim ReAct
            #       replay + "Observation:" prefix, which landed after it).
            #   vs PRE_COMMIT  — the SYSTEM PROMPT must differ (this change:
            #       tool catalog rendered in-prompt, native contract dropped).
            if not must_match:
                if ref == BASE_COMMIT and conv_same:
                    return False, (f"prompt_mode={mode} CONVERSATION is identical "
                                   f"to {ref} — the transcript fix is not in effect")
                if ref == PRE_COMMIT and sys_same:
                    return False, (f"prompt_mode={mode} SYSTEM PROMPT is identical "
                                   f"to {ref} — the sole-contract fix did not "
                                   f"take effect")
            if verbose and must_match:
                print(f"\n--- prompt_mode={mode}: current vs {ref} "
                      f"(AgentRunner.run_task, {len(now)} messages) ---")
                print(f"identical (incl. system prompt): {same}")
        (unchanged if must_match else changed).append(mode)

    # The eval-script loop shares the same helpers; check it separately against
    # the anchors' rendering contract, since their copy of that script is a
    # different file.
    for mode in ("default", "strong_tool"):
        with tempfile.TemporaryDirectory() as td:
            runner = _mk(mode, JSON_SCRIPT, td)
            env = _env()
            task = _first_task(env)
            env._task_map[task["id"]] = task
            _out, final = _run_single_task_multiturn_debug(runner, task, env, 6)
        conv = _conversation(final)
        asst = [_text(m) for m in conv if m["role"] == "assistant"]
        expected = [json.dumps(parse_tool_call(r)) for r in JSON_SCRIPT[:len(asst)]]
        if asst != expected:
            return False, (f"eval_script/{mode}: assistant turns changed\n"
                           f"  before: {expected}\n  now   : {asst}")
        obs = [_text(m) for m in conv if m["role"] == "user"][1:]
        if any(not o.startswith("Tool result for ") for o in obs):
            return False, f"eval_script/{mode}: tool-result prefix changed"
        if "Observation:" in "\n".join(obs):
            return False, f"eval_script/{mode}: 'Observation:' leaked into {mode}"
        unchanged.append(f"eval_script/{mode}")

    # Transcript rendering is domain-independent, but the SYSTEM PROMPT is not:
    # sweep every domain x tool-set x task-mode and require the default and
    # strong_tool prompts to be byte-identical to both. This is what protects
    # the Base+AR row of the results table.
    from bioagents.evaluation.agent_runner import build_system_prompt
    from bioagents.gym.agent_env import BioAgentGymEnv, _DOMAIN_REGISTRY, _load_default_domains

    _load_default_domains()
    n_render = 0
    for dom in sorted(_DOMAIN_REGISTRY):
        env = BioAgentGymEnv(domain=dom, max_turns=6)
        if not env._tasks:
            continue
        task = env._tasks[0]
        _obs, info = env.reset(options={"task_id": task["id"]})
        no_think_tools = [
            t for t in info["tools"]
            if t.get("function", {}).get("name") != "think"
        ]
        for tools in (info["tools"], no_think_tools):
            for tk in (task, None):
                for mode in ("default", "strong_tool"):
                    a = build_system_prompt(info["policy"], tools, domain=dom,
                                            task=tk, prompt_mode=mode)
                    for ref in ANCHORS:
                        b = head_module(ref).build_system_prompt(
                            info["policy"], tools, domain=dom,
                            task=tk, prompt_mode=mode)
                        n_render += 1
                        if a != b:
                            return False, (f"system prompt CHANGED vs {ref}: "
                                           f"domain={dom} mode={mode} "
                                           f"task={'yes' if tk else 'no'} "
                                           f"n_think={len(tools)}")
    if verbose:
        print(f"\nsystem prompts: {n_render} comparisons "
              f"(10 domains x 2 tool sets x 2 task modes x 2 modes x "
              f"{len(ANCHORS)} anchors) byte-identical")

    return True, (f"byte-identical to {' + '.join(ANCHORS)} for "
                  f"{', '.join(unchanged)} + {n_render} default/strong_tool "
                  f"system-prompt comparisons; "
                  f"changed (as intended) for {', '.join(changed)}")


def check_parse_tool_call_unchanged(verbose: bool = False):
    """The parse_tool_call refactor must not have changed a single parse.

    Diffed against the anchors' real parse_tool_call. Corpus: one probe per accepted
    format + 20k randomized non-tool-call outputs.
    """
    head_parses = {ref: head_module(ref).parse_tool_call for ref in ANCHORS}
    corpus = [
        # xml_tool_call
        '<tool_call>{"name": "search_evidence", "arguments": {"query": "x"}}</tool_call>',
        '<|tool_call|>{"name": "think", "arguments": {"thought": "y"}}<|/tool_call|>',
        # xml_qwen35
        '<tool_call><function=search_evidence><parameter=query>sepsis</parameter>'
        '</function></tool_call>',
        # react_strict
        'Thought: t\nAction: search_evidence\nAction Input: {"query": "sepsis"}',
        '**Action:** think\n**Action Input:** {"thought": "z"}\nObservation: ignored',
        # json_code_block
        '```json\n{"name": "submit_answer", "arguments": {"answer": "A"}}\n```',
        # json_direct
        '{"name": "search_guidelines", "arguments": {"condition": "sepsis"}}',
        # alt keys
        '{"tool": "think", "args": {"thought": "q"}}',
        '{"action": "search_evidence", "action_input": {"query": "q"}}',
        '{"function": "think", "parameters": {"thought": "q"}}',
        # json_embedded / json_scan
        'Sure, here you go: {"name": "think", "arguments": {"thought": "a"}} done.',
        # none
        'The answer is A because the bundle was completed within the hour.',
        'Observation: nothing to do here.',
        '',
    ]
    rng = random.Random(20260728)
    alphabet = string.ascii_letters + string.digits + ' .,:;{}[]"\'\n_-'
    for _ in range(20000):
        corpus.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 160))))

    mismatches = []
    for text in corpus:
        now = parse_tool_call(text)
        for ref, head_parse in head_parses.items():
            head = head_parse(text)
            if now != head:
                mismatches.append((ref, text[:60], head, now))
        _call, label = parse_tool_call_with_format(text)
        if _call != now:
            return False, f"wrapper disagrees with impl on {text[:60]!r}"
        if (now is None) != (label == "none"):
            return False, f"label/parse disagree on {text[:60]!r}: {label} vs {now}"
    if mismatches:
        return False, (f"{len(mismatches)} parses differ from an anchor; "
                       f"first: {mismatches[:2]}")

    # Every label must be a declared one, and the react probes must be labelled
    # as ReAct rather than absorbed by another branch.
    from bioagents.evaluation.agent_runner import TOOL_CALL_FORMATS, REACT_FORMATS

    labels = [parse_tool_call_with_format(t)[1] for t in corpus[:14]]
    bad = [x for x in labels if x not in TOOL_CALL_FORMATS]
    if bad:
        return False, f"undeclared format labels: {bad}"
    if labels[3] not in REACT_FORMATS or labels[4] not in REACT_FORMATS:
        return False, f"ReAct probes mislabelled: {labels[3]!r}, {labels[4]!r}"
    if labels[6] != "json_direct" or labels[5] != "json_code_block":
        return False, f"JSON probes mislabelled: {labels[5]!r}, {labels[6]!r}"
    if verbose:
        print(f"\nformat labels for the 14 hand-written probes: {labels}")
    return True, (f"{len(corpus)} outputs parse identically to "
                  f"{' + '.join(ANCHORS)} (incl. 20000 randomized); "
                  f"labels well-formed")


def check_adherence_metric_in_output(verbose: bool = False):
    """The metric must be computed AND land in the written results JSON."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_benchmark_multiturn as EB

    seen = {}
    for mode, script, expect_react in (
        ("react", REACT_SCRIPT, True),
        ("default", JSON_SCRIPT, False),
    ):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir()
            runner = _mk(mode, script, td)
            env = _env()
            task = dict(_first_task(env))
            task["correct_answer"] = "A"
            summary = EB.run_benchmark_multiturn(
                benchmark_name="mock",
                tasks=[task, dict(task, id=task["id"] + "_b")],
                runner=runner,
                domain=DOMAIN,
                max_turns=6,
                output_dir=out_dir,
            )
            written = sorted(out_dir.glob("mock_multiturn_*.json"))
            if not written:
                return False, "no results JSON written"
            on_disk = json.loads(written[-1].read_text())

        for obj, where in ((summary, "summary"), (on_disk, "results JSON")):
            if "format_adherence" not in obj:
                return False, f"{mode}: format_adherence missing from {where}"
            if "react_rate" not in obj["format_adherence"]:
                return False, f"{mode}: react_rate missing from {where}"
            for r in obj["results"]:
                if "react_rate" not in r:
                    return False, f"{mode}: per-task react_rate missing from {where}"

        fa = on_disk["format_adherence"]
        seen[mode] = fa
        if verbose:
            print(f"\nprompt_mode={mode}: results-JSON format_adherence = "
                  f"{json.dumps(fa, ensure_ascii=False)}")
            print(f"  per-task react_rate = "
                  f"{[r['react_rate'] for r in on_disk['results']]}")

        if expect_react and fa["react_rate"] != 1.0:
            return False, f"react mode react_rate={fa['react_rate']}, expected 1.0"
        if not expect_react and fa["react_rate"] != 0.0:
            return False, f"default mode react_rate={fa['react_rate']}, expected 0.0"
        if fa["n_turns"] == 0:
            return False, f"{mode}: n_turns == 0"

    # The metric's whole purpose: a model that IGNORES the mandated format
    # under prompt_mode="react" must be visible as such, not silently scored
    # as a ReAct result.
    with tempfile.TemporaryDirectory() as td:
        runner = _mk("react", JSON_SCRIPT, td)  # react prompt, JSON-emitting model
        env = _env()
        res = runner.run_task(_first_task(env), env)
        ignored = res.format_adherence
        if summarize_format_adherence(res.turns)["react_rate"] != ignored["react_rate"]:
            return False, "TaskResult.format_adherence disagrees with its own turns"
    if ignored["react_rate"] != 0.0:
        return False, (f"a JSON-emitting model under prompt_mode=react reported "
                       f"react_rate={ignored['react_rate']}, expected 0.0")
    if verbose:
        print(f"\nnon-adherent model under prompt_mode=react: "
              f"{json.dumps(ignored, ensure_ascii=False)}")

    return True, (f"react react_rate={seen['react']['react_rate']:.2f}, "
                  f"default react_rate={seen['default']['react_rate']:.2f}, "
                  "non-adherent react-mode model detected (react_rate=0.00)")


def check_hallucinated_observation_not_replayed(verbose: bool = False):
    """A self-continued turn must not put invented tool output in the context.

    Neither backend can stop generation at "Observation:", so a ReAct model
    routinely writes its own Observation and a further Action that the
    environment never executed. Replaying that verbatim would let the model
    feed itself fabricated evidence.
    """
    fabricated = "FABRICATED_OBSERVATION_MUST_NOT_APPEAR"
    script = [
        'Thought: I need evidence first.\n'
        'Action: search_evidence\n'
        'Action Input: {"query": "sepsis bundle", "max_results": 2}\n'
        f'Observation: {fabricated}\n'
        'Thought: Great, that settles it.\n'
        'Action: submit_answer\n'
        'Action Input: {"answer": "Z", "reasoning": "never executed"}',

        'Thought: I now have enough evidence to answer.\n'
        'Action: submit_answer\n'
        'Action Input: {"answer": "A", "reasoning": "grounded"}',
    ]
    with tempfile.TemporaryDirectory() as td:
        runner = _mk("react", script, td)
        env = _env()
        res = runner.run_task(_first_task(env), env)
        final = runner.seen[-1]

    conv = _conversation(final)
    joined = "\n".join(_text(m) for m in conv)
    asst = [_text(m) for m in conv if m["role"] == "assistant"]

    if verbose:
        print("\n--- self-continued ReAct turn: what is actually replayed ---")
        print(render_transcript(conv, width=400))

    if fabricated in joined:
        return False, "the model's INVENTED Observation was replayed into the context"
    if not asst or "search_evidence" not in asst[0]:
        return False, f"executed action missing from replay: {asst[:1]}"
    if "never executed" in joined:
        return False, "an action the environment never executed was replayed"
    # The real observation still arrives, from the environment.
    obs = [_text(m) for m in conv if m["role"] == "user" and _text(m).startswith("Observation:")]
    if not obs:
        return False, "no environment Observation in the transcript"
    executed = [t.parsed_tool_call.get("name") for t in res.turns if t.parsed_tool_call]
    if executed[:1] != ["search_evidence"]:
        return False, f"wrong action executed: {executed}"
    return True, (f"self-continued tail dropped; replayed turn = executed triple; "
                  f"{len(obs)} environment Observation(s); executed={executed}")


def _domains_with_tasks():
    """Yield (domain, task, policy, tools) for every registered domain."""
    from bioagents.gym.agent_env import (
        BioAgentGymEnv, _DOMAIN_REGISTRY, _load_default_domains,
    )

    _load_default_domains()
    for dom in sorted(_DOMAIN_REGISTRY):
        env = BioAgentGymEnv(domain=dom, max_turns=6)
        if not env._tasks:
            continue
        task = env._tasks[0]
        _obs, info = env.reset(options={"task_id": task["id"]})
        yield dom, task, info["policy"], info["tools"]


def _template_tools_block(tok, tools) -> str:
    """The <tools> block the REAL chat template injects, extracted verbatim."""
    import re

    txt = tok.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}],
        tools=tools, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    m = re.search(r"<tools>\n.*?\n</tools>", txt, re.S)
    return m.group(0) if m else ""


def check_react_tool_catalog(verbose: bool = False):
    """The react prompt must carry the FULL spec — same tools, names, schemas.

    Two independent assertions, over all 10 domains:
      (a) every tool name and every argument name in the live registry appears
          in the react system prompt, with its declared type;
      (b) the catalog block in the prompt is byte-for-byte the <tools> block
          the real Qwen3.5 chat template would have injected — so this is a
          relocation of the specification, not a re-description of it.
    Also reports the token cost, so "the full render is too long" would be a
    number rather than a silent truncation.
    """
    from bioagents.evaluation.agent_runner import (
        build_system_prompt, render_tool_catalog,
    )

    tok = None
    if Path(TOKENIZER_PATH).exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    rows, n_tools_total, n_args_total = [], 0, 0
    for dom, task, policy, tools in _domains_with_tasks():
        prompt = build_system_prompt(policy, tools, domain=dom, task=task,
                                     prompt_mode="react")
        for spec in tools:
            fn = spec.get("function", spec)
            name = fn.get("name")
            n_tools_total += 1
            if f'"name": "{name}"' not in prompt:
                return False, f"{dom}: tool {name!r} missing from react prompt"
            props = ((fn.get("parameters") or {}).get("properties") or {})
            for arg, schema in props.items():
                n_args_total += 1
                if f'"{arg}": {{' not in prompt:
                    return False, (f"{dom}: argument {name}.{arg!r} missing "
                                   f"from react prompt")
                atype = (schema or {}).get("type")
                if atype and f'"{arg}": {{"type": "{atype}"' not in prompt:
                    return False, (f"{dom}: argument {name}.{arg} lost its "
                                   f"declared type {atype!r}")

        block = render_tool_catalog(tools)
        if block not in prompt:
            return False, f"{dom}: rendered catalog block absent from react prompt"
        if tok is not None:
            native = _template_tools_block(tok, tools)
            if not native:
                return False, f"{dom}: could not extract template <tools> block"
            if native != block:
                return False, (f"{dom}: in-prompt catalog differs from the "
                               f"template's <tools> block "
                               f"({len(block)} vs {len(native)} chars)")
        n_tok = len(tok(prompt)["input_ids"]) if tok is not None else -1
        rows.append((dom, len(tools), len(block), n_tok))

    if verbose:
        print(f"\n{'domain':22} {'n_tools':>8} {'catalog chars':>14} "
              f"{'react sys-prompt tokens':>24}")
        for r in rows:
            print(f"{r[0]:22} {r[1]:8d} {r[2]:14d} {r[3]:24d}")

    tokens = [r[3] for r in rows if r[3] > 0]
    tok_note = (f"; react system prompt {min(tokens)}-{max(tokens)} tokens"
                if tokens else "; tokenizer unavailable, token cost not measured")
    return True, (f"{n_tools_total} tools / {n_args_total} arguments across "
                  f"{len(rows)} domains present verbatim; catalog byte-identical "
                  f"to the chat template's <tools> block{tok_note}")


def check_react_native_contract_suppressed(verbose: bool = False):
    """Rendered through the REAL template: react has no native call contract.

    The template injects the catalog and the native contract together, so this
    renders the final prompt string the backend actually sends and asserts:
      - react   : no "<tool_call>" / "<function=" instruction block, and the
                  ReAct contract present;
      - default : the native contract still present, and the WHOLE rendered
                  string byte-identical to PRE_COMMIT's — the strongest form of the
                  "default did not move" claim, since it covers the template
                  layer as well as build_system_prompt.
    """
    if not Path(TOKENIZER_PATH).exists():
        return False, f"tokenizer not found at {TOKENIZER_PATH}"

    import bioagents.evaluation.agent_runner as CUR
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    H = head_module(PRE_COMMIT)

    def render(mod, dom, policy, tools, task, mode):
        """Render exactly what the backend sends, using ``mod``'s own gate.

        PRE_COMMIT has no native_tools_for_prompt_mode — it always passed the tool
        list through — so the anchor's behaviour is reproduced by falling back
        to the identity gate rather than by applying the current one to it.
        """
        sysp = mod.build_system_prompt(policy, tools, domain=dom, task=task,
                                       prompt_mode=mode)
        gate = getattr(mod, "native_tools_for_prompt_mode",
                       lambda t, _m: t)
        native = gate(tools, mode)
        return tok.apply_chat_template(
            [{"role": "system", "content": sysp},
             {"role": "user", "content": "TASK TICKET"}],
            tools=native if native else None,
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    NATIVE_MARKERS = ("<function=example_function_name>",
                      "an inner <function=...></function> block")
    n_dom, deltas = 0, []
    for dom, task, policy, tools in _domains_with_tasks():
        n_dom += 1
        cur_react = render(CUR, dom, policy, tools, task, "react")
        old_react = render(H, dom, policy, tools, task, "react")

        for marker in NATIVE_MARKERS:
            if marker in cur_react:
                return False, (f"{dom}: react prompt STILL contains the native "
                               f"tool-call contract ({marker!r})")
            if marker not in old_react:
                return False, (f"{dom}: control failed — {PRE_COMMIT}'s react "
                               f"prompt should contain {marker!r}")
        if "Action Input:" not in cur_react:
            return False, f"{dom}: react prompt lost its ReAct contract"

        # default must keep the native contract AND be byte-identical to PRE_COMMIT's
        # at the fully-rendered level.
        cur_def = render(CUR, dom, policy, tools, task, "default")
        old_def = render(H, dom, policy, tools, task, "default")
        if cur_def != old_def:
            return False, (f"{dom}: DEFAULT rendered prompt changed vs "
                           f"{PRE_COMMIT} ({len(old_def)} -> {len(cur_def)} chars)")
        if NATIVE_MARKERS[0] not in cur_def:
            return False, f"{dom}: default prompt lost the native contract"

        deltas.append((dom,
                       len(tok(old_react)["input_ids"]),
                       len(tok(cur_react)["input_ids"])))

    if verbose:
        print(f"\n{'domain':22} {'old react tokens':>18} {'new react tokens':>18} "
              f"{'delta':>8}")
        for d, o, n in deltas:
            print(f"{d:22} {o:18d} {n:18d} {n - o:+8d}")

    return True, (f"{n_dom} domains: react prompt has no native tool-call "
                  f"contract and keeps the ReAct one; default rendered prompt "
                  f"byte-identical to {PRE_COMMIT}; react token delta "
                  f"{min(n - o for _, o, n in deltas):+d}.."
                  f"{max(n - o for _, o, n in deltas):+d}")


CHECKS = {
    "react_transcript_agent_runner": check_react_transcript_agent_runner,
    "react_transcript_eval_script": check_react_transcript_eval_script,
    "hallucinated_observation_not_replayed": check_hallucinated_observation_not_replayed,
    "default_transcript_unchanged": check_default_transcript_unchanged,
    "parse_tool_call_unchanged": check_parse_tool_call_unchanged,
    "adherence_metric_in_output": check_adherence_metric_in_output,
    "react_tool_catalog": check_react_tool_catalog,
    "react_native_contract_suppressed": check_react_native_contract_suppressed,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write JSON report to this path")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the full transcripts the model saw")
    args = ap.parse_args()

    results = []
    for name, fn in CHECKS.items():
        try:
            passed, detail = fn(verbose=args.verbose)
        except Exception as e:  # noqa: BLE001
            import traceback
            passed, detail = False, f"EXCEPTION: {type(e).__name__}: {e}\n" + \
                traceback.format_exc()
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
