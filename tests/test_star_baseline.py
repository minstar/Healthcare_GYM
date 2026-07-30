"""Tests for the STaR rejection-sampling SFT baseline (scripts/rebuttal/star_*.py).

No GPU, no Slurm, no served model.  A scripted policy is wired into a *real*
``AgentRunner`` (only ``generate`` is replaced) and driven against *real* tasks
from the decontaminated pool through the *real* ``BioAgentGymEnv``, so the real
tool-call parser, the real environment tools and the real scorers all execute.

What is asserted:

* the data-domain -> gym-domain map is identical to the project's single source
  of truth (``scripts/verl/bioagents_tool.py``), read without importing verl;
* the ground truth STaR scores against is byte-identical to the ground truth in
  the parquet the RL arms actually train on;
* the composite reward really can rank a WRONG rollout above a CORRECT one —
  the empirical justification for accepting on accuracy rather than the
  composite;
* the acceptance filter's threshold behaviour, including NaN/ungraded rollouts;
* accepted trajectories load through ``sft_trainer.build_sft_dataset`` in BOTH
  shapes it supports, and the ``sft_path`` shape (unlike ``trajectory_dir``)
  preserves the task ticket;
* a degenerate acceptance rate is surfaced and exits non-zero rather than
  training silently;
* re-filtering under a different signal consumes no new rollouts;
* the outer loop's bookkeeping: per-iteration state, the curve, resume,
  ``--init-from base``, ``--dataset-scope``, and the SFT config it emits.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "rebuttal" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sg = _load("star_generate")
si = _load("star_iterate")

POOL = "full_4modality_clean"
POOL_DIR = REPO / "data" / "domains" / POOL
needs_pool = pytest.mark.skipif(
    not (POOL_DIR / "tasks.json").exists(), reason="decontaminated pool not present")


# ══════════════════════════════════════════════════════════════════════
#  Fixtures: real MC tasks + a scripted policy with a controllable hit rate
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def mc_tasks():
    """Four real multiple-choice tasks from the pool the RL arms train on."""
    tasks = sg.load_pool(POOL, "train", root=REPO)
    mc = [t for t in tasks if t.get("options") and t.get("correct_answer") in "ABCDE"]
    assert len(mc) >= 4, "pool has too few multiple-choice tasks to test with"
    return mc[:4]


def scripted_policy(answer_for: dict, calls: list | None = None):
    """A policy that searches once, then submits ``answer_for[task_key]``.

    Keyed on a unique prefix of the task's question text, which is the only
    identifying thing the policy can see in its messages — exactly the
    information a real policy has.
    """
    def gen(messages, tools=None):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if calls is not None:
            calls.append(len(messages))
        if "Tool result" not in text:
            return json.dumps({"name": "search_pubmed",
                               "arguments": {"query": "differential diagnosis"}})
        ans = "Z"
        for key, val in answer_for.items():
            if key in text:
                ans = val
                break
        return json.dumps({
            "name": "submit_answer",
            "arguments": {"answer": ans,
                          "reasoning": f"Evidence supports this option. Answer: {ans}"},
        })
    return gen


def key_of(task: dict) -> str:
    return (task.get("raw_question") or task.get("ticket") or "")[:60]


def gen_cfg(out_dir, **kw) -> "sg.GenConfig":
    base = dict(model="mock-policy", backend="mock", pool=POOL, split="train",
                samples_per_task=1, max_turns=4, workers=1, temperature=1.0,
                out_dir=str(out_dir))
    base.update(kw)
    return sg.GenConfig(**base)


# ══════════════════════════════════════════════════════════════════════
#  1. The map and the ground truth cannot drift from the RL arms
# ══════════════════════════════════════════════════════════════════════


def test_domain_map_is_identical_to_the_projects_source_of_truth():
    """DATA_TO_BIO_DOMAIN is copied from bioagents_tool.py; prove the copy is exact.

    bioagents_tool.py imports verl at module scope and cannot be imported here,
    so the literal is read with ast instead.
    """
    truth = sg.read_data_to_bio_domain_literal(
        REPO / "scripts" / "verl" / "bioagents_tool.py")
    assert sg.DATA_TO_BIO_DOMAIN == truth


@needs_pool
def test_verl_ground_truth_matches_the_parquet_the_rl_arms_train_on():
    """STaR's verl_acc must be scored against the RL arms' exact ground truth."""
    pq = Path("/data/project/private/minstar/workspace/hcgym_rebuttal/data/verl_parquet") \
        / POOL / "train.parquet"
    if not pq.exists():
        pytest.skip(f"{pq} not present")
    import pandas as pd

    df = pd.read_parquet(pq)
    tasks = {t["id"]: t for t in json.loads((POOL_DIR / "tasks.json").read_text())}
    checked = 0
    for _, row in df.iterrows():
        extra = row["extra_info"]
        tid = extra["task_id"]
        task = tasks.get(tid)
        if task is None:
            continue
        gt, has_options = sg.verl_ground_truth(task)
        assert gt == row["reward_model"]["ground_truth"], tid
        assert bool(has_options) == bool(extra["has_options"]), tid
        checked += 1
    assert checked == len(df), f"only {checked}/{len(df)} rows cross-checked"
    print(f"\n  ground truth verified on all {checked} rows of {pq.name}")


# ══════════════════════════════════════════════════════════════════════
#  2. Why the default acceptance signal is accuracy, not the composite
# ══════════════════════════════════════════════════════════════════════


TERSE_TEMPLATE = "Answer: {gold}"
FLUENT_WRONG_TEMPLATE = (
    "<think>Let me reason through this step by step. The clinical picture "
    "raises several possibilities. First I considered the anatomy, then I "
    "weighed each option against the reported findings, because the history "
    "and the examination must both be explained. Therefore the evidence "
    "indicates the diagnosis below.</think>\n"
    "Based on the evidence gathered from the literature search and a careful "
    "review of the differential diagnosis, the findings are most consistent "
    "with the option selected here, and the evidence search performed supports "
    "that reasoning.\nAnswer: {wrong}")


@needs_pool
def test_composite_reward_ranks_a_wrong_fluent_rollout_above_a_correct_terse_one():
    """The empirical basis for NOT accepting on the composite.

    Accuracy is 0.25 of the composite; format, process, safety and coherence
    together are 0.60.  Measured on REAL tasks from the training pool, with
    each task's own expected actions and nl_assertions, a WRONG answer written
    fluently outscores a CORRECT answer written tersely on a substantial
    fraction of tasks.  A STaR baseline accepting on the composite would
    therefore select for presentation — adaptation to the reward's surface form,
    the confound this arm exists to rule out — which is why the default
    acceptance signal is accuracy alone.
    """
    from bioagents.evaluation.rewards import compute_composite_reward

    tasks = [t for t in sg.load_pool(POOL, "train", root=REPO)
             if t.get("correct_answer") in "ABCDE"][:200]
    assert tasks, "no multiple-choice tasks in the pool"

    tool_log = [
        {"tool_name": "search_pubmed", "arguments": {"query": "differential"},
         "response": "Retrieved evidence."},
        {"tool_name": "submit_answer", "arguments": {"answer": "D"}, "response": "ok"},
    ]
    inversions, margins = 0, []
    for t in tasks:
        ec = t.get("evaluation_criteria", {}) or {}
        gold = t["correct_answer"]
        wrong = next(c for c in "ABCDE" if c != gold)
        kw = dict(correct_answer=gold, tool_call_log=tool_log,
                  expected_actions=ec.get("actions", []),
                  nl_assertions=ec.get("nl_assertions", []), is_final=True)
        c = compute_composite_reward(response=TERSE_TEMPLATE.format(gold=gold), **kw)
        w = compute_composite_reward(
            response=FLUENT_WRONG_TEMPLATE.format(wrong=wrong), **kw)
        assert c["accuracy"] == 1.0 and w["accuracy"] == 0.0
        margins.append(w["total"] - c["total"])
        inversions += int(w["total"] > c["total"])

    frac = inversions / len(tasks)
    print(f"\n  composite inversion on {inversions}/{len(tasks)} real tasks "
          f"({frac:.1%}); mean(wrong-correct) = {sum(margins)/len(margins):+.3f}")
    assert inversions > 0, (
        "the composite never ranked a wrong-fluent rollout above a "
        "correct-terse one — re-examine the acceptance-signal argument in "
        "baselines/STAR_PORT.md before relying on it")


def test_accuracy_only_signals_are_declared():
    assert sg.ACCURACY_ONLY_SIGNALS == {"qa_accuracy", "accuracy", "verl_acc"}
    assert sg.GenConfig().accept_on in sg.ACCURACY_ONLY_SIGNALS


# ══════════════════════════════════════════════════════════════════════
#  3. The filter
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("value,threshold,expected", [
    (1.0, 1.0, True),
    (0.999, 1.0, False),
    (0.5, 0.5, True),
    (0.0, 0.0, True),
    (-999.0, 0.5, False),
    (float("nan"), 0.0, False),      # ungraded task: no correctness signal
    (None, 0.0, False),              # signal absent entirely
])
def test_acceptance_threshold_behaviour(value, threshold, expected):
    scores = {} if value is None else {"qa_accuracy": value}
    assert sg.is_accepted(scores, "qa_accuracy", threshold) is expected


def test_default_thresholds_cover_every_signal():
    assert set(sg.DEFAULT_THRESHOLDS) == set(sg.SIGNAL_NAMES)
    for sig in sg.SIGNAL_NAMES:
        assert sg.GenConfig(accept_on=sig).resolved_threshold() == sg.DEFAULT_THRESHOLDS[sig]
    assert sg.GenConfig(accept_on="accuracy", accept_threshold=0.77).resolved_threshold() == 0.77


def test_max_accepted_per_task_caps_easy_tasks():
    """k samples of one easy task must not outweigh a hard task in the SFT set."""
    def rec(tid, k, score):
        return {"task_id": tid, "star": {"sample_idx": k, "accepted": True,
                                         "scores": {"qa_accuracy": score,
                                                    "composite_reward": score}}}
    records = [rec("easy", 0, 1.0), rec("easy", 1, 1.0), rec("easy", 2, 1.0),
               rec("hard", 0, 1.0)]
    cfg = sg.GenConfig(out_dir="/tmp", max_accepted_per_task=1)
    kept = sg.select_accepted(records, cfg)
    assert sorted(r["task_id"] for r in kept) == ["easy", "hard"]

    cfg_all = sg.GenConfig(out_dir="/tmp", max_accepted_per_task=0)
    assert len(sg.select_accepted(records, cfg_all)) == 4


# ══════════════════════════════════════════════════════════════════════
#  4. End-to-end generation with a scripted policy
# ══════════════════════════════════════════════════════════════════════


@needs_pool
def test_generation_end_to_end_separates_correct_from_incorrect(tmp_path, mc_tasks):
    """Two tasks answered correctly, two answered wrong -> 50% acceptance."""
    answers = {}
    for i, t in enumerate(mc_tasks):
        gold = t["correct_answer"]
        wrong = next(c for c in "ABCDE" if c != gold)
        answers[key_of(t)] = gold if i < 2 else wrong

    cfg = gen_cfg(tmp_path / "gen")
    stats = sg.run_generation(cfg, tasks=mc_tasks,
                              generate_fn=scripted_policy(answers))

    assert stats["n_tasks"] == 4
    assert stats["n_rollouts"] == 4
    assert stats["n_accepted"] == 2, stats["acceptance_if_accepted_on"]
    assert stats["acceptance_rate"] == 0.5
    assert stats["task_coverage"] == 0.5
    assert stats["threshold_verdict"] == "ok"
    assert stats["accept_signal_is_accuracy_only"] is True
    assert stats["n_errors"] == 0

    # The environment's own accuracy dimension agrees with its submit grader.
    dist = stats["score_distribution"]
    assert dist["qa_accuracy"]["mean"] == 0.5
    assert dist["accuracy"]["mean"] == 0.5

    traj_dir = Path(stats["paths"]["trajectory_dir"])
    assert len(list(traj_dir.glob("*.json"))) == 2
    sft = Path(stats["paths"]["sft_path"])
    assert len([ln for ln in sft.read_text().splitlines() if ln.strip()]) == 2
    assert len(list(Path(stats["paths"]["rollouts"]).glob("*.json"))) == 4

    # The recorded reward fields are the environment's genuine values, not the
    # acceptance decision written back.
    for p in traj_dir.glob("*.json"):
        rec = json.loads(p.read_text())
        assert rec["star"]["accepted"] is True
        assert rec["star"]["scores"]["qa_accuracy"] == 1.0
        assert rec["final_reward"] == pytest.approx(
            rec["star"]["scores"]["composite_reward"])
        assert rec["final_reward"] < 1.0     # composite never reaches 1.0 in practice


@needs_pool
def test_verl_acc_agrees_with_the_env_grader_on_multiple_choice(tmp_path, mc_tasks):
    """reward_fn (the RL arms' scorer) and the GYM grader must not disagree on MC."""
    answers = {key_of(t): t["correct_answer"] for t in mc_tasks}
    stats = sg.run_generation(gen_cfg(tmp_path / "gen"), tasks=mc_tasks,
                              generate_fn=scripted_policy(answers))
    rows = [json.loads(ln) for ln in
            (tmp_path / "gen" / "all_scores.jsonl").read_text().splitlines() if ln.strip()]
    assert rows
    for r in rows:
        assert r["scores"]["qa_accuracy"] == 1.0
        assert r["scores"]["verl_acc"] == 1.0, r
    assert stats["acceptance_if_accepted_on"]["verl_acc"]["acceptance_rate"] == 1.0


@needs_pool
def test_worker_threads_do_not_change_the_result(tmp_path, mc_tasks):
    """--workers > 1 shares one AgentRunner; prove no cross-rollout leakage.

    The runner's ``config.domain`` decides the system prompt, so the pass runs
    one gym domain at a time.  A scripted policy is deterministic, so the
    parallel and sequential accepted sets must be identical.
    """
    answers = {}
    for i, t in enumerate(mc_tasks):
        gold = t["correct_answer"]
        answers[key_of(t)] = gold if i < 2 else next(c for c in "ABCDE" if c != gold)

    seq = sg.run_generation(gen_cfg(tmp_path / "seq", workers=1, samples_per_task=2),
                            tasks=mc_tasks, generate_fn=scripted_policy(answers))
    par = sg.run_generation(gen_cfg(tmp_path / "par", workers=4, samples_per_task=2),
                            tasks=mc_tasks, generate_fn=scripted_policy(answers))

    def accepted_ids(d):
        return sorted(p.name for p in (Path(d) / "trajectory_dir").glob("*.json"))
    assert accepted_ids(tmp_path / "seq") == accepted_ids(tmp_path / "par")
    assert seq["n_accepted"] == par["n_accepted"] == 4
    assert seq["score_distribution"]["qa_accuracy"]["mean"] == \
        par["score_distribution"]["qa_accuracy"]["mean"]


@needs_pool
def test_cli_smoke_over_the_real_pool_with_the_builtin_mock_backend(tmp_path, capsys):
    """The whole script, through argparse, over the real mixed pool, no GPU."""
    rc = sg.main(["--backend", "mock", "--limit", "24", "--samples-per-task", "1",
                  "--workers", "4", "--max-turns", "3", "--allow-uninformative",
                  "--out-dir", str(tmp_path / "smoke")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "STaR generation" in out
    assert "rollout budget" in out
    stats = json.loads((tmp_path / "smoke" / "gen_stats.json").read_text())
    assert stats["n_rollouts"] == 24
    # Every rollout that the env DID score must carry all six signals.
    rows = [json.loads(ln) for ln in
            (tmp_path / "smoke" / "all_scores.jsonl").read_text().splitlines() if ln.strip()]
    for r in rows:
        assert set(sg.SIGNAL_NAMES) <= set(r["scores"])
    # The mixed pool spans several gym domains, which is the part of the loop
    # that a single-domain test cannot exercise.
    assert len({r["source_domain"] for r in rows}) >= 3


@needs_pool
def test_env_grader_failures_are_counted_not_hidden(tmp_path):
    """agent_runner._compute_qa_accuracy raises on WRONG answers to null-option tasks.

    142/3390 train tasks (all multimodal_vqa with a <=2-char gold answer) carry
    ``"options": null``.  ``_compute_qa_accuracy`` returns early when the
    submitted answer matches, and only reaches ``options.get(...)`` when it does
    NOT — so the AttributeError fires exactly on incorrect rollouts.  run_task
    swallows it, so the trajectory survives with no ``reward_details`` at all.

    FIXED 2026-07-28: ``options = task.get("options") or {}``.  This test was
    written to pin the buggy behaviour so the loss could be counted; it now pins
    the repair instead — those rollouts are scored by the environment grader like
    any other, and nothing is silently dropped from the composite means.

    Kept rather than deleted because the shape it exercises (``options`` present
    but null, answered incorrectly) is the one that used to fail, and a
    regression here would silently shrink the denominator again.
    """
    pool = sg.load_pool(POOL, "train", root=REPO)
    victims = [t for t in pool
               if not t.get("options") and 0 < len(str(t.get("correct_answer", ""))) <= 2][:3]
    assert victims, "pool no longer contains the affected task shape"

    # Answer every one of them WRONG — the branch that reaches options.get().
    answers = {key_of(t): "zz" for t in victims}
    stats = sg.run_generation(gen_cfg(tmp_path / "gen"), tasks=victims,
                              generate_fn=scripted_policy(answers))
    assert stats["n_errors"] == 0, "null options must no longer crash the grader"
    assert stats["n_unscored_by_env_grader"] == 0, "these rollouts are now scored"
    assert stats["n_rollouts"] == len(victims)
    assert stats["acceptance_if_accepted_on"]["qa_accuracy"]["n_accepted"] == 0
    # verl_acc still has an opinion about them, and it is "wrong".
    rows = [json.loads(ln) for ln in
            (tmp_path / "gen" / "all_scores.jsonl").read_text().splitlines() if ln.strip()]
    assert all(r["scores"]["verl_acc"] == 0.0 for r in rows)

    # Answering the SAME tasks correctly returns early and does not crash,
    # which is what bounds the damage to incorrect rollouts.
    ok = sg.run_generation(gen_cfg(tmp_path / "gen_ok"), tasks=victims,
                           generate_fn=scripted_policy(
                               {key_of(t): str(t["correct_answer"]) for t in victims}))
    assert ok["n_errors"] == 0
    assert ok["n_accepted"] == len(victims)


def test_render_verl_solution_emits_markup_reward_fn_can_read():
    turns = [
        {"raw_output": '{"name": "search_pubmed", "arguments": {"query": "x"}}',
         "parsed_tool_call": {"name": "search_pubmed", "arguments": {"query": "x"}},
         "tool_response": "some evidence"},
        {"raw_output": '{"name": "submit_answer", "arguments": {"answer": "D"}}',
         "parsed_tool_call": {"name": "submit_answer",
                              "arguments": {"answer": "D", "reasoning": "Answer: D"}},
         "tool_response": "ok"},
    ]
    s = sg.render_verl_solution(turns)
    rf = sg._reward_fn()
    assert rf.tool_call_names(s) == ["search_pubmed", "submit_answer"]
    assert rf.count_invalid_tool_calls(s) == 0
    assert rf.extract_answer_letter(s) == "D"
    payload = rf.compute_score("bioagents_medical", s, "D", {"has_options": True})
    assert payload["acc"] == 1.0

    # A turn that already carries native qwen3_coder markup is not double-counted.
    native = [{"raw_output": '<function=think>\n{"thought": "hm"}\n</function>',
               "parsed_tool_call": {"name": "think", "arguments": {"thought": "hm"}},
               "tool_response": None}]
    assert rf.tool_call_names(sg.render_verl_solution(native)) == ["think"]


# ══════════════════════════════════════════════════════════════════════
#  5. The accepted data actually loads through the existing trainer
# ══════════════════════════════════════════════════════════════════════


@needs_pool
def test_accepted_trajectories_load_through_build_sft_dataset(tmp_path, mc_tasks):
    """Both shapes sft_trainer supports must load without modification."""
    from bioagents.training.sft_trainer import BioAgentSFTConfig, build_sft_dataset

    answers = {key_of(t): t["correct_answer"] for t in mc_tasks}
    stats = sg.run_generation(gen_cfg(tmp_path / "gen"), tasks=mc_tasks,
                              generate_fn=scripted_policy(answers))

    # (a) trajectory_dir + min_reward, the path the trainer already had.
    cfg_traj = BioAgentSFTConfig(
        trajectory_dir=stats["paths"]["trajectory_dir"],
        qa_tasks_path="", instruction_path="", sft_path="",
        min_reward=0.0, train_ratio=0.75)
    train_ds, _ = build_sft_dataset(cfg_traj)
    assert len(train_ds) >= 1
    assert isinstance(train_ds[0]["messages"], list)

    # (b) sft_path, the pre-generated-jsonl path.
    cfg_sft = BioAgentSFTConfig(
        sft_path=stats["paths"]["sft_path"], trajectory_dir="",
        qa_tasks_path="", instruction_path="", train_ratio=0.75)
    train_ds2, _ = build_sft_dataset(cfg_sft)
    assert len(train_ds2) >= 1
    roles = [m["role"] for m in train_ds2[0]["messages"]]
    assert roles[0] == "system"


@needs_pool
def test_sft_path_shape_keeps_the_task_ticket_and_trajectory_dir_does_not(tmp_path, mc_tasks):
    """Why star_iterate defaults to --sft-source sft_path.

    ``sft_generator.trajectory_to_sft`` rebuilds messages from turns alone, and
    turns hold no user question, so a trajectory_dir example teaches the model
    to emit tool calls with no task in context.  The sft_path shape carries the
    conversation the policy actually saw.
    """
    from bioagents.data_pipeline.sft_generator import trajectory_to_sft

    answers = {key_of(t): t["correct_answer"] for t in mc_tasks}
    stats = sg.run_generation(gen_cfg(tmp_path / "gen"), tasks=mc_tasks,
                              generate_fn=scripted_policy(answers))
    task0 = mc_tasks[0]
    needle = (task0.get("raw_question") or task0["ticket"])[:40]

    sft_rows = [json.loads(ln) for ln in
                Path(stats["paths"]["sft_path"]).read_text().splitlines() if ln.strip()]
    row = next(r for r in sft_rows if r["metadata"]["task_id"] == task0["id"])
    joined = "\n".join(m["content"] for m in row["messages"])
    assert needle in joined
    assert any(m["role"] == "user" for m in row["messages"])

    tpath = next(p for p in Path(stats["paths"]["trajectory_dir"]).glob("*.json")
                 if json.loads(p.read_text())["task_id"] == task0["id"])
    legacy = trajectory_to_sft(str(tpath), min_reward=0.0)
    legacy_joined = "\n".join(m["content"] for m in legacy[0]["messages"])
    assert needle not in legacy_joined, (
        "trajectory_to_sft unexpectedly carries the ticket now — "
        "re-check the sft_source default in star_iterate.py")


# ══════════════════════════════════════════════════════════════════════
#  6. A degenerate acceptance rate must be loud, not silent
# ══════════════════════════════════════════════════════════════════════


@needs_pool
@pytest.mark.parametrize("mode,verdict", [("all_wrong", "uninformative_zero"),
                                          ("all_right", "uninformative_high")])
def test_degenerate_acceptance_rate_is_flagged(tmp_path, mc_tasks, mode, verdict, capsys):
    if mode == "all_right":
        answers = {key_of(t): t["correct_answer"] for t in mc_tasks}
    else:
        answers = {key_of(t): next(c for c in "ABCDE" if c != t["correct_answer"])
                   for t in mc_tasks}
    stats = sg.run_generation(gen_cfg(tmp_path / "gen"), tasks=mc_tasks,
                              generate_fn=scripted_policy(answers))
    assert stats["threshold_verdict"] == verdict
    assert "UNINFORMATIVE THRESHOLD" in capsys.readouterr().out


@needs_pool
def test_uninformative_pass_exits_nonzero(tmp_path, mc_tasks, monkeypatch):
    """A pipeline must stop, not scroll past the warning."""
    answers = {key_of(t): next(c for c in "ABCDE" if c != t["correct_answer"])
               for t in mc_tasks}
    monkeypatch.setattr(sg, "load_pool", lambda *a, **k: mc_tasks)
    monkeypatch.setattr(sg, "_make_runner", _runner_with(sg, answers))
    rc = sg.main(["--backend", "mock", "--out-dir", str(tmp_path / "gen"),
                  "--samples-per-task", "1", "--workers", "1", "--max-turns", "4"])
    assert rc == 3
    rc_ok = sg.main(["--backend", "mock", "--out-dir", str(tmp_path / "gen2"),
                     "--samples-per-task", "1", "--workers", "1", "--max-turns", "4",
                     "--allow-uninformative"])
    assert rc_ok == 0


def _runner_with(module, answers):
    """A _make_runner replacement that installs the scripted policy."""
    original = module._make_runner

    def patched(cfg, log_dir):
        runner = original(cfg, log_dir)
        runner.generate = scripted_policy(answers)
        return runner
    return patched


# ══════════════════════════════════════════════════════════════════════
#  7. Re-filtering costs no rollouts
# ══════════════════════════════════════════════════════════════════════


@needs_pool
def test_refilter_under_a_different_signal_uses_no_new_rollouts(tmp_path, mc_tasks):
    calls: list[int] = []
    answers = {}
    for i, t in enumerate(mc_tasks):
        gold = t["correct_answer"]
        answers[key_of(t)] = gold if i < 2 else next(c for c in "ABCDE" if c != gold)

    sg.run_generation(gen_cfg(tmp_path / "gen"), tasks=mc_tasks,
                      generate_fn=scripted_policy(answers, calls))
    n_calls_after_generation = len(calls)
    assert n_calls_after_generation > 0

    cfg2 = gen_cfg(tmp_path / "refiltered", accept_on="composite_reward",
                   accept_threshold=0.0)
    stats2 = sg.refilter(cfg2, tmp_path / "gen")

    assert len(calls) == n_calls_after_generation, "re-filter rolled out again"
    assert stats2["n_rollouts"] == 4
    assert stats2["n_accepted"] == 4          # threshold 0.0 accepts everything
    assert stats2["accept_signal_is_accuracy_only"] is False
    assert stats2["refiltered_from"].endswith("gen")


# ══════════════════════════════════════════════════════════════════════
#  8. Budget accounting
# ══════════════════════════════════════════════════════════════════════


def test_star_budget_matches_the_rl_arms_by_default(monkeypatch):
    monkeypatch.setenv("ROLLOUT_N", "3")
    monkeypatch.setenv("TOTAL_EPOCHS", "3")
    b = sg.budget_report(n_tasks=3390, samples_per_task=3, iterations=3)
    assert b["star_rollouts_total"] == 30510
    assert b["rl_reference"]["rollouts_total"] == 30510
    assert b["matched"] is True
    assert b["ratio_star_over_rl"] == 1.0

    b2 = sg.budget_report(n_tasks=3390, samples_per_task=6, iterations=3)
    assert b2["matched"] is False
    assert b2["ratio_star_over_rl"] == 2.0


# ══════════════════════════════════════════════════════════════════════
#  9. Outer-loop bookkeeping
# ══════════════════════════════════════════════════════════════════════


class FakeHooks(si.Hooks):
    """Records what the loop would do; fabricates plausible stats."""

    def __init__(self, acceptance=(0.30, 0.45, 0.60), selected=(120, 180, 240)):
        super().__init__(dry_run=False)
        self.acceptance, self.selected = acceptance, selected
        self.served: list[str] = []
        self.sft_inits: list[str] = []
        self.gen_dirs: list[Path] = []
        self._i = 0

    def serve(self, cfg, model, log_path):
        self.served.append(model)
        return None, "http://127.0.0.1:0"

    def stop(self, srv):
        pass

    def generate(self, cmd, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_only = "--eval-only" in cmd
        if eval_only:
            return {"n_tasks": 10, "score_distribution": {
                "qa_accuracy": {"mean": 0.10 + 0.05 * self._i},
                "verl_acc": {"mean": 0.11 + 0.05 * self._i},
                "composite_reward": {"mean": 0.40}}}
        i = min(self._i, len(self.acceptance) - 1)
        self.gen_dirs.append(out_dir)
        n_sel = self.selected[i]
        (out_dir / "star_sft.jsonl").write_text(
            "".join(json.dumps({"messages": [{"role": "user", "content": f"q{j}"}]}) + "\n"
                    for j in range(n_sel)), encoding="utf-8")
        self._i += 1
        return {"acceptance_rate": self.acceptance[i], "task_coverage": self.acceptance[i],
                "n_rollouts": 900, "n_accepted": int(900 * self.acceptance[i]),
                "n_selected": n_sel, "threshold_verdict": "ok",
                "acceptance_if_accepted_on": {}}

    def sft(self, cmd, output_dir):
        cfg_path = Path(cmd[cmd.index("--config") + 1])
        import yaml
        self.sft_inits.append(yaml.safe_load(cfg_path.read_text())["model"]["name_or_path"])
        final = output_dir / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "config.json").write_text("{}", encoding="utf-8")
        return str(final)


def loop_cfg(tmp_path, **kw):
    base = dict(base_model="/models/BASE", out_root=str(tmp_path / "star"),
                iterations=3, heldout_size=10)
    base.update(kw)
    return si.LoopConfig(**base)


def test_loop_runs_n_iterations_and_persists_the_curve(tmp_path):
    hooks = FakeHooks()
    out = si.run_loop(loop_cfg(tmp_path), hooks)

    curve = json.loads(Path(out["curve_path"]).read_text())["iterations"]
    trained = [r for r in curve if not r.get("final")]
    assert [r["iteration"] for r in trained] == [0, 1, 2]
    assert [r["acceptance_rate"] for r in trained] == [0.30, 0.45, 0.60]
    assert [r["n_train_examples"] for r in trained] == [120, 180, 240]
    # A held-out score at every iteration, plus one after the last SFT.
    assert all(r["heldout"]["qa_accuracy"] is not None for r in trained)
    assert curve[-1]["final"] is True
    for it in range(3):
        st = json.loads((tmp_path / "star" / f"iter_{it:02d}" / "state.json").read_text())
        assert st["done"] is True


def test_init_from_base_refits_the_original_backbone_every_iteration(tmp_path):
    """Zelikman et al. train from the ORIGINAL model each outer loop."""
    hooks = FakeHooks()
    si.run_loop(loop_cfg(tmp_path, init_from="base"), hooks)
    assert hooks.sft_inits == ["/models/BASE"] * 3
    # ...while the policy that is ROLLED OUT is the previous iteration's model.
    assert hooks.served[0] == "/models/BASE"
    assert hooks.served[1].endswith("iter_00/sft/final")
    assert hooks.served[2].endswith("iter_01/sft/final")

    hooks2 = FakeHooks()
    si.run_loop(loop_cfg(tmp_path / "b", init_from="previous"), hooks2)
    assert hooks2.sft_inits[0] == "/models/BASE"
    assert hooks2.sft_inits[1].endswith("iter_00/sft/final")


def test_dataset_scope_current_vs_accumulated(tmp_path):
    hooks = FakeHooks()
    si.run_loop(loop_cfg(tmp_path, dataset_scope="current"), hooks)
    cur = [r["n_train_examples"] for r in
           json.loads((tmp_path / "star" / "star_curve.json").read_text())["iterations"]
           if not r.get("final")]
    assert cur == [120, 180, 240]

    hooks2 = FakeHooks()
    si.run_loop(loop_cfg(tmp_path / "acc", dataset_scope="accumulated"), hooks2)
    acc = [r["n_train_examples"] for r in
           json.loads((tmp_path / "acc" / "star" / "star_curve.json").read_text())["iterations"]
           if not r.get("final")]
    assert acc == [120, 300, 540]


def test_resume_skips_completed_iterations(tmp_path):
    si.run_loop(loop_cfg(tmp_path), FakeHooks())
    hooks2 = FakeHooks()
    si.run_loop(loop_cfg(tmp_path), hooks2)
    # Only the final held-out probe should have needed a server.
    assert hooks2.sft_inits == []
    assert len(hooks2.gen_dirs) == 0


def test_loop_refuses_to_train_on_an_empty_accepted_set(tmp_path):
    hooks = FakeHooks(acceptance=(0.0,), selected=(0,))
    with pytest.raises(RuntimeError, match="degenerate|no accepted"):
        si.run_loop(loop_cfg(tmp_path, iterations=1), hooks)


def test_loop_stops_on_a_degenerate_acceptance_rate_at_iteration_zero(tmp_path):
    class H(FakeHooks):
        def generate(self, cmd, out_dir):
            st = super().generate(cmd, out_dir)
            if "--eval-only" not in cmd:
                st["threshold_verdict"] = "uninformative_high"
            return st
    with pytest.raises(RuntimeError, match="degenerate"):
        si.run_loop(loop_cfg(tmp_path, iterations=1), H())


# ══════════════════════════════════════════════════════════════════════
#  10. The SFT config the loop emits
# ══════════════════════════════════════════════════════════════════════


def test_sft_config_never_mixes_in_oracle_qa_demonstrations(tmp_path):
    """The trainer defaults qa_tasks_path to a gold-answer file; it must be empty."""
    cfg = loop_cfg(tmp_path)
    d = si.sft_config_dict(cfg, "/models/BASE", tmp_path / "d.jsonl", tmp_path / "out", "r")
    assert d["dataset"]["qa_tasks_path"] == ""
    assert d["dataset"]["instruction_path"] == ""
    assert d["dataset"]["min_reward"] == 0.0
    assert d["peft"]["enabled"] is False          # full-parameter, like the RL arms
    assert d["dataset"]["max_samples"] >= 200000  # no silent subsampling
    assert d["model"]["attn_implementation"] == "sdpa"

    from bioagents.training.sft_trainer import BioAgentSFTConfig
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    loaded = BioAgentSFTConfig.from_yaml(str(p))
    assert loaded.qa_tasks_path == ""
    assert loaded.peft_enabled is False
    assert loaded.sft_path.endswith("d.jsonl")
    assert loaded.trajectory_dir == ""


def test_sft_source_trajectory_dir_switches_the_loader(tmp_path):
    cfg = loop_cfg(tmp_path, sft_source="trajectory_dir")
    d = si.sft_config_dict(cfg, "/models/BASE", tmp_path / "tdir", tmp_path / "out", "r")
    assert d["dataset"]["trajectory_dir"].endswith("tdir")
    assert d["dataset"]["sft_path"] == ""


def test_sft_command_uses_the_same_student_gpu_count_as_the_rl_arms(tmp_path):
    cmd = si.sft_command(loop_cfg(tmp_path), tmp_path / "c.yaml")
    assert cmd[:2] == ["accelerate", "launch"]
    assert cmd[cmd.index("--num_processes") + 1] == "7"


def test_serve_command_carries_the_blackwell_attention_backends(tmp_path):
    srv = si.SglangServer("/models/BASE", 31500, 8, tmp_path / "s.log")
    cmd = srv.command()
    assert cmd[cmd.index("--attention-backend") + 1] == "triton"
    # The standalone-server counterpart of the RL arms'
    # engine_kwargs.sglang.mm_attention_backend=triton_attn (verl's "fa3"
    # default is fatal on Blackwell for VLM backbones).
    assert cmd[cmd.index("--mm-attention-backend") + 1] == "triton_attn"
    assert cmd[cmd.index("--dp-size") + 1] == "8"


def test_generate_command_holds_the_turn_budget_and_pool(tmp_path):
    cfg = loop_cfg(tmp_path)
    cmd = si.generate_command(cfg, "/m", "http://x", tmp_path, split="train", samples=3,
                              temperature=1.0, eval_only=False, limit=0)
    assert cmd[cmd.index("--max-turns") + 1] == "5"
    assert cmd[cmd.index("--pool") + 1] == "full_4modality_clean"
    assert cmd[cmd.index("--accept-on") + 1] == "qa_accuracy"
    assert "--eval-only" not in cmd

    hcmd = si.generate_command(cfg, "/m", "http://x", tmp_path, split="test", samples=1,
                               temperature=0.1, eval_only=True, limit=200)
    assert "--eval-only" in hcmd and "--allow-uninformative" in hcmd


def test_dry_run_prints_a_plan_and_runs_nothing(tmp_path, capsys):
    rc = si.main(["--base-model", "/models/BASE", "--out-root", str(tmp_path / "s"),
                  "--iterations", "2", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRYRUN ok" in out
    assert "DRYRUN generate:" in out
    assert "DRYRUN sft:" in out
    assert "DRYRUN serve:" in out
    assert "star_generate.py" in out and "sft_trainer.py" in out
    assert not (tmp_path / "s" / "iter_00" / "sft" / "final" / "config.json").exists()
    # A dry run must not leave a "done" marker: --resume would skip the real run.
    assert not (tmp_path / "s" / "iter_00" / "state.json").exists()
    assert not (tmp_path / "s" / "star_curve.json").exists()
    # ...but the resolved SFT config IS written, so the plan can be inspected.
    assert (tmp_path / "s" / "iter_00" / "sft_config.yaml").exists()
