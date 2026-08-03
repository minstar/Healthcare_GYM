"""Tests for the Reflexion baseline driver (bioagents.evaluation.reflexion_runner).

No GPU and no model: a scripted mock model is wired into a *real*
``AgentRunner`` (only ``generate`` is overridden) and driven against a fake
gym environment, so the real ``run_task`` loop, the real tool-call parser and
the real scorers (``_compute_qa_accuracy`` / ``_compute_action_score``) all
execute.  What is asserted:

* attempts per strategy rung match the ladder,
* the reflection text actually reaches the retry prompt (the mock only answers
  correctly *because* it sees the reflection token),
* the same-model constraint is enforced, including mid-run drift,
* extra-model-call accounting is exact,
* ungraded tasks degrade to a defined, recorded behaviour.
"""

import json

import pytest

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
from bioagents.evaluation.reflexion_runner import (
    LAST_TRIAL_HEADER,
    MEMORY_BLOCK_OPEN,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
    AttemptRecord,
    GoldAnswerLeak,
    ReflexionConfig,
    ReflexionEnvProxy,
    ReflexionRunner,
    ReflexionStrategy,
    SameModelViolation,
    SuccessSignal,
    UngradedPolicy,
    format_last_attempt,
    format_reflections,
    render_scratchpad,
    truncate_scratchpad,
)

REFLECT_TOKEN = "RXTOKEN"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_guidelines",
            "description": "Search clinical guidelines.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def make_mcqa_task(task_id="medqa_test_0001"):
    return {
        "id": task_id,
        "ticket": (
            "QUESTION: An 8-month-old boy with abdominal distension and failure to pass "
            "meconium at birth. Which diagnosis is most likely?\n"
            "Option A: Meckel diverticulum\nOption B: DiGeorge syndrome\n"
            "Option C: Duodenal atresia\nOption D: Hirschsprung disease"
        ),
        "correct_answer": "D",
        "options": {
            "A": "Meckel diverticulum",
            "B": "DiGeorge syndrome",
            "C": "Duodenal atresia",
            "D": "Hirschsprung disease",
        },
        "evaluation_criteria": {"actions": [{"name": "search_guidelines"}]},
    }


def make_ungraded_task(task_id="ungraded_0001", actions=None):
    """No gold answer.  ``actions=[]`` is the trap: _compute_action_score returns
    1.0 for an empty expectation list, which would fake a success."""
    return {
        "id": task_id,
        "ticket": "Draft a management plan for this admission.",
        "evaluation_criteria": {"actions": actions if actions is not None else []},
    }


class FakeEnv:
    """Minimal BioAgentGymEnv stand-in exposing exactly what run_task touches."""

    def __init__(self, task):
        self.task = task
        self._tool_call_log = []
        self._task_map = {task["id"]: task}
        self._tasks = [task]
        self.reset_calls = 0

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        self._tool_call_log = []
        obs = f"[TASK TICKET]\n{self.task['ticket']}"
        info = {
            "task_id": self.task["id"],
            "task_description": {},
            "domain": "medical_qa",
            "policy": "Use the tools, then submit_answer.",
            "tools": TOOLS,
            "max_turns": 5,
            "image_path": None,
        }
        return obs, info

    def step(self, action):
        call = json.loads(action)
        name = call.get("name", "")
        self._tool_call_log.append({"tool_name": name, "arguments": call.get("arguments", {})})
        if name == "submit_answer":
            return "Answer recorded.", 0.0, True, False, {}
        return f"Guideline snippet for {call.get('arguments', {}).get('query', '')}.", 0.0, False, False, {}

    def get_trajectory(self):
        return {
            "domain": "medical_qa",
            "task_id": self.task["id"],
            "total_turns": len(self._tool_call_log),
            "tool_call_log": self._tool_call_log,
            "conversation_history": [],
            "final_reward": 0.0,
        }


class ScriptedRunner(AgentRunner):
    """Real AgentRunner; only the model call is scripted.

    Two model calls per attempt: a ``search_guidelines`` call, then
    ``submit_answer``.  The submitted answer is correct ONLY if the reflection
    token is present in the prompt, so success is causally tied to the
    reflection reaching the retry.
    """

    def __init__(self, config, *, obey_reflection=True, judge_script=None, on_call=None):
        super().__init__(config)
        self.obey_reflection = obey_reflection
        self.judge_script = list(judge_script or [])
        self.on_call = on_call
        self.attempt_prompts = []   # user message of every attempt turn
        self.reflect_prompts = []
        self.judge_prompts = []
        self.n_reflections = 0

    def generate(self, messages, tools=None):
        if self.on_call is not None:
            self.on_call(self, messages)
        user_text = messages[-1]["content"]
        conversation = "\n".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )

        if user_text.rstrip().endswith("Reflection:"):
            self.reflect_prompts.append(user_text)
            self.n_reflections += 1
            return (
                f"{REFLECT_TOKEN}-{self.n_reflections}: I failed because I submitted before "
                "checking the guideline. Next time I will call search_guidelines first and "
                "then answer D."
            )

        if user_text.rstrip().endswith("Verdict:"):
            self.judge_prompts.append(user_text)
            if self.judge_script:
                return self.judge_script.pop(0)
            return "UNSOLVED - the attempt never gathered evidence."

        self.attempt_prompts.append(user_text)
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        if n_assistant == 0:
            return json.dumps(
                {"name": "search_guidelines", "arguments": {"query": "congenital megacolon"}}
            )
        # The whole conversation is inspected: the memory block arrives in the
        # first user message, but the answer is emitted one turn later.
        saw_reflection = REFLECT_TOKEN in conversation
        answer = "D" if (saw_reflection and self.obey_reflection) else "A"
        return json.dumps({"name": "submit_answer", "arguments": {"answer": answer}})


def build(tmp_path, task, strategy, **cfg_kwargs):
    runner = ScriptedRunner(
        RunConfig(
            model_name_or_path="/fake/models/policy-under-test",
            backend="transformers",
            domain="medical_qa",
            max_turns=5,
            log_dir=str(tmp_path / "runs"),
        ),
        **{k: cfg_kwargs.pop(k) for k in list(cfg_kwargs) if k in ("obey_reflection", "judge_script", "on_call")},
    )
    rx = ReflexionRunner(
        runner,
        ReflexionConfig(
            strategy=strategy,
            max_attempts=cfg_kwargs.pop("max_attempts", 3),
            output_dir=str(tmp_path / "reflexion"),
            **cfg_kwargs,
        ),
    )
    return runner, rx, FakeEnv(task)


# ══════════════════════════════════════════════════════════════════
#  1. The ladder
# ══════════════════════════════════════════════════════════════════


class TestStrategyLadder:
    def test_enum_values_match_reference_implementation(self):
        assert [s.value for s in ReflexionStrategy] == [
            "base",
            "last_trial",
            "reflexion",
            "last_trial_and_reflexion",
        ]
        assert ReflexionStrategy.from_str("last_trial_and_reflexion") is (
            ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
        )
        assert ReflexionStrategy.from_str("NONE") is ReflexionStrategy.NONE
        with pytest.raises(ValueError):
            ReflexionStrategy.from_str("self_consistency")

    def test_none_is_single_attempt(self, tmp_path):
        runner, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.NONE)
        res = rx.run_task(make_mcqa_task(), env)
        assert res.attempts_used == 1
        assert res.success is False
        assert res.stop_reason == "strategy_none_single_attempt"
        assert runner.n_reflections == 0
        assert res.model_calls_total == 2
        assert res.extra_model_calls == 0
        assert res.call_overhead_ratio == 1.0
        assert res.attempts[0].memory_injected is False

    def test_none_with_retry_is_budget_matched_resampling(self, tmp_path):
        runner, rx, env = build(
            tmp_path, make_mcqa_task(), ReflexionStrategy.NONE, retry_on_none=True
        )
        res = rx.run_task(make_mcqa_task(), env)
        assert res.attempts_used == 3
        assert runner.n_reflections == 0
        assert all(a.memory_injected is False for a in res.attempts)
        assert res.model_calls_total == 6
        assert res.extra_model_calls == 4

    def test_last_attempt_injects_trajectory_without_a_model_call(self, tmp_path):
        runner, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.LAST_ATTEMPT)
        res = rx.run_task(make_mcqa_task(), env)
        assert res.attempts_used == 3          # never solved: no reflection token exists
        assert runner.n_reflections == 0       # this rung spends ZERO extra reflection calls
        assert res.model_calls_reflection == 0
        assert res.attempts[0].memory_injected is False
        for rec in res.attempts[1:]:
            assert rec.memory_injected is True
        mem = res.attempts[1].memory_preview
        assert MEMORY_BLOCK_OPEN in mem
        assert LAST_TRIAL_HEADER.strip() in mem
        assert "search_guidelines" in runner.attempt_prompts[2]  # raw trajectory in context

    def test_reflexion_solves_on_retry_and_reflection_reaches_the_prompt(self, tmp_path):
        runner, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        res = rx.run_task(make_mcqa_task(), env)

        assert res.success is True
        assert res.attempts_used == 2
        assert res.attempts_to_success == 2
        assert runner.n_reflections == 1

        # the reflection text is verbatim in the retry's user prompt
        reflection = res.reflections[0]
        retry_prompt = runner.attempt_prompts[2]      # attempt 2, turn 1
        assert REFLECT_TOKEN in reflection
        assert reflection in retry_prompt
        assert REFLECTION_HEADER.strip() in retry_prompt
        assert MEMORY_BLOCK_OPEN in retry_prompt
        # ... and the ticket still follows the memory block
        assert "[TASK TICKET]" in retry_prompt
        assert retry_prompt.index(MEMORY_BLOCK_OPEN) < retry_prompt.index("[TASK TICKET]")

        # attempt 1 saw no memory; success is causally due to the reflection
        assert res.attempts[0].success is False
        assert res.attempts[1].success is True
        assert res.attempts[0].reflection_text == reflection

    def test_last_attempt_and_reflexion_carries_both(self, tmp_path):
        runner, rx, env = build(
            tmp_path, make_mcqa_task(), ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
        )
        res = rx.run_task(make_mcqa_task(), env)
        assert res.success is True
        assert res.attempts_used == 2
        assert runner.n_reflections == 1
        retry_prompt = runner.attempt_prompts[2]
        assert LAST_TRIAL_HEADER.strip() in retry_prompt              # last trial
        assert REFLECTION_AFTER_LAST_TRIAL_HEADER.strip() in retry_prompt  # + reflection
        assert REFLECT_TOKEN in retry_prompt
        assert "(END PREVIOUS TRIAL)" in retry_prompt

    def test_reflexion_accumulates_but_last_attempt_and_reflexion_resets(self, tmp_path):
        """Faithful to the reference: REFLEXION does `+=`, LA+R reassigns.

        Asserted on the third attempt's actual prompt (attempt 3, turn 0 ->
        attempt_prompts[4]), which carries the full memory block.
        """
        runner_a, rx_a, env_a = build(
            tmp_path / "a", make_mcqa_task(), ReflexionStrategy.REFLEXION, obey_reflection=False
        )
        res_a = rx_a.run_task(make_mcqa_task(), env_a)
        third_prompt_a = runner_a.attempt_prompts[4]
        assert res_a.attempts_used == 3
        assert len(res_a.reflections) == 2
        assert f"{REFLECT_TOKEN}-1" in third_prompt_a
        assert f"{REFLECT_TOKEN}-2" in third_prompt_a  # accumulated

        runner_b, rx_b, env_b = build(
            tmp_path / "b",
            make_mcqa_task(),
            ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
            obey_reflection=False,
        )
        res_b = rx_b.run_task(make_mcqa_task(), env_b)
        third_prompt_b = runner_b.attempt_prompts[4]
        assert res_b.attempts_used == 3
        assert f"{REFLECT_TOKEN}-2" in third_prompt_b       # newest kept
        assert f"{REFLECT_TOKEN}-1" not in third_prompt_b   # older dropped (reset branch)


# ══════════════════════════════════════════════════════════════════
#  2. Cost accounting
# ══════════════════════════════════════════════════════════════════


class TestCallAccounting:
    def test_exact_extra_call_accounting(self, tmp_path):
        _, rx, env = build(
            tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION, obey_reflection=False
        )
        res = rx.run_task(make_mcqa_task(), env)

        # 3 attempts x 2 calls + 2 reflections (none after the last attempt)
        assert [a.model_calls_attempt for a in res.attempts] == [2, 2, 2]
        assert [a.model_calls_reflection for a in res.attempts] == [1, 1, 0]
        assert res.model_calls_attempt == 6
        assert res.model_calls_reflection == 2
        assert res.model_calls_judge == 0
        assert res.model_calls_total == 8
        assert res.model_calls_total == sum(
            a.model_calls_attempt + a.model_calls_reflection + a.model_calls_judge
            for a in res.attempts
        )
        assert res.baseline_model_calls == 2          # what plain AgentRunner would spend
        assert res.extra_model_calls == 6
        assert res.call_overhead_ratio == 4.0

    def test_meter_is_removed_from_the_runner_afterwards(self, tmp_path):
        runner, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        before = runner.generate
        rx.run_task(make_mcqa_task(), env)
        assert "generate" not in runner.__dict__
        assert runner.generate.__func__ is before.__func__

    def test_reflection_decode_cap_is_restored(self, tmp_path):
        runner, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        original = runner.config.max_new_tokens
        rx.run_task(make_mcqa_task(), env)
        assert runner.config.max_new_tokens == original

    def test_summary_reports_the_cost_column(self, tmp_path):
        _, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        res = rx.run_task(make_mcqa_task(), env)
        summary = rx.summarize([res])
        assert summary["success_rate_all"] == 1.0
        assert summary["mean_attempts_to_success"] == 2.0
        assert summary["attempts_to_success_histogram"] == {"1": 0, "2": 1, "3": 0}
        assert summary["extra_model_calls"] == 3      # 2 retry calls + 1 reflection
        assert summary["call_overhead_ratio"] == 2.5
        assert summary["same_model_reflection"] is True
        assert summary["reflection_model_id"] == "/fake/models/policy-under-test"


# ══════════════════════════════════════════════════════════════════
#  3. Same-model constraint
# ══════════════════════════════════════════════════════════════════


class TestSameModelConstraint:
    def test_external_reflector_is_rejected_at_construction(self, tmp_path):
        runner, _, _ = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        stronger = ScriptedRunner(
            RunConfig(
                model_name_or_path="/fake/models/much-stronger-critic",
                backend="transformers",
                domain="medical_qa",
                log_dir=str(tmp_path / "runs2"),
            )
        )
        with pytest.raises(SameModelViolation):
            ReflexionRunner(runner, ReflexionConfig(save_records=False), reflector=stronger)

    def test_model_identity_drift_mid_run_raises(self, tmp_path):
        def swap_model_after_first_turn(runner, messages):
            runner.config.model_name_or_path = "/fake/models/much-stronger-critic"

        runner, rx, env = build(
            tmp_path,
            make_mcqa_task(),
            ReflexionStrategy.REFLEXION,
            on_call=swap_model_after_first_turn,
        )
        with pytest.raises(SameModelViolation):
            rx.run_task(make_mcqa_task(), env)

    def test_reflection_is_attributed_to_the_model_under_test(self, tmp_path):
        _, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        res = rx.run_task(make_mcqa_task(), env)
        assert res.attempts[0].reflection_model_id == "/fake/models/policy-under-test"
        assert res.model_id == "/fake/models/policy-under-test"

    def test_gold_answer_never_enters_the_reflection_prompt(self, tmp_path):
        task = make_mcqa_task()
        task.pop("correct_answer")
        task["answer"] = "Hirschsprung disease with aganglionic distal colon segment"
        _, rx, env = build(tmp_path, task, ReflexionStrategy.REFLEXION, obey_reflection=False)
        runner = rx.runner
        rx.run_task(task, env)
        for prompt in runner.reflect_prompts:
            assert task["answer"].lower() not in prompt.lower()

    def test_gold_leak_guard_fires(self, tmp_path):
        task = make_mcqa_task()
        task.pop("correct_answer")
        task["answer"] = "a very distinctive free text gold answer string"
        _, rx, _ = build(tmp_path, task, ReflexionStrategy.REFLEXION)
        with pytest.raises(GoldAnswerLeak):
            rx._assert_no_gold_leak(task, "reflect on: a very distinctive free text gold answer string")


# ══════════════════════════════════════════════════════════════════
#  4. Failure signal / ungraded degradation
# ══════════════════════════════════════════════════════════════════


class TestFailureSignal:
    def test_qa_accuracy_is_the_signal_when_a_gold_answer_exists(self, tmp_path):
        _, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        res = rx.run_task(make_mcqa_task(), env)
        assert res.attempts[0].success_signal == SuccessSignal.QA_ACCURACY.value
        assert res.attempts[0].qa_accuracy == 0.0
        assert res.attempts[1].qa_accuracy == 1.0
        assert res.graded is True

    def test_action_score_signal_when_no_gold_answer(self, tmp_path):
        task = make_ungraded_task(actions=[{"name": "search_guidelines"}])
        _, rx, env = build(tmp_path, task, ReflexionStrategy.REFLEXION)
        res = rx.run_task(task, env)
        assert res.attempts[0].success_signal == SuccessSignal.ACTION_SCORE.value
        assert res.attempts[0].action_score == 1.0
        assert res.success is True
        assert res.attempts_used == 1

    def test_empty_expected_actions_must_not_fake_a_success(self, tmp_path):
        """_compute_action_score returns 1.0 for an empty list -- the guard must
        classify this as ungraded instead of silently 'solved'."""
        task = make_ungraded_task(actions=[])
        _, rx, env = build(tmp_path, task, ReflexionStrategy.REFLEXION)
        res = rx.run_task(task, env)
        assert res.attempts[0].success_signal == SuccessSignal.UNGRADED.value
        assert res.attempts[0].success is None
        assert res.success is False

    def test_ungraded_retry_all_is_the_default(self, tmp_path):
        task = make_ungraded_task()
        _, rx, env = build(tmp_path, task, ReflexionStrategy.REFLEXION)
        res = rx.run_task(task, env)
        assert res.attempts_used == 3
        assert res.graded is False
        assert all(a.success is None for a in res.attempts)
        assert all(a.assumed_failure is True for a in res.attempts)
        assert res.degraded_to_single_attempt is False
        summary = rx.summarize([res])
        assert summary["n_ungraded"] == 1
        assert summary["success_rate_graded"] is None

    def test_ungraded_stop_after_first_is_recorded_loudly(self, tmp_path):
        task = make_ungraded_task()
        _, rx, env = build(
            tmp_path,
            task,
            ReflexionStrategy.REFLEXION,
            ungraded_policy=UngradedPolicy.STOP_AFTER_FIRST,
        )
        res = rx.run_task(task, env)
        assert res.attempts_used == 1
        assert res.degraded_to_single_attempt is True
        assert res.stop_reason == "ungraded_stop_after_first"
        assert rx.summarize([res])["n_degraded_to_single_attempt"] == 1

    def test_ungraded_self_judge_uses_the_same_model_and_is_billed(self, tmp_path):
        task = make_ungraded_task()
        _, rx, env = build(
            tmp_path,
            task,
            ReflexionStrategy.REFLEXION,
            ungraded_policy=UngradedPolicy.SELF_JUDGE,
            judge_script=["UNSOLVED - no evidence gathered.", "SOLVED - plan is complete."],
        )
        res = rx.run_task(task, env)
        assert res.attempts_used == 2
        assert res.success is True
        assert res.attempts[0].success_signal == SuccessSignal.SELF_JUDGE.value
        assert [a.model_calls_judge for a in res.attempts] == [1, 1]
        assert res.model_calls_judge == 2
        assert res.model_calls_total == 2 + 2 + 1 + 2   # attempts + reflection + judges

    def test_run_task_error_counts_as_failure(self, tmp_path):
        _, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)

        class Boom(FakeEnv):
            def get_trajectory(self):
                raise RuntimeError("env exploded")

            def step(self, action):
                raise RuntimeError("env exploded")

        res = rx.run_task(make_mcqa_task(), Boom(make_mcqa_task()))
        assert res.attempts[0].success is False
        assert res.attempts[0].success_signal == SuccessSignal.ERROR.value
        assert res.attempts_used == 3


# ══════════════════════════════════════════════════════════════════
#  5. Plumbing: proxy, persistence, helpers
# ══════════════════════════════════════════════════════════════════


class TestPlumbing:
    def test_env_proxy_is_transparent(self):
        env = FakeEnv(make_mcqa_task())
        proxy = ReflexionEnvProxy(env)
        assert proxy._task_map is env._task_map
        obs, info = proxy.reset(options={"task_id": "x"})
        assert obs.startswith("[TASK TICKET]")
        proxy.set_memory("MEMORY-HERE")
        obs2, _ = proxy.reset(options={"task_id": "x"})
        assert obs2.startswith("MEMORY-HERE")
        assert "[TASK TICKET]" in obs2
        proxy.step(json.dumps({"name": "submit_answer", "arguments": {"answer": "A"}}))
        assert env._tool_call_log[-1]["tool_name"] == "submit_answer"
        assert proxy._tool_call_log is env._tool_call_log

    def test_records_are_persisted_per_attempt(self, tmp_path):
        _, rx, env = build(
            tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION, obey_reflection=False
        )
        res = rx.run_task(make_mcqa_task(), env)
        attempts_file = tmp_path / "reflexion" / "attempts.jsonl"
        tasks_file = tmp_path / "reflexion" / "tasks.jsonl"
        rows = [json.loads(l) for l in attempts_file.read_text().splitlines()]
        assert len(rows) == 3 == res.attempts_used
        assert [r["attempt_idx"] for r in rows] == [0, 1, 2]
        assert rows[0]["model_calls_reflection"] == 1
        assert rows[1]["memory_injected"] is True
        task_rows = [json.loads(l) for l in tasks_file.read_text().splitlines()]
        assert task_rows[0]["extra_model_calls"] == 6
        assert (tmp_path / "reflexion" / "reflexion_config.json").exists()

    def test_run_all_tasks_writes_a_summary(self, tmp_path):
        _, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.REFLEXION)
        results = rx.run_all_tasks(env=env)
        assert len(results) == 1
        summary = json.loads((tmp_path / "reflexion" / "summary.json").read_text())
        assert summary["n_tasks"] == 1
        assert summary["strategy"] == "reflexion"

    def test_truncate_scratchpad_shrinks_biggest_observations_first(self):
        pad = "Action 0: call\nObservation 0: " + "x" * 5000 + "\nAction 1: call\nObservation 1: short"
        out = truncate_scratchpad(pad, max_chars=400)
        assert "[truncated tool output]" in out
        assert "Observation 1: short" in out
        assert "Action 0: call" in out
        assert len(out) <= 400

    def test_format_helpers_match_the_reference_shape(self):
        assert format_reflections([]) == ""
        text = format_reflections(["a", "b"])
        assert text.startswith(REFLECTION_HEADER)
        assert "Reflections:\n- a\n- b" in text
        last = format_last_attempt("Q?", "Action 0: x")
        assert last.startswith(LAST_TRIAL_HEADER)
        assert "(END PREVIOUS TRIAL)" in last

    def test_render_scratchpad_from_a_real_task_result(self, tmp_path):
        runner, rx, env = build(tmp_path, make_mcqa_task(), ReflexionStrategy.NONE)
        res = rx.run_task(make_mcqa_task(), env)
        pad = render_scratchpad(res.final_task_result)
        assert "search_guidelines" in pad
        assert "Observation 0:" in pad
        assert "submit_answer" in pad

    def test_attempt_record_is_json_serialisable(self):
        rec = AttemptRecord(
            task_id="t", domain="d", strategy="reflexion", attempt_idx=0,
            success=None, success_signal="ungraded", score=0.0,
        )
        assert json.loads(json.dumps(rec.to_dict()))["success"] is None
