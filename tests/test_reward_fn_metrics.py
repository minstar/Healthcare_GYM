"""Tests for scripts/verl/reward_fn.py after it started returning a dict.

The load-bearing test here is `TestScoreIsBitIdentical`.  `compute_score`
used to return a bare float that fed the GRPO advantage; it now returns a dict
whose "score" key must be the SAME float, to the bit, for every input and every
environment configuration.  A regression there silently corrupts a training run
rather than failing loudly, so the baseline is not a hand-written table: it is
the actual pre-change module, loaded from git rev `dd962b9` (with a vendored
snapshot as fallback) and executed side by side with the current one.

The rest of the file covers the new metrics: `acc` must be invariant to
COSINE_REWARD and to response length (that is the whole point — `score` is not),
the DEGENERATE_EXCLUDE sentinel must survive the dict, and the payload must obey
the two constraints verl imposes on reward_extra_info (identical keys on every
sample, JSON/numpy-safe value types).

Run:
    PYTHONPATH=<repo> python -m pytest tests/test_reward_fn_metrics.py -v
"""

import contextlib
import importlib.util
import itertools
import json
import os
import pathlib
import random
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
REWARD_FN = REPO / "scripts" / "verl" / "reward_fn.py"

# The commit reward_fn.py sat at before the dict return was introduced.
BASELINE_REV = "dd962b9"
BASELINE_REL = "scripts/verl/reward_fn.py"
VENDORED_BASELINE = pathlib.Path(__file__).parent / "data" / "reward_fn_baseline_dd962b9.py"


# ── module loading under a controlled environment ────────────────────
# reward_fn.py reads every knob at import time, so a configuration is a module
# load, not a function argument.

ENV_KEYS = (
    "REWARD_DEBUG_LOG",
    "DEGENERATE_FILTER", "DEGENERATE_REWARD", "DEGENERATE_NGRAM_THRESHOLD",
    "DEGENERATE_MIN_LENGTH", "DEGENERATE_EXCLUDE", "DEGENERATE_GIBBERISH",
    "COSINE_REWARD", "COSINE_L_MAX", "COSINE_CHARS_PER_TOKEN",
    "COSINE_R0_CORRECT", "COSINE_RL_CORRECT",
    "COSINE_R0_WRONG", "COSINE_RL_WRONG", "COSINE_R_EXCEED",
)

_load_counter = itertools.count()


@contextlib.contextmanager
def _isolated_env(overrides):
    """Clear every knob reward_fn.py reads, then apply `overrides`."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: str(v) for k, v in overrides.items()})
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_reward_module(path, env_overrides):
    """Import `path` as a fresh module with `env_overrides` in place."""
    with _isolated_env(env_overrides):
        name = f"_reward_fn_{next(_load_counter)}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


# ── environment matrix ───────────────────────────────────────────────
# COSINE_* values are the ones runs/train_hcgym.slurm exports, so the two
# "study_*" entries are literally the grpo and grpo_cosine/ttopd arm configs.

_STUDY_COSINE = {
    "COSINE_REWARD": "1",
    "COSINE_L_MAX": "12288",
    "COSINE_CHARS_PER_TOKEN": "5.0",
    "COSINE_R0_CORRECT": "1.1",
    "COSINE_RL_CORRECT": "0.7",
    "COSINE_R0_WRONG": "0.0",
    "COSINE_RL_WRONG": "-0.3",
    "COSINE_R_EXCEED": "-0.5",
}
_STUDY_DEGEN = {
    "DEGENERATE_FILTER": "1",
    "DEGENERATE_EXCLUDE": "1",
    "DEGENERATE_NGRAM_THRESHOLD": "0.3",
    "DEGENERATE_MIN_LENGTH": "200",
    "DEGENERATE_GIBBERISH": "1",
}

ENV_CONFIGS = {
    # nothing on — the plainest possible reward
    "bare": {},
    # cosine only, at library defaults
    "cosine_default": {"COSINE_REWARD": "1"},
    # cosine with non-default schedule, to exercise the env parsing
    "cosine_custom": {
        "COSINE_REWARD": "1", "COSINE_L_MAX": "4096", "COSINE_CHARS_PER_TOKEN": "3.5",
        "COSINE_R0_CORRECT": "1.4", "COSINE_RL_CORRECT": "0.2",
        "COSINE_R0_WRONG": "-0.1", "COSINE_RL_WRONG": "-0.9", "COSINE_R_EXCEED": "-1.25",
    },
    # degenerate filter with the soft penalty
    "degen_soft": {"DEGENERATE_FILTER": "1"},
    # degenerate filter with a non-default soft penalty
    "degen_soft_custom": {"DEGENERATE_FILTER": "1", "DEGENERATE_REWARD": "-2.5"},
    # degenerate filter with the sentinel
    "degen_sentinel": {"DEGENERATE_FILTER": "1", "DEGENERATE_EXCLUDE": "1"},
    # + enhanced gibberish detection
    "degen_gibberish": {"DEGENERATE_FILTER": "1", "DEGENERATE_EXCLUDE": "1",
                        "DEGENERATE_GIBBERISH": "1"},
    # the real arms, verbatim from train_hcgym.slurm (REWARD_DEBUG_LOG included)
    "study_grpo": {"REWARD_DEBUG_LOG": "1", **_STUDY_DEGEN},
    "study_cosine": {"REWARD_DEBUG_LOG": "1", **_STUDY_DEGEN, **_STUDY_COSINE},
}


# ── input matrix ─────────────────────────────────────────────────────

_VOCAB = (
    "the is of and in to a for with that this patient presents acute onset "
    "dyspnea pleuritic chest pain differential includes pulmonary embolism "
    "pneumothorax myocardial infarction pericarditis costochondritis workup "
    "electrocardiogram troponin d-dimer computed tomography angiography "
    "ventilation perfusion scan echocardiogram arterial blood gas lactate "
    "hemoglobin platelets creatinine electrolytes coagulation panel imaging "
    "history examination auscultation percussion palpation inspection vitals "
    "tachycardia hypotension hypoxemia fever chills diaphoresis syncope "
    "anticoagulation heparin warfarin apixaban thrombolysis embolectomy "
    "oxygen supplementation fluid resuscitation vasopressor norepinephrine "
    "risk stratification wells score geneva criteria pesi index outcome "
    "prognosis mortality morbidity recurrence prophylaxis compression "
    "guideline recommendation evidence quality strength consensus statement"
).split()


def _filler(n_chars, seed=1234):
    """Deterministic, non-repetitive, non-degenerate prose of exactly n chars."""
    rng = random.Random(seed)
    parts, total = [], 0
    while total < n_chars + 32:
        w = rng.choice(_VOCAB)
        parts.append(w)
        total += len(w) + 1
    return " ".join(parts)[:n_chars]


def _mc(n_chars, letter="C", marker="Answer: ", seed=1234, prefix=""):
    """A multiple-choice response of exactly n_chars ending in an answer."""
    tail = "\n" + marker + letter
    body = _filler(n_chars - len(tail) - len(prefix), seed=seed)
    return prefix + body + tail


def _repetitive(n_chars):
    unit = "The patient has a fever and needs treatment now. "
    return (unit * (n_chars // len(unit) + 2))[:n_chars]


def _gibberish(n_chars, seed=7):
    """ASCII word salad: unique 4-grams, so only the GIBBERISH check catches it."""
    rng = random.Random(seed)
    alpha = "bcdfghjklmnpqrstvwxz"
    parts, total = [], 0
    while total < n_chars + 8:
        w = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 3)))
        parts.append(w)
        total += len(w) + 1
    return " ".join(parts)[:n_chars]


_VALID_CALL = "<function=search_pubmed>{}</function>"
_INVALID_CALL = "<function=totally_made_up_tool>{}</function>"

_OPEN_GT = "pulmonary embolism anticoagulation heparin"


def _open(n_chars, answer, seed=99, prefix=""):
    tail = "\nAnswer: " + answer
    body = _filler(n_chars - len(tail) - len(prefix), seed=seed)
    return prefix + body + tail


# id -> (solution_str, ground_truth, extra_info)
CASES = {
    # --- multiple choice, correct, across the cosine length sweep -------------
    "mc_correct_1996":   (_mc(1996), "C", {"has_options": True}),
    "mc_correct_8003":   (_mc(8003), "C", {"has_options": True}),
    "mc_correct_20002":  (_mc(20002), "C", {"has_options": True}),
    "mc_correct_45001":  (_mc(45001), "C", {"has_options": True}),
    # past COSINE_L_MAX * COSINE_CHARS_PER_TOKEN (61440) -> R_EXCEED clip
    "mc_correct_62000":  (_mc(62000), "C", {"has_options": True}),
    # --- multiple choice, wrong / unparseable --------------------------------
    "mc_wrong_1996":     (_mc(1996, letter="B"), "C", {"has_options": True}),
    "mc_wrong_20002":    (_mc(20002, letter="B"), "C", {"has_options": True}),
    "mc_no_answer":      (_filler(2000), "C", {"has_options": True}),
    # correct but not in "Answer: X" form -> no format bonus
    "mc_correct_noformat": (_mc(2000, marker="the answer is "), "C", {"has_options": True}),
    # parenthesised trailing letter
    "mc_correct_paren":  (_filler(1980) + "\nThe best choice is (C).", "C", {"has_options": True}),
    # --- hallucinated tool calls ---------------------------------------------
    "mc_correct_1invalid": (_mc(2000, prefix=_INVALID_CALL), "C", {"has_options": True}),
    "mc_correct_2invalid": (_mc(2000, prefix=_INVALID_CALL * 2), "C", {"has_options": True}),
    "mc_correct_valid_tools": (_mc(2000, prefix=_VALID_CALL * 3), "C", {"has_options": True}),
    "mc_wrong_3invalid": (_mc(2000, letter="B", prefix=_INVALID_CALL * 3), "C", {"has_options": True}),
    # --- degenerate ----------------------------------------------------------
    "mc_degen_repetitive": (_repetitive(4000) + "\nAnswer: C", "C", {"has_options": True}),
    "mc_degen_tooshort":   ("Answer: C", "C", {"has_options": True}),
    "mc_degen_assistant":  ("Answer: C\n" + "\nassistant\n" * 40 + _filler(600),
                            "C", {"has_options": True}),
    # only DEGENERATE_GIBBERISH catches this one
    "mc_gibberish_34k":  (_gibberish(34000), "C", {"has_options": True}),
    # --- ground-truth normalisation / has_options inference ------------------
    "mc_gt_answer_paren": (_mc(2000, letter="D"), "ANSWER: (D)", {}),
    "mc_gt_paren_only":   (_mc(2000, letter="D"), "(D)", {}),
    "mc_gt_bare":         (_mc(2000, letter="D"), "D", {}),
    "mc_extra_info_none": (_mc(2000, letter="D"), "D", None),
    "mc_extra_info_str":  (_mc(2000, letter="D"), "D", json.dumps({"has_options": True})),
    # --- open ended ----------------------------------------------------------
    "open_correct_1996":  (_open(1996, "pulmonary embolism treated with anticoagulation heparin"),
                           _OPEN_GT, {"has_options": False}),
    "open_correct_8003":  (_open(8003, "pulmonary embolism treated with anticoagulation heparin"),
                           _OPEN_GT, {"has_options": False}),
    "open_correct_45001": (_open(45001, "pulmonary embolism treated with anticoagulation heparin"),
                           _OPEN_GT, {"has_options": False}),
    "open_correct_62000": (_open(62000, "pulmonary embolism treated with anticoagulation heparin"),
                           _OPEN_GT, {"has_options": False}),
    "open_partial":       (_open(2000, "possibly a pneumothorax with heparin held for now"),
                           _OPEN_GT, {"has_options": False}),
    "open_zero":          (_open(2000, "reassurance and discharge home"),
                           _OPEN_GT, {"has_options": False}),
    "open_no_marker":     (_filler(2000), _OPEN_GT, {"has_options": False}),
    "open_submit_tool":   ("<function=submit_answer>{pulmonary embolism anticoagulation heparin}"
                           "</function>" + _filler(1500), _OPEN_GT, {"has_options": False}),
    "open_empty_solution": ("", _OPEN_GT, {"has_options": False}),
    "open_empty_gt":      (_open(2000, "pulmonary embolism anticoagulation heparin"),
                           "", {"has_options": False}),
    "open_2invalid":      (_open(2000, "pulmonary embolism treated with anticoagulation heparin",
                                 prefix=_INVALID_CALL * 2), _OPEN_GT, {"has_options": False}),
    "open_degen_repetitive": (_repetitive(4000) + "\nAnswer: pulmonary embolism anticoagulation heparin",
                              _OPEN_GT, {"has_options": False}),
    "open_gibberish_34k": (_gibberish(34000), _OPEN_GT, {"has_options": False}),
}

MATRIX = [(env_id, case_id) for env_id in ENV_CONFIGS for case_id in CASES]
MATRIX_IDS = [f"{e}-{c}" for e, c in MATRIX]


# ── fixtures ─────────────────────────────────────────────────────────

def _baseline_source_from_git():
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASELINE_REV}:{BASELINE_REL}"],
            capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.decode()


@pytest.fixture(scope="session")
def baseline_path(tmp_path_factory):
    """The pre-change reward_fn.py, from git if possible, else the snapshot."""
    src = _baseline_source_from_git()
    if src is None:
        src = VENDORED_BASELINE.read_text()
    out = tmp_path_factory.mktemp("reward_baseline") / "reward_fn_baseline.py"
    out.write_text(src)
    return out


@pytest.fixture(scope="session")
def modules(baseline_path):
    """{env_id: (baseline_module, current_module)} — one import pair per config."""
    pairs = {}
    for env_id, overrides in ENV_CONFIGS.items():
        pairs[env_id] = (
            load_reward_module(baseline_path, overrides),
            load_reward_module(REWARD_FN, overrides),
        )
    return pairs


def _call(module, case_id):
    solution, gt, extra = CASES[case_id]
    return module.compute_score("bioagents_medical", solution, gt, extra)


# ── the differential test ────────────────────────────────────────────

class TestScoreIsBitIdentical:
    """`score` must equal the float the pre-change function returned."""

    def test_snapshot_matches_git(self):
        """The vendored fallback is the real thing, not a stale copy."""
        src = _baseline_source_from_git()
        if src is None:
            pytest.skip("git unavailable; cannot cross-check the vendored snapshot")
        assert src == VENDORED_BASELINE.read_text(), (
            f"{VENDORED_BASELINE} has drifted from {BASELINE_REV}:{BASELINE_REL}"
        )

    def test_baseline_really_returns_a_float(self, modules):
        """Guard against silently diffing the new module against itself."""
        baseline, current = modules["study_cosine"]
        assert isinstance(_call(baseline, "mc_correct_1996"), float)
        assert isinstance(_call(current, "mc_correct_1996"), dict)

    @pytest.mark.parametrize("env_id,case_id", MATRIX, ids=MATRIX_IDS)
    def test_score_unchanged(self, modules, env_id, case_id):
        baseline, current = modules[env_id]
        before = _call(baseline, case_id)
        after = _call(current, case_id)["score"]
        assert after == before, (
            f"{env_id}/{case_id}: score changed {before!r} -> {after!r}"
        )
        # float equality above is the real assertion; repr equality catches a
        # -0.0/0.0 or int/float swap that == would let through.
        assert repr(after) == repr(before)

    def test_matrix_actually_exercises_the_shaping(self, modules):
        """A differential test that compares constants proves nothing."""
        distinct = set()
        for env_id in ENV_CONFIGS:
            baseline, _ = modules[env_id]
            for case_id in CASES:
                distinct.add(round(_call(baseline, case_id), 6))
        assert len(distinct) >= 20, f"matrix only produced {len(distinct)} distinct scores"


# ── the defect this change exists to fix ─────────────────────────────

class TestAccIsArmComparable:
    """`acc` must not move when the shaping does. That is the whole point."""

    _CORRECT = ["mc_correct_1996", "mc_correct_8003", "mc_correct_20002",
                "mc_correct_45001", "mc_correct_62000"]

    def test_score_does_move_with_cosine(self, modules):
        """Sanity: the defect is real, so the fix has something to fix."""
        _, plain = modules["study_grpo"]
        _, cosine = modules["study_cosine"]
        plain_scores = [_call(plain, c)["score"] for c in self._CORRECT]
        cosine_scores = [_call(cosine, c)["score"] for c in self._CORRECT]
        assert len(set(plain_scores)) == 1, plain_scores
        assert len(set(cosine_scores)) > 1, cosine_scores
        assert plain_scores != cosine_scores

    def test_acc_invariant_to_cosine_and_length(self, modules):
        _, plain = modules["study_grpo"]
        _, cosine = modules["study_cosine"]
        for case_id in self._CORRECT:
            assert _call(plain, case_id)["acc"] == 1.0
            assert _call(cosine, case_id)["acc"] == 1.0

    @pytest.mark.parametrize("case_id", sorted(CASES))
    def test_acc_identical_across_every_env(self, modules, case_id):
        """acc, and every other metric, must be the same under all 9 configs."""
        seen = {}
        for env_id in ENV_CONFIGS:
            _, current = modules[env_id]
            payload = _call(current, case_id)
            seen[env_id] = {k: v for k, v in payload.items()
                            if k not in ("score", "degenerate")}
        reference = seen["bare"]
        for env_id, payload in seen.items():
            assert payload == reference, f"{case_id}: {env_id} differs from bare"

    def test_acc_is_binary(self, modules):
        for env_id in ENV_CONFIGS:
            _, current = modules[env_id]
            for case_id in CASES:
                assert _call(current, case_id)["acc"] in (0.0, 1.0)

    def test_wrong_and_unanswered_score_zero_acc(self, modules):
        _, current = modules["study_cosine"]
        for case_id in ("mc_wrong_1996", "mc_wrong_20002", "mc_no_answer",
                        "open_zero", "open_empty_solution"):
            assert _call(current, case_id)["acc"] == 0.0, case_id

    def test_acc_partial_gives_open_ended_credit(self, modules):
        _, current = modules["study_cosine"]
        partial = _call(current, "open_partial")
        assert partial["acc"] == 0.0
        assert 0.0 < partial["acc_partial"] <= 0.5
        # multiple choice has no partial credit, so the two agree there
        for case_id in [c for c in CASES if c.startswith("mc_")]:
            payload = _call(current, case_id)
            assert payload["acc"] == payload["acc_partial"], case_id

    def test_answer_found_separates_wrong_from_silent(self, modules):
        _, current = modules["study_cosine"]
        assert _call(current, "mc_wrong_1996")["answer_found"] == 1.0
        assert _call(current, "mc_no_answer")["answer_found"] == 0.0
        assert _call(current, "open_correct_1996")["answer_found"] == 1.0
        assert _call(current, "open_submit_tool")["answer_found"] == 1.0
        # tail-window fallback: nothing was actually declared as an answer
        assert _call(current, "open_no_marker")["answer_found"] == 0.0
        assert _call(current, "open_empty_solution")["answer_found"] == 0.0

    def test_has_options_flags_the_branch(self, modules):
        _, current = modules["study_cosine"]
        for case_id in CASES:
            expected = 1.0 if case_id.startswith("mc_") else 0.0
            assert _call(current, case_id)["has_options"] == expected, case_id


class TestToolCallMetrics:
    """The 15.8%-hallucinated-tools number the paper reports, per sample."""

    def test_counts(self, modules):
        _, current = modules["study_cosine"]
        expected = {
            "mc_correct_1996": (0, 0),
            "mc_correct_1invalid": (1, 1),
            "mc_correct_2invalid": (2, 2),
            "mc_correct_valid_tools": (3, 0),
            "mc_wrong_3invalid": (3, 3),
            "open_2invalid": (2, 2),
            "open_submit_tool": (1, 0),
        }
        for case_id, (total, invalid) in expected.items():
            payload = _call(current, case_id)
            assert payload["n_tool_calls"] == float(total), case_id
            assert payload["n_invalid_tool_calls"] == float(invalid), case_id
            assert payload["has_invalid_tool_call"] == (1.0 if invalid else 0.0), case_id

    def test_rate_is_recoverable_as_a_ratio_of_sums(self, modules):
        """sum(invalid)/sum(total) over a batch, which is what the paper reports."""
        _, current = modules["study_cosine"]
        batch = ["mc_correct_1invalid", "mc_correct_valid_tools", "mc_wrong_3invalid"]
        payloads = [_call(current, c) for c in batch]
        total = sum(p["n_tool_calls"] for p in payloads)
        invalid = sum(p["n_invalid_tool_calls"] for p in payloads)
        assert total == 7.0 and invalid == 4.0
        assert invalid / total == pytest.approx(4 / 7)

    def test_counts_are_not_affected_by_the_penalty(self, modules):
        """The counts are raw; only `score` carries INVALID_TOOL_PENALTY."""
        _, current = modules["study_cosine"]
        clean = _call(current, "mc_correct_1996")
        dirty = _call(current, "mc_correct_2invalid")
        assert dirty["acc"] == clean["acc"] == 1.0
        assert dirty["score"] < clean["score"]


class TestDegenerateSentinel:
    """DEGENERATE_SENTINEL is load-bearing: core_algos.py matches on the value."""

    def test_sentinel_reaches_score_verbatim(self, modules):
        _, current = modules["degen_sentinel"]
        for case_id in ("mc_degen_repetitive", "mc_degen_tooshort", "mc_degen_assistant"):
            payload = _call(current, case_id)
            assert payload["score"] == -999.0, case_id
            assert payload["degenerate"] == 1.0, case_id

    def test_sentinel_survives_core_algos_threshold(self, modules):
        """Reproduce the exclusion predicate from core_algos.compute_grpo_*."""
        _, current = modules["study_cosine"]
        threshold = current.DEGENERATE_SENTINEL + 1.0
        assert _call(current, "mc_degen_repetitive")["score"] < threshold
        # ...and nothing legitimate is anywhere near it
        for case_id in CASES:
            payload = _call(current, case_id)
            if payload["degenerate"] == 0.0:
                assert payload["score"] > threshold, case_id

    def test_soft_penalty_when_exclude_is_off(self, modules):
        _, current = modules["degen_soft"]
        assert _call(current, "mc_degen_repetitive")["score"] == -1.0
        _, custom = modules["degen_soft_custom"]
        assert _call(custom, "mc_degen_repetitive")["score"] == -2.5

    def test_filter_off_means_flag_off(self, modules):
        _, current = modules["bare"]
        for case_id in CASES:
            assert _call(current, case_id)["degenerate"] == 0.0, case_id

    def test_gibberish_flag_is_what_distinguishes_the_configs(self, modules):
        _, without = modules["degen_sentinel"]
        _, with_gib = modules["degen_gibberish"]
        assert _call(without, "mc_gibberish_34k")["degenerate"] == 0.0
        assert _call(with_gib, "mc_gibberish_34k")["degenerate"] == 1.0

    def test_degenerate_still_reports_accuracy(self, modules):
        """The rollout is dropped from the advantage but still counted for acc."""
        _, current = modules["study_cosine"]
        payload = _call(current, "mc_degen_repetitive")
        assert payload["score"] == -999.0
        assert payload["acc"] == 1.0
        assert payload["answer_found"] == 1.0


class TestVerlContract:
    """The two constraints verl's reward plumbing imposes on the payload."""

    def test_same_keys_at_every_return_point(self, modules):
        """reward_loop.py:358 keys off sample 0 and indexes every other sample."""
        expected = None
        for env_id in ENV_CONFIGS:
            _, current = modules[env_id]
            assert set(current.METRIC_KEYS) == set(
                _call(current, "mc_correct_1996").keys()
            )
            for case_id in CASES:
                keys = tuple(sorted(_call(current, case_id).keys()))
                if expected is None:
                    expected = keys
                assert keys == expected, f"{env_id}/{case_id} returned {keys}"
        assert "score" in expected and "acc" in expected

    def test_every_value_is_a_plain_float(self, modules):
        """np.int64 is not JSON serialisable; _dump_generations would crash."""
        for env_id in ENV_CONFIGS:
            _, current = modules[env_id]
            for case_id in CASES:
                for key, value in _call(current, case_id).items():
                    assert type(value) is float, f"{env_id}/{case_id}/{key}={value!r}"

    def test_survives_the_numpy_and_json_round_trip(self, modules):
        """End-to-end shape of reward_loop.py:361 + ray_trainer._dump_generations."""
        np = pytest.importorskip("numpy")
        _, current = modules["study_cosine"]
        payloads = [_call(current, case_id) for case_id in CASES]
        keys = list(payloads[0].keys())  # exactly what reward_loop.py does
        columns = {k: np.array([p[k] for p in payloads]) for k in keys}
        for k, col in columns.items():
            assert col.dtype == np.float64, f"{k} -> {col.dtype}"
        for i in range(len(payloads)):
            json.dumps({k: v[i] for k, v in columns.items()})

    def test_verl_picks_acc_as_the_core_validation_metric(self, modules):
        """ray_trainer._val_metrics_update: core_var = "acc" if "acc" in vars."""
        _, current = modules["study_cosine"]
        assert "acc" in _call(current, "mc_correct_1996")


class TestValidateFlagIsGone:
    """Decision: shaping stays on everywhere; `acc` is the cross-arm metric."""

    def test_validate_true_no_longer_changes_anything(self, modules):
        """The only input whose score differs from the old code — and it is
        unreachable: no per-item extra_info anywhere sets "validate"."""
        _, current = modules["study_cosine"]
        solution, gt, extra = CASES["mc_correct_20002"]
        without = current.compute_score("bioagents_medical", solution, gt, dict(extra))
        with_flag = current.compute_score(
            "bioagents_medical", solution, gt, dict(extra, validate=True)
        )
        assert with_flag == without

    def test_old_code_did_honour_the_flag(self, modules):
        """Documents the divergence rather than hiding it."""
        baseline, current = modules["study_cosine"]
        solution, gt, extra = CASES["mc_correct_20002"]
        old_plain = baseline.compute_score("bioagents_medical", solution, gt, dict(extra))
        old_flagged = baseline.compute_score(
            "bioagents_medical", solution, gt, dict(extra, validate=True)
        )
        assert old_flagged != old_plain, "expected the old flag to disable shaping"
        new_flagged = current.compute_score(
            "bioagents_medical", solution, gt, dict(extra, validate=True)
        )["score"]
        assert new_flagged == old_plain

    def test_no_live_reference_to_the_flag_remains(self):
        source = REWARD_FN.read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert "is_validate" not in code
        assert 'get("validate"' not in code
        # ...but the reader is told why, loudly.
        assert "validate" in source

    def test_dataset_extra_info_never_carries_validate(self):
        """The claim the decision rests on, checked against the real parquet."""
        pd = pytest.importorskip("pandas")
        parquet = pathlib.Path(
            "/data/project/private/minstar/workspace/hcgym_rebuttal/data/"
            "verl_parquet/full_4modality_clean/test.parquet"
        )
        if not parquet.exists():
            pytest.skip(f"{parquet} not present")
        df = pd.read_parquet(parquet)
        keys = set()
        for row in df["extra_info"]:
            keys |= set(row.keys())
        assert "validate" not in keys
        assert {"has_options", "split"} <= keys
