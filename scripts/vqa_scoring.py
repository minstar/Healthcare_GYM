"""CF-EM: closed-form exact match for the Healthcare-GYM visual-QA benchmarks.

WHY THIS EXISTS
---------------
The shipped scorer (`eval_benchmark_multiturn._check_answer`) ends in

    if gold and gold.lower() in submitted.lower():
        return True

Plain substring containment. VQA-RAD and SLAKE golds are short (62% of VQA-RAD
golds are <=4 chars: no=133, yes=118 of 451; SLAKE 54%: no=180, yes=175 of
1061), so any sufficiently long hedging answer contains a gold token by
accident. ONE fixed 317-character paragraph, submitted for every task, never
looking at the image, is scored correct on 56.5% of VQA-RAD and 45.4% of SLAKE.
That is above three of the SLAKE rows the paper prints.

WHAT THIS DOES INSTEAD
----------------------
VQA-RAD/SLAKE/PathVQA are closed-vocabulary classification, not free text.
VQA-RAD has 121 distinct golds over 451 items and 68% of items carry a gold
seen >=3x; SLAKE 132 over 1061 with 92%. The answer space is finite and ships
with the benchmark, so nothing needs to be fuzzy-matched:

    1. EXTRACT the span the model committed to -- score the ANSWER, not the
       transcript.
    2. CLASSIFY that span into a SET of labels drawn from the benchmark's own
       gold vocabulary (forced choice; no commitment = wrong).
    3. Exact-match the predicted set against the gold set.

The vocabulary is built from `test.json` golds ONLY. No model output ever
enters it, and question conditioning is leave-one-out so an item's own gold is
never visible to its own scoring.

SCOPE
-----
Only meaningful where the answer space is actually closed. Measured over
`datasets/vqa/*/test.json`:

    benchmark        n   %distinct golds   %items w/ gold seen>=3x   mean gold chars
    vqa_rad        451             26.8%                     68.1%              7.8
    slake         1061             12.4%                     91.8%              7.3
    pathvqa       6719             19.3%                     79.8%             10.6
    pmc_vqa       1996             80.7%                     18.6%             21.2
    vqa_med_2021   500             73.0%                     25.4%             51.6

pmc_vqa and vqa_med_2021 fail both tests, and the shipped substring defect is
absent there anyway (the blind hedge paragraph scores 2.25 and 0.00, at or
below chance). CF-EM is therefore scoped to CLOSED_VOCAB_BENCHMARKS and those
two sets keep the substring rule.

METRICS
-------
CF-EM   : accuracy of the predicted label set. Directly comparable to the
          accuracy the paper prints.
CF-BAcc : macro-averaged per-class recall over the closed vocabulary. A GUARD,
          not a headline -- it macro-averages 117 VQA-RAD classes of which 56%
          hold a single item, and its bootstrap CI is ~12 pp wide on a 25-point
          number. Its job is to pin any constant answerer at ~1/|labels|
          (measured: 0.76-0.85%), which is what makes constant-answer gaming
          visible at a glance.

CALIBRATION (all figures reproduced by scripts/rebuttal/test_vqa_scoring.py)
----------------------------------------------------------------------------
                                       vqa_rad            slake
    image-blind hedge paragraph   shipped 56.54    shipped 45.43
                                   CF-EM  29.49     CF-EM  16.97   (== chance)
    all-golds label dump          shipped 100.00   shipped 100.00
                                   CF-EM   2.66     CF-EM   3.11
    gold dressed in 319c of hedge  CF-EM  99.78     CF-EM  98.78

KNOWN LIMIT
-----------
CF-EM reads the FRONT of the answer, so an answer that leads with a
contradicting commitment and never marks its final answer is scored on the
lead. Measured cost on real rollouts is ~1% of rows (94-97% of polar answers
open with a literal yes/no), but it is a genuine emission-contract dependence
and is declared here rather than hidden.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict

# Bump when a change can move a published number. Recorded inside every
# results artifact so a score can always be traced to the rule that made it.
CF_EM_VERSION = "cf_em/1.0"

# Benchmarks whose answer space is closed enough for CF-EM to be meaningful.
CLOSED_VOCAB_BENCHMARKS = frozenset({"vqa_rad", "slake", "pathvqa"})

# All visual-QA benchmarks the harness knows about.
VQA_BENCHMARKS = frozenset({
    "vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa",
})

POLAR = ("yes", "no")
POLAR_WINDOW = 8    # a polarity cue must fire in the first K tokens of the span
SPAN_CAP = 100      # characters
OPEN_SLACK = 2      # content tokens of slack, applied to the SHORT span only

STOP = {"the", "a", "an", "of", "is", "are", "in", "on", "at", "to", "and",
        "it", "this", "that", "there", "s", "was", "were", "be", "as",
        "with", "for"}

PREP = {"of", "in", "on", "within", "at", "inside", "near", "from", "the"}

# Generic trailing head nouns stripped during vocabulary canonicalisation, so
# the benchmark's own surface duplicates collapse ('right'/'right side',
# 'axial'/'axial plane').
GENERIC_TAIL = ("side", "region", "area", "part", "zone", "aspect",
                "plane", "view")

_PUNCT_RE = re.compile(r"[^0-9a-z%\.\-\s]+")
_ART_RE = re.compile(r"^(the|a|an)\s+")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def norm(s: str) -> str:
    """Lowercase, strip punctuation, keep decimals ('3.4' survives, 'T2.' -> 't2')."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = s.replace("’", "'").replace("_", " ").replace("/", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    prev = None
    while prev != s:
        prev = s
        s = _ART_RE.sub("", s)
    return s.strip()


def toks(s: str) -> list:
    return norm(s).split()


def content(s: str) -> frozenset:
    return frozenset(t for t in toks(s) if t not in STOP)


def canon(label: str) -> str:
    """Vocabulary canonicalisation. Applied to GOLDS, never to model output."""
    t = norm(label).split()
    while len(t) > 1 and t[-1] in GENERIC_TAIL:
        t.pop()
    return " ".join(t)


def gold_set(gold: str) -> frozenset:
    """SLAKE ships multi-label golds ('Lung, Spinal Cord'); a gold is a SET."""
    return frozenset(p for p in (canon(x) for x in str(gold).split(",")) if p)


# --------------------------------------------------------------------------
# STAGE 1: answer extraction -- score the ANSWER, not the transcript
# --------------------------------------------------------------------------

_MARKER_RE = re.compile(
    r"final answer\s*[:\-]|\banswer\s*[:\-]|\bthe answer is\b|\banswer is\b", re.I)

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+|\n+|;\s+")

# Clause boundaries inside the first sentence. Deliberately excludes " with "
# (gold 'cardiomegaly with pulmonary edema') and " of " (gold 'head of the
# pancreas') so multi-word golds survive.
_CLAUSE_SPLIT = re.compile(
    r",|\s+[-–—]\s+|\s+\bbecause\b|\s+\bsince\b|\s+\bhowever\b|"
    r"\s+\bbut\b|\s+\balthough\b|\s+\bwhich\b|\s+\bwhile\b|\s+\bbased on\b|"
    r"\s+\bgiven\b|\s+\bdue to\b|\s+\bthough\b|\s+\bso that\b|\s+\btherefore\b",
    re.I)

# Leading clauses that are pure framing and carry no answer.
_PREAMBLE_RE = re.compile(
    r"^\s*(based on|looking at|from (the|this)|after (a )?(review|reviewing|"
    r"analy\w+|examin\w+|careful\w*)|upon (review\w*|examin\w*|inspect\w*)|"
    r"according to|considering|given (the|this)|examining|reviewing|"
    r"analy(zing|sing)|in (this|the) (image|scan|ct|mri|x-ray|xray|"
    r"radiograph|study|figure|photo)|on (this|the) (image|scan|ct|mri|x-ray|"
    r"xray|radiograph|study)|as (a|an) (radiologist|ai|assistant)|"
    r"to answer (this|the|your)|first|overall|in summary|in short|"
    r"let me|i (will|have|need|should)|the (user|question) (is|asks|wants))\b",
    re.I)


def _demark(text: str) -> str:
    """Strip markdown, then cut to the text after the LAST answer marker."""
    t = re.sub(r"[*_`#>]+", " ", str(text)).strip().strip('"').strip()
    if not t:
        return ""
    hits = list(_MARKER_RE.finditer(t))
    if hits:
        tail = t[hits[-1].end():].strip()
        tail = _SENT_SPLIT.split(tail)[0].strip() if tail else ""
        if tail:
            return tail
    return t


def clause_span(submitted: str) -> str:
    """First clause of the first sentence. Used by the POLAR head: polarity is
    announced at the head of the sentence."""
    if not submitted:
        return ""
    t = str(submitted).strip()
    t = re.sub(r"[*_`#>]+", " ", t)
    t = re.sub(r"^\s*[-•]\s*", "", t)
    t = t.strip().strip('"').strip()
    if not t:
        return ""
    t = _demark(t)
    first = _SENT_SPLIT.split(t)[0].strip() or t[:SPAN_CAP]
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(first) if p and p.strip()]
    if parts:
        dropped = 0
        while len(parts) > 1 and dropped < 2 and _PREAMBLE_RE.match(parts[0]):
            parts.pop(0)
            dropped += 1
        first = parts[0]
    return first[:SPAN_CAP].strip()


def sentence_span(submitted: str) -> str:
    """First sentence with commas RETAINED, leading framing clauses dropped.
    Used by the CLOSED head: labels are the sentence's complement and can sit
    after a comma."""
    if not submitted:
        return ""
    t = re.sub(r"[*_`#>]+", " ", str(submitted)).strip().strip('"').strip()
    if not t:
        return ""
    t = _demark(t)
    first = (_SENT_SPLIT.split(t)[0] or t).strip()
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(first) if p and p.strip()]
    dropped = 0
    while len(parts) > 1 and dropped < 2 and _PREAMBLE_RE.match(parts[0]):
        parts.pop(0)
        dropped += 1
    return ", ".join(parts)[:2 * SPAN_CAP].strip()


# --------------------------------------------------------------------------
# STAGE 2a: polarity head (yes/no golds)
# --------------------------------------------------------------------------

NO_CUES = {
    3: {"there is no", "there are no", "it is not", "there is not",
        "there are not", "i do not", "this is not", "that is not",
        "there is none", "no there is", "does not appear", "do not appear",
        "there was no"},
    2: {"is not", "are not", "does not", "do not", "did not", "was not",
        "were not", "no evidence", "not present", "no significant",
        "no acute", "not seen", "not visible", "not appear", "no i",
        "cannot be", "no focal", "no definite", "no obvious"},
    1: {"no", "not", "none", "never", "negative", "absent", "false",
        "incorrect", "nope", "without", "nothing", "neither", "non",
        "cannot", "doesn't", "isn't", "aren't", "don't", "didn't",
        "wasn't", "weren't", "unlikely", "normal"},
}
YES_CUES = {
    3: {"there is a", "there is an", "yes there is", "yes it is",
        "there appears to", "there does appear"},
    2: {"there is", "there are", "it is", "this is", "it does", "that is",
        "image shows", "shows a", "yes there", "appears to", "i can",
        "the finding", "consistent with"},
    1: {"yes", "yeah", "yep", "yup", "correct", "true", "affirmative",
        "present", "positive", "indeed", "affirmed", "confirmed",
        "abnormal"},
}


def polarity(span: str, K: int = POLAR_WINDOW) -> str:
    """'yes' | 'no' | '' (no commitment).

    First cue in reading order wins; at one position the longer phrase wins,
    and NO is checked before YES at equal length so 'there is no' is not
    swallowed by 'there is'. The cue must land in the first K tokens: a cue
    buried deep in a sentence is a restatement of the question, not a
    commitment. K is inert -- K=3, K=8 and K=inf give the same arm ordering
    and land within 0.2 pp of the same score.
    """
    tk = toks(span)
    n = len(tk)
    for i in range(min(n, K)):
        for L in (3, 2, 1):
            if i + L > n:
                continue
            ph = " ".join(tk[i:i + L])
            if ph in NO_CUES[L]:
                return "no"
            if ph in YES_CUES[L]:
                return "yes"
    return ""


# --------------------------------------------------------------------------
# STAGE 2b: closed-set head
# --------------------------------------------------------------------------

def _positions(span_tokens, label_tokens):
    n, m = len(span_tokens), len(label_tokens)
    if m == 0 or m > n:
        return []
    return [i for i in range(n - m + 1) if span_tokens[i:i + m] == label_tokens]


class Vocab:
    """The benchmark's own closed answer vocabulary.

    Built from the FULL test.json, never from a --limit subset: a vocabulary
    that changed with the sample size would make scores incomparable between
    runs.
    """

    def __init__(self, items, knn=10, min_sim=0.35, min_group=3):
        self.items = items
        self.gsets = [gold_set(x["answer"]) for x in items]
        self.freq = Counter(l for g in self.gsets for l in g)
        self.labels = {l for l in self.freq if l and l not in POLAR}
        self.ltok = {l: l.split() for l in self.labels}
        self.head = {l for l in self.labels if self.freq[l] >= 3}
        self.qtok = [toks(x.get("question", "")) for x in items]
        self.qset = [frozenset(t for t in q if t not in STOP) for q in self.qtok]
        self.knn, self.min_sim, self.min_group = knn, min_sim, min_group
        self._admis = {}

    def admissible(self, idx):
        """Question-conditioned admissible label set, LEAVE-ONE-OUT.

        The item's own gold is excluded by construction (j != idx), so this
        cannot leak the answer. Returns None when the neighbourhood is too
        small, in which case no filtering happens -- the rule can only ever
        remove a distractor, never add information.
        """
        if idx in self._admis:
            return self._admis[idx]
        q = self.qset[idx]
        sims = sorted(((len(q & qj) / len(q | qj), j)
                       for j, qj in enumerate(self.qset)
                       if j != idx and q and qj
                       and len(q & qj) / len(q | qj) >= self.min_sim),
                      reverse=True)
        nb = [j for _, j in sims[:self.knn]]
        out = None
        if len(nb) >= self.min_group:
            lab = {l for j in nb for l in self.gsets[j]} & self.labels
            if lab:
                out = lab
        self._admis[idx] = out
        return out

    def kind(self, gold):
        g = gold_set(gold)
        return "polar" if (len(g) == 1 and next(iter(g)) in POLAR) else "closed"

    def predict_set(self, span, admis=None, qtok=None, drop_pp=True,
                    drop_qwords=True):
        """Longest-first, non-overlapping label matches, then three filters."""
        st = toks(span)
        if not st:
            return frozenset()
        hits = [(l, i, i + len(self.ltok[l]))
                for l in self.labels for i in _positions(st, self.ltok[l])]
        if not hits:
            return frozenset()
        hits.sort(key=lambda h: (-(h[2] - h[1]), h[1]))
        taken, spans = [], []
        for l, a, b in hits:
            if any(a < y and x < b for x, y in spans):
                continue
            taken.append((l, a, b))
            spans.append((a, b))

        # (1) question conditioning: filters MATCHES, never the matching itself
        if admis is not None and len(taken) > 1:
            k = [t for t in taken if t[0] in admis]
            if k:
                taken = k
        # (2) a label the QUESTION already names is given, not answered. Guarded
        #     by "only if something survives", so an alternative-forced question
        #     ('is this a CT or an MRI?') is left untouched.
        if drop_qwords and qtok and len(taken) > 1:
            k = [t for t in taken if not _positions(qtok, self.ltok[t[0]])]
            if k:
                taken = k
        # (3) a label inside a PP hanging off another match is a modifier:
        #     'right upper lobe OF THE lung' answers 'right upper lobe'
        if drop_pp and len(taken) > 1:
            ends = {b for _, _, b in taken}
            k = []
            for l, a, b in taken:
                j = a - 1
                while j >= 0 and st[j] in PREP:
                    j -= 1
                if j < a - 1 and (j + 1) in ends and (j + 1) != a:
                    continue
                k.append((l, a, b))
            if k:
                taken = k
        return frozenset(l for l, _, _ in taken)


# --------------------------------------------------------------------------
# STAGE 3: decision
# --------------------------------------------------------------------------

def cf_predict(submitted, gold, vocab, idx=None, use_qtype=True,
               use_sentence=True, slack=True, drop_pp=True, drop_qwords=True,
               K=POLAR_WINDOW):
    """-> (predicted label set, is_correct, kind, scored span).

    Correct iff the predicted set EQUALS the gold set. No commitment (empty
    set) or a set naming several things is wrong. For a single-label gold this
    reduces exactly to 'commit to exactly one label'; the set formulation
    exists only to handle SLAKE's multi-answer golds.
    """
    gs = gold_set(gold)
    kind = vocab.kind(gold)

    if kind == "polar":
        sp = clause_span(submitted)
        p = polarity(sp, K)
        return (frozenset([p]) if p else frozenset()), (p == next(iter(gs))), kind, sp

    sp = sentence_span(submitted) if use_sentence else clause_span(submitted)
    if not sp:
        return frozenset(), False, kind, sp
    adm = vocab.admissible(idx) if (use_qtype and idx is not None) else None
    qt = vocab.qtok[idx] if idx is not None else None
    ps = vocab.predict_set(sp, adm, qt, drop_pp=drop_pp, drop_qwords=drop_qwords)
    if ps:
        return ps, (ps == gs), kind, sp

    # Nothing matched: one bounded fallback, content-token-set equality with
    # <=2 tokens of slack on the SHORT span only, so it can never degenerate
    # into paragraph containment.
    if slack and len(gs) == 1:
        cs, cg = content(sp), content(next(iter(gs)))
        if cs and cg and (cs == cg or (cg <= cs and len(cs) <= len(cg) + OPEN_SLACK)):
            return gs, True, kind, sp
    return frozenset(), False, kind, sp


def check_answer_cf(submitted, gold, vocab, idx=None) -> bool:
    """Boolean drop-in matching `_check_answer`'s contract."""
    return cf_predict(submitted, gold, vocab, idx=idx)[1]


# --------------------------------------------------------------------------
# metrics + loading
# --------------------------------------------------------------------------

def score_all(records, vocab, idx_of=None, **kw):
    """records: [{'submitted','gold', ...}] -> (rows, summary dict)."""
    if idx_of is None:
        idx_of = lambda r: r.get("_i")  # noqa: E731
    rows = []
    for r in records:
        ps, ok, kind, sp = cf_predict(r["submitted"], r["gold"], vocab,
                                      idx=idx_of(r), **kw)
        rows.append({**r, "cf_pred": sorted(ps), "cf_correct": ok,
                     "cf_kind": kind, "cf_span": sp})
    n = len(rows)
    s = {"n": n, "cf_em": sum(x["cf_correct"] for x in rows) / n if n else 0.0}
    for key, sel in [("polar", lambda x: x["cf_kind"] == "polar"),
                     ("closed_head", lambda x: x["cf_kind"] == "closed"
                      and gold_set(x["gold"]) <= vocab.head),
                     ("closed_tail", lambda x: x["cf_kind"] == "closed"
                      and not gold_set(x["gold"]) <= vocab.head)]:
        sub = [x["cf_correct"] for x in rows if sel(x)]
        s[f"n_{key}"] = len(sub)
        s[f"cf_em_{key}"] = (sum(sub) / len(sub)) if sub else float("nan")
    bk = defaultdict(list)
    for x in rows:
        bk[tuple(sorted(gold_set(x["gold"])))].append(x["cf_correct"])
    rec = [sum(v) / len(v) for v in bk.values()]
    s["cf_bacc"] = sum(rec) / len(rec) if rec else 0.0
    s["n_labels"] = len(rec)
    s["no_commit_rate"] = sum(1 for x in rows if not x["cf_pred"]) / n if n else 0.0
    return rows, s


def task_row_index(task_id, benchmark_name):
    """`load_vqa_benchmark` mints ids as f'{name}_{row}', so the dataset row is
    recoverable from the id. Returns None if the id does not follow the scheme,
    in which case question conditioning is simply skipped for that item."""
    if not task_id or not benchmark_name:
        return None
    prefix = f"{benchmark_name}_"
    if not str(task_id).startswith(prefix):
        return None
    tail = str(task_id)[len(prefix):]
    return int(tail) if tail.isdigit() else None


_VOCAB_CACHE = {}


def load_vocab(benchmark_name, project_root, **kw):
    """Build (and cache) the closed vocabulary from the FULL benchmark file.

    Returns None when the benchmark file is missing or the benchmark is not a
    closed-vocabulary set, so the caller falls back to the substring rule.
    """
    if benchmark_name in _VOCAB_CACHE:
        return _VOCAB_CACHE[benchmark_name]
    if benchmark_name not in CLOSED_VOCAB_BENCHMARKS:
        _VOCAB_CACHE[benchmark_name] = None
        return None
    from pathlib import Path
    data_dir = Path(project_root) / "datasets" / "vqa" / benchmark_name
    path = None
    for cand in ("test.json", "test.jsonl"):
        if (data_dir / cand).exists():
            path = data_dir / cand
            break
    if path is None:
        _VOCAB_CACHE[benchmark_name] = None
        return None
    with open(path) as f:
        if path.suffix == ".jsonl":
            items = [json.loads(l) for l in f if l.strip()]
        else:
            items = json.load(f)
    items = [{"question": it.get("question", it.get("input", "")),
              "answer": str(it.get("answer", it.get("output", ""))).strip()}
             for it in items]
    vocab = Vocab(items, **kw)
    _VOCAB_CACHE[benchmark_name] = vocab
    return vocab
