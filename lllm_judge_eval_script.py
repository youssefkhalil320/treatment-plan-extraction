"""
Medical Treatment Plan Extraction — LLM-as-Judge Evaluation Script
===================================================================
Compares model extraction output against human annotator (gold) annotations
using google/medgemma-4b-it as the judge model.

─────────────────────────────────────────────────────────────────────────────
SCORING APPROACH
─────────────────────────────────────────────────────────────────────────────

  The judge model scores every (pred, gold) string pair using a fixed
  rubric embedded in the system prompt. Scores are strictly ordinal:

    1.0  — correct / clinically equivalent
    0.5  — partially correct (field-type-specific definition)
    0.0  — wrong, missing, or hallucinated

  Eight field-type groups each have their own rubric section.
  The judge receives the group name and both strings; it outputs a JSON
  object with a single key:

    {"score": 0 | 0.5 | 1}

  Trivial cases (null/null → 1.0, null/value or value/null → 0.0,
  exact match → 1.0) are resolved without calling the model.

─────────────────────────────────────────────────────────────────────────────
SCORING STRATEGY BY FIELD TYPE
─────────────────────────────────────────────────────────────────────────────

  Structured item lists  (labs, imaging, medications, monitoring, referrals)
  ──────────────────────
    • Items matched by name using greedy judge-score similarity
    • Match threshold: --match-threshold (default 0.5)
    • Matched pairs → sub-fields judged individually with the rubric
    • Unmatched gold items (missed) → 0.0
    • Unmatched pred items (hallucinated) → 0.0
    • field_avg = avg(all item scores including unmatched)

  Flat string lists  (conservative_method, lifestyle_habit_modifications)
  ─────────────────
    • Lists joined with " | " → single string per side
    • Judged as one unit using Group 6 rubric

  Single string fields  (patient_education, when_to_seek_medical_care,
                         follow_up sub-fields, referral aim)
  ─────────────────────
    • Direct judgement with the appropriate rubric group

─────────────────────────────────────────────────────────────────────────────
EFFICIENCY
─────────────────────────────────────────────────────────────────────────────

  All (group, pred, gold) triples are collected and deduplicated upfront.
  The judge model runs exactly once per unique triple.
  Results are cached in memory and optionally persisted to JSON with
  --cache-judgements so repeated runs are instant.

─────────────────────────────────────────────────────────────────────────────
CONFUSION DETECTION  (optional, enabled with --confusion)
─────────────────────────────────────────────────────────────────────────────

  Same confusion pairs as the reference scripts.
  Judge scores gold field_A text against pred field_B text.
  Flagged when score >= --confusion-threshold.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

  python evaluate_extraction_llm_judge.py --pred predictions.json --gold annotations.json
  python evaluate_extraction_llm_judge.py --pred predictions.json --gold annotations.json \\
      --output results.json
  python evaluate_extraction_llm_judge.py --pred predictions.json --gold annotations.json \\
      --confusion --confusion-threshold 0.5 --batch-size 4
  python evaluate_extraction_llm_judge.py --pred predictions.json --gold annotations.json \\
      --cache-judgements cache.json --device cuda
"""

import json
import re
import os
import argparse
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from huggingface_hub import login as hf_login
    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Rubric groups
# ══════════════════════════════════════════════════════════════════════════════

# Each field in the schema is assigned to exactly one rubric group.
# The group name is passed to the judge so it applies the correct rubric section.

FIELD_TO_GROUP = {
    # Group 1 — Discrete safety-critical
    "route":                    "discrete",
    "treatment_status":         "discrete",
    "sample_source":            "discrete",
    "contrast":                 "discrete",

    # Group 2 — Numeric / quantitative
    "dose":                     "numeric",
    "duration":                 "numeric",
    "frequency":                "numeric",
    "order_timing":             "numeric",
    "frequency_interval":       "numeric",
    "timing_instructions":      "numeric",

    # Group 3 — Item name matching
    "lab_investigation_name":   "name",
    "monitoring_parameter":     "name",
    "specialty_or_doctor":      "name",
    "medical_name":             "name",
    "imaging_key":              "name",
    "equipment_name":           "name",
    "tissue_sampling_method":   "name",

    # Group 4 — Anatomical / spatial
    "imaging_modality":         "anatomical",
    "imaging_site":             "anatomical",
    "imaging_view":             "anatomical",

    # Group 5 — Dosage form
    "dosage_form":              "dosage_form",

    # Group 6 — Flat list fields (judged as joined string)
    "conservative_method":      "flat_list",
    "lifestyle_habit_modifications": "flat_list",

    # Group 7 — Free text / clinical intent
    "patient_education":        "free_text",
    "when_to_seek_medical_care": "free_text",
    "aim_of_follow_up":         "free_text",
    "aim_of_referral":          "free_text",
    "discontinuation_criteria": "free_text",
    "side_effects_contraindications": "free_text",
    "drug_class":               "free_text",
    "condition_treated":        "free_text",

    # Group 8 — Follow-up scheduled time
    "scheduled_follow_up_time": "followup_time",
}

# Fallback group for any sub-field not explicitly listed above
_DEFAULT_GROUP = "free_text"


def get_group(field_name: str) -> str:
    """Return the rubric group for a given field name."""
    # Strip field prefix (e.g. "medical.dose" → "dose")
    bare = field_name.split(".")[-1]
    return FIELD_TO_GROUP.get(bare, _DEFAULT_GROUP)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — System prompt with full rubric
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a medical expert evaluating clinical information extraction quality.

You will be given:
  - field_group : the type of field being evaluated
  - gold        : the annotator's reference value
  - pred        : the model's predicted value

Your task is to score how well pred matches gold using ONLY these three values:
  1.0 — correct or clinically equivalent
  0.5 — partially correct (see rubric below)
  0.0 — wrong, missing, or hallucinated

══════════════════════════════════════════════════════
GLOBAL NULL CONTRACT  (applies to every field group)
══════════════════════════════════════════════════════
- Both null/empty              → 1.0
- pred null, gold has value    → 0.0  (missed)
- pred has value, gold null    → 0.0  (hallucinated)

══════════════════════════════════════════════════════
GROUP RUBRICS
══════════════════════════════════════════════════════

GROUP: discrete
Fields: route, treatment_status, sample_source, contrast
No 0.5 — the value is either right or wrong.
  1.0 — Clinically identical or standard synonym / abbreviation
        (oral = by mouth = po,  start = initiate,  blood = venous blood,
         no contrast = without contrast)
  0.0 — Different value, wrong meaning, or any null mismatch

GROUP: numeric
Fields: dose, duration, frequency, order_timing, frequency_interval, timing_instructions
  1.0 — Same value, unit-converted equivalent, or standard abbreviation
        (500 mg = 0.5 g,  14 days = two weeks,  tid = three times daily,
         every 8 hours = three times daily)
  0.5 — Core instruction present but incomplete or imprecise:
        • Missing a constraint that appears in gold
          ("initially then 1 tablet after each loose stool" vs
           "2 tablets initially, then 1 tablet after each loose stool,
            not exceeding 8 tablets in 24 hours" — cap is missing)
        • Vague timing that implies the right ballpark but lacks specificity
          ("soon" vs "in 3 weeks")
  0.0 — Wrong value with clinical significance
        (250 mg vs 500 mg,  twice daily vs three times daily,
         10 days vs 14 days)

GROUP: name
Fields: lab_investigation_name, monitoring_parameter, specialty_or_doctor,
        medical drug name, imaging composite key, equipment name,
        tissue_sampling_method
  1.0 — Clinically identical, standard abbreviation, or brand/generic equivalent
        (CBC = complete blood count,  amoxil = amoxicillin,
         rheumatology = rheumatologist,  MRI = magnetic resonance imaging)
  0.5 — Same clinical category or closely related but not equivalent:
        • Less specific than gold
          (blood work vs CBC,  knee imaging vs MRI left knee)
        • Related specialty treating the same condition
          (orthopedic surgeon vs rheumatologist for a joint condition)
        • Same drug class, different molecule
          (amoxicillin-clavulanate vs amoxicillin)
        • Monitoring captures the method but loses the clinical target
          (blood work vs inflammation monitoring via blood work)
  0.0 — Different entity entirely
        (ELISA vs CBC,  cardiologist vs rheumatologist,
         "resources and support programs" vs rheumatologist,
         doxycycline vs amoxicillin)

GROUP: anatomical
Fields: imaging_modality, imaging_site, imaging_view
  1.0 — Exact match or full equivalent
        (left knee = left knee,  MRI = magnetic resonance imaging)
  0.5 — Correct region or modality but missing or adding a specifier:
        • Missing laterality: knee vs left knee
        • Added specification not in gold: MRI with contrast vs MRI
        • Correct modality family, different variant:
          CT without contrast vs CT (contrast unspecified in gold)
  0.0 — Wrong laterality (right knee vs left knee),
        wrong modality (CT vs MRI),  wrong site (spine vs knee)

GROUP: dosage_form
Fields: dosage_form
  1.0 — Same form or abbreviation  (tablets = tab,  capsules = caps)
  0.5 — Different form but same administration route
        (capsules vs tablets — both oral, clinically minor difference)
  0.0 — Different administration route implied
        (injection vs tablets,  inhaler vs tablets)

GROUP: flat_list
Fields: conservative_method, lifestyle_habit_modifications
Context: pred and gold are each a pipe-separated string of all list items.
         Order does NOT matter. Merging multiple gold items into one pred
         item is acceptable if the clinical content is fully preserved.
  1.0 — All key recommendations present and clinically equivalent.
        Minor wording differences, paraphrasing, or merging are fine.
        (healthy diet | reduce alcohol | smoking cessation  vs
         improve your diet by including more vegetables |
         reduce your alcohol intake | stop smoking)
  0.5 — Some recommendations captured but at least one gold item is
        missing OR at least one pred item is hallucinated.
        Use 0.5 whenever the match is partial — regardless of whether
        it is one item missing out of ten or five out of six.
        (drink plenty of water | avoid dairy  vs
         drink plenty of water | avoid dairy |
         maintain a balanced diet | consider quitting smoking)
  0.0 — Completely wrong content, pred is empty when gold is not,
        or pred covers none of the gold recommendations.

GROUP: free_text
Fields: patient_education, when_to_seek_medical_care, aim_of_follow_up,
        aim_of_referral, discontinuation_criteria, condition_treated,
        drug_class, side_effects_contraindications
  1.0 — Same clinical intent and all key information conveyed.
        Paraphrasing, minor truncation, or added detail that does not
        contradict gold are acceptable.
        ("monitor condition closely" vs "to monitor your progress",
         "for further evaluation and management" vs "for further evaluation")
  0.5 — Correct general intent but missing important specifics, OR
        a generic statement where gold is specific.
        ("for routine checkup" vs "for further evaluation",
         "seek care if unwell" vs "seek care if symptoms worsen or
          new neurological symptoms develop")
  0.0 — Wrong clinical intent, contradicts gold, or hallucinated
        when gold is null.

GROUP: followup_time
Fields: scheduled_follow_up_time
  1.0 — Equivalent time specification
        ("in 3 weeks" = "after 3 weeks" = "3 weeks from now",
         "in 3 days if no improvement or sooner if symptoms worsen" ≈
         "in 3 days if you don't see any improvement or sooner if
          your symptoms worsen")
  0.5 — Time implied but not specified, OR correct timeframe but
        conditional logic present in gold is missing from pred.
        ("soon" vs "in 3 weeks",
         "in 3 days" vs "in 3 days if no improvement or sooner if
          symptoms worsen" — timeframe correct but condition missing)
  0.0 — Wrong timeframe (2 weeks vs 3 weeks), or completely vague
        with no usable timing information.

══════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════
Return ONLY a JSON object with a single key:
  {"score": <0 or 0.5 or 1>}

No other text. No explanation. No markdown. Just the JSON object.
Examples of valid outputs:
  {"score": 1}
  {"score": 0.5}
  {"score": 0}
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Prompt builder and output parser
# ══════════════════════════════════════════════════════════════════════════════

def _build_user_prompt(field_group: str,
                       pred: str | None,
                       gold: str | None) -> str:
    pred_str = pred.strip() if pred and pred.strip() else "(empty)"
    gold_str = gold.strip() if gold and gold.strip() else "(empty)"
    return (
        f"field_group: {field_group}\n\n"
        f"gold:\n{gold_str}\n\n"
        f"pred:\n{pred_str}\n\n"
        "JSON:"
    )


def _parse_score(text: str) -> float | None:
    """
    Extract the score from the judge's output.

    Accepts:
      {"score": 1}   {"score": 0.5}   {"score": 0}
      {"score": 1.0} {"score": 0.0}

    Returns 0.0, 0.5, or 1.0, or None if parsing fails.
    """
    # Strip any prompt echo before the last "JSON:"
    if "JSON:" in text:
        text = text[text.rfind("JSON:") + len("JSON:"):]

    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            raw = float(obj.get("score", -1))
            return _snap_score(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Regex fallback
    score_match = re.search(r'"?\bscore\b"?\s*:\s*([0-9.]+)', text)
    if score_match:
        try:
            return _snap_score(float(score_match.group(1)))
        except (ValueError, TypeError):
            pass

    return None


def _snap_score(raw: float) -> float | None:
    """Snap a raw float to the nearest valid score: 0.0, 0.5, or 1.0."""
    if raw < 0:
        return None
    if raw <= 0.25:
        return 0.0
    if raw <= 0.75:
        return 0.5
    if raw <= 1.0:
        return 1.0
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Judge engine
# ══════════════════════════════════════════════════════════════════════════════

class JudgeEngine:
    """
    Loads MedGemma-4b-it once, runs batched rubric-based judgements,
    and caches results.

    Cache key  : (field_group, pred_normalised, gold_normalised)
    Cache value: {"score": float, "parse_ok": bool}

    Trivial cases (null/null, null/value, exact match) are resolved
    without calling the model — they are filled directly into the cache
    during build_cache() so all downstream lookups are uniform.
    """

    def __init__(self,
                 model_id: str = "google/medgemma-4b-it",
                 device: str = "auto",
                 batch_size: int = 4,
                 max_new_tokens: int = 16,
                 hf_token: str | None = None):

        self.model_id       = model_id
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self._cache: dict[tuple, dict] = {}

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if hf_token and _HF_HUB_AVAILABLE:
            hf_login(token=hf_token)
            print(f"[JudgeEngine] Authenticated with HuggingFace Hub.")

        print(f"[JudgeEngine] Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, token=hf_token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[JudgeEngine] Loading model → {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            token=hf_token,
        )
        self.model.eval()
        print("[JudgeEngine] Model ready.")

    # ── cache persistence ──────────────────────────────────────────────────

    def save_cache(self, path: str) -> None:
        serialisable = {json.dumps(list(k)): v for k, v in self._cache.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        print(f"[JudgeEngine] Cache saved → {path}  ({len(self._cache)} entries)")

    def load_cache(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self._cache = {tuple(json.loads(k)): v for k, v in raw.items()}
        print(f"[JudgeEngine] Cache loaded: {len(self._cache)} entries from {path}")

    # ── prompt formatting ──────────────────────────────────────────────────

    def _apply_chat_template(self, user_text: str) -> str:
        """
        Format system + user messages using the model's chat template.
        Falls back to plain concatenation if no template is available.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return (
                f"<|system|>{_SYSTEM_PROMPT}\n"
                f"<|user|>{user_text}\n"
                f"<|assistant|>"
            )

    # ── generation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Run one forward + generate pass over a batch of formatted prompts.
        Returns only the newly generated tokens per prompt (prompt stripped).
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_len  = inputs["input_ids"].shape[1]
        new_tokens = gen[:, input_len:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # ── cache building ─────────────────────────────────────────────────────

    def build_cache(self,
                    triples: list[tuple[str, str | None, str | None]]) -> None:
        """
        Pre-judge all unique (field_group, pred, gold) triples.

        Trivial cases are resolved immediately without model calls.
        Non-trivial cases are batched and judged once.

        Args:
            triples: list of (field_group, pred_text, gold_text)
        """
        # Normalise and deduplicate
        unique = list({
            (fg, _norm(p), _norm(g))
            for fg, p, g in triples
        })

        # Partition into trivial vs non-trivial
        trivial     = [(fg, p, g) for fg, p, g in unique if _is_trivial(p, g)]
        non_trivial = [(fg, p, g) for fg, p, g in unique if not _is_trivial(p, g)]

        # Fill trivial cases immediately
        for fg, p, g in trivial:
            self._cache[(fg, p, g)] = _trivial_result(p, g)

        print(f"[JudgeEngine] {len(unique)} unique triples — "
              f"{len(trivial)} trivial (no model call), "
              f"{len(non_trivial)} to judge  (batch_size={self.batch_size})")

        n_failed = 0
        for i in tqdm(range(0, len(non_trivial), self.batch_size),
                      desc="Judging pairs", unit="batch"):
            batch   = non_trivial[i : i + self.batch_size]
            prompts = [
                self._apply_chat_template(_build_user_prompt(fg, p, g))
                for fg, p, g in batch
            ]
            outputs = self._generate_batch(prompts)

            for (fg, p, g), raw in zip(batch, outputs):
                score = _parse_score(raw)
                if score is None:
                    n_failed += 1
                    # Default to 0.0 on parse failure rather than crashing
                    result = {"score": 0.0, "parse_ok": False, "raw": raw[:120]}
                else:
                    result = {"score": score, "parse_ok": True}
                self._cache[(fg, p, g)] = result

        print(f"[JudgeEngine] Cache built: {len(self._cache)} entries "
              f"({n_failed} parse failures).")

    # ── public interface ───────────────────────────────────────────────────

    def judge(self,
              field_group: str,
              pred: str | None,
              gold: str | None) -> dict:
        """
        Return the cached judgement for a (field_group, pred, gold) triple.
        Falls back to on-the-fly judging for cache misses (e.g. new strings
        after a partial run loaded from cache).

        Returns:
            {"score": float, "parse_ok": bool}
        """
        p   = _norm(pred)
        g   = _norm(gold)
        key = (field_group, p, g)

        if key in self._cache:
            return self._cache[key]

        if _is_trivial(p, g):
            result = _trivial_result(p, g)
        else:
            prompt = self._apply_chat_template(_build_user_prompt(field_group, p, g))
            raw    = self._generate_batch([prompt])[0]
            score  = _parse_score(raw)
            result = ({"score": score, "parse_ok": True} if score is not None
                      else {"score": 0.0, "parse_ok": False, "raw": raw[:120]})

        self._cache[key] = result
        return result

    def score(self,
              field_group: str,
              pred: str | None,
              gold: str | None) -> float:
        """Convenience wrapper — returns just the score float."""
        return self.judge(field_group, pred, gold)["score"]


# ── helpers ────────────────────────────────────────────────────────────────

def _norm(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip().lower()
    return s if s else None


def _is_trivial(p: str | None, g: str | None) -> bool:
    if not p and not g:
        return True
    if not p or not g:
        return True
    if p == g:
        return True
    return False


def _trivial_result(p: str | None, g: str | None) -> dict:
    if not p and not g:
        return {"score": 1.0, "parse_ok": True}
    if not p or not g:
        return {"score": 0.0, "parse_ok": True}
    if p == g:
        return {"score": 1.0, "parse_ok": True}
    return {"score": 0.0, "parse_ok": True}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — String collection  (feeds build_cache)
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_triples(pred_list: list, gold_list: list) -> list[tuple]:
    """
    Walk every extraction in pred and gold, collect all
    (field_group, pred_text, gold_text) triples that need judging.

    For flat list fields the ' | '-joined string is collected.
    For structured lists, all name-pair combinations are collected for
    matching, plus sub-field pairs from heuristically-paired items.
    Confusion detection field strings are also pre-collected.
    """
    triples = set()
    gold_by_id = {item["row_index"]: item for item in gold_list}

    for pred_item in pred_list:
        row_id = pred_item.get("original_row_id")
        if pred_item.get("failed") or row_id not in gold_by_id:
            continue
        gold_item = gold_by_id[row_id]
        _collect_extraction_triples(
            pred_item.get("extraction", {}) or {},
            gold_item.get("extraction", {}) or {},
            triples,
        )

    return list(triples)


def _collect_extraction_triples(pred_ext: dict,
                                 gold_ext: dict,
                                 triples: set) -> None:
    pred_inv = pred_ext.get("investigations", {}) or {}
    gold_inv = gold_ext.get("investigations", {}) or {}
    pred_tx  = pred_ext.get("treatment", {}) or {}
    gold_tx  = gold_ext.get("treatment", {}) or {}

    # ── labs ──────────────────────────────────────────────────────────────
    _collect_list_triples(
        pred_inv.get("labs", []), gold_inv.get("labs", []),
        name_key="lab_investigation_name", field_prefix="labs", triples=triples,
    )

    # ── imaging ───────────────────────────────────────────────────────────
    _collect_list_triples(
        _aug_imaging(pred_inv.get("imaging", [])),
        _aug_imaging(gold_inv.get("imaging", [])),
        name_key="_key", field_prefix="imaging", triples=triples,
    )

    # ── monitoring ────────────────────────────────────────────────────────
    _collect_list_triples(
        pred_ext.get("monitoring", []), gold_ext.get("monitoring", []),
        name_key="monitoring_parameter", field_prefix="monitoring", triples=triples,
    )

    # ── medical ───────────────────────────────────────────────────────────
    _collect_list_triples(
        pred_tx.get("medical", []), gold_tx.get("medical", []),
        name_key="name", field_prefix="medical", triples=triples,
    )

    # ── conservative flat lists ───────────────────────────────────────────
    pred_c = pred_tx.get("conservative") or {}
    gold_c = gold_tx.get("conservative") or {}
    for sub in ("conservative_method", "lifestyle_habit_modifications"):
        triples.add(("flat_list", _join_list(pred_c.get(sub)), _join_list(gold_c.get(sub))))

    # ── follow_up (positional) ────────────────────────────────────────────
    pred_fu = pred_ext.get("follow_up") or []
    gold_fu = gold_ext.get("follow_up") or []
    for i in range(max(len(pred_fu), len(gold_fu))):
        p = pred_fu[i] if i < len(pred_fu) else {}
        g = gold_fu[i] if i < len(gold_fu) else {}
        for key in set(list((p or {}).keys()) + list((g or {}).keys())):
            group = get_group(f"follow_up.{key}")
            triples.add((group, _val_to_str((p or {}).get(key)), _val_to_str((g or {}).get(key))))

    # ── referral ──────────────────────────────────────────────────────────
    _collect_list_triples(
        pred_ext.get("referral", []), gold_ext.get("referral", []),
        name_key="specialty_or_doctor", field_prefix="referral", triples=triples,
    )

    # ── single string fields ──────────────────────────────────────────────
    for field in ("patient_education", "when_to_seek_medical_care"):
        group = get_group(field)
        triples.add((group, _val_to_str(pred_ext.get(field)), _val_to_str(gold_ext.get(field))))

    # ── confusion detection pre-collection ───────────────────────────────
    _collect_confusion_triples(pred_ext, gold_ext, triples)


def _collect_list_triples(pred_list, gold_list,
                           name_key, field_prefix, triples) -> None:
    """
    Collect name-pair combinations (for matching) and sub-field pairs
    (from heuristically-matched items) for a structured list field.
    """
    pred_list = [p for p in (pred_list or []) if p]
    gold_list = [g for g in (gold_list or []) if g]

    pred_names = [_val_to_str(p.get(name_key) if isinstance(p, dict) else p) for p in pred_list]
    gold_names = [_val_to_str(g.get(name_key) if isinstance(g, dict) else g) for g in gold_list]

    # All name combinations needed for the greedy matching matrix
    name_group = get_group(f"{field_prefix}.{name_key}")
    for pn in pred_names:
        for gn in gold_names:
            triples.add((name_group, pn, gn))

    # Heuristic sub-field collection
    matched_gold = set()
    for pred_item in pred_list:
        if not isinstance(pred_item, dict):
            continue
        pn         = _val_to_str(pred_item.get(name_key, "")).lower()
        best_gi    = max(
            ((gi, _token_overlap(pn, _val_to_str(g.get(name_key, "")).lower()))
             for gi, g in enumerate(gold_list)
             if gi not in matched_gold and isinstance(g, dict)),
            key=lambda x: x[1],
            default=(None, -1),
        )
        gi, best_score = best_gi
        if gi is not None and best_score > 0:
            matched_gold.add(gi)
            gold_item = gold_list[gi]
            all_keys  = set(list(pred_item.keys()) + list(gold_item.keys()))
            for key in all_keys:
                if key == name_key:
                    continue
                group = get_group(f"{field_prefix}.{key}")
                triples.add((group, _val_to_str(pred_item.get(key)), _val_to_str(gold_item.get(key))))

    # Unmatched gold items → sub-fields vs None
    for gi, gold_item in enumerate(gold_list):
        if gi in matched_gold or not isinstance(gold_item, dict):
            continue
        for key, val in gold_item.items():
            if key == name_key:
                continue
            group = get_group(f"{field_prefix}.{key}")
            triples.add((group, None, _val_to_str(val)))


def _collect_confusion_triples(pred_ext: dict,
                                gold_ext: dict,
                                triples: set) -> None:
    """Pre-collect field strings needed for confusion detection."""
    for field_name in _CONFUSION_FIELD_NAMES:
        pred_text = _extract_field_text(field_name, pred_ext)
        gold_text = _extract_field_text(field_name, gold_ext)
        if pred_text or gold_text:
            triples.add(("name", pred_text, gold_text))
            triples.add(("name", gold_text, pred_text))


def _aug_imaging(items):
    out = []
    for item in (items or []):
        item = dict(item)
        item["_key"] = f"{item.get('imaging_modality','')} {item.get('site','')}".strip()
        out.append(item)
    return out


# ── string helpers ─────────────────────────────────────────────────────────

def _join_list(items) -> str | None:
    cleaned = [str(x).strip() for x in (items or []) if str(x).strip()]
    return " | ".join(cleaned) if cleaned else None


def _val_to_str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, list):
        cleaned = [str(v).strip() for v in val if str(v).strip()]
        return " | ".join(cleaned) if cleaned else None
    s = str(val).strip()
    return s if s else None


def _token_overlap(a: str, b: str) -> float:
    ta = set((a or "").lower().split())
    tb = set((b or "").lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def avg(values) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def score_subfields(pred_item: dict | None,
                    gold_item: dict,
                    field_prefix: str,
                    engine: JudgeEngine,
                    skip_keys: set = None) -> dict:
    """
    Judge each sub-field of a matched item pair using the appropriate rubric group.
    pred_item = None when the gold item was unmatched (model missed it entirely).
    """
    skip_keys = skip_keys or set()
    scores    = {}

    if isinstance(pred_item, str):
        pred_item = {"name": pred_item}
    if isinstance(gold_item, str):
        gold_item = {"name": gold_item}

    for key, gold_val in gold_item.items():
        if key in skip_keys:
            continue
        pred_val  = pred_item.get(key) if pred_item else None
        pred_str  = _val_to_str(pred_val)
        gold_str  = _val_to_str(gold_val)
        group     = get_group(f"{field_prefix}.{key}")
        scores[key] = engine.judge(group, pred_str, gold_str)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Structured item list matcher  (greedy judge-score matching)
# ══════════════════════════════════════════════════════════════════════════════

def match_and_score_list(pred_list: list,
                         gold_list: list,
                         name_key: str,
                         field_prefix: str,
                         engine: JudgeEngine,
                         skip_keys: set = None,
                         match_threshold: float = 0.5) -> dict:
    """
    Match pred items to gold items by best judge score on name_key,
    then judge sub-fields of matched pairs with the appropriate rubric group.

    match_threshold: judge score below this → items treated as unrelated.
    """
    skip_keys = skip_keys or set()
    pred_list = [{"name": p} if isinstance(p, str) else (p or {}) for p in (pred_list or [])]
    gold_list = [{"name": g} if isinstance(g, str) else (g or {}) for g in (gold_list or [])]

    if not gold_list and not pred_list:
        return {"per_item": [], "unmatched_gold": [], "unmatched_pred": [],
                "field_avg": 1.0}
    if not gold_list:
        return {"per_item": [], "unmatched_gold": [],
                "unmatched_pred": [str(p.get(name_key, "")) for p in pred_list],
                "field_avg": 0.0}

    name_group = get_group(f"{field_prefix}.{name_key}")
    gold_names = [_val_to_str(g.get(name_key, "")) for g in gold_list]
    pred_names = [_val_to_str(p.get(name_key, "")) for p in pred_list]

    # Build judge-score matrix on item names
    sim_matrix = [
        [engine.score(name_group, pred_names[j], gold_names[i])
         for j in range(len(pred_list))]
        for i in range(len(gold_list))
    ]

    # Greedy matching
    candidates = [
        (sim_matrix[i][j], i, j)
        for i in range(len(gold_list))
        for j in range(len(pred_list))
        if sim_matrix[i][j] >= match_threshold
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gold = {}
    matched_pred = {}
    for score, gi, pi in candidates:
        if gi not in matched_gold and pi not in matched_pred:
            matched_gold[gi] = pi
            matched_pred[pi] = gi

    all_item_avgs    = []
    per_item_results = []

    for gi, gold_item in enumerate(gold_list):
        pi         = matched_gold.get(gi)
        pred_item  = pred_list[pi] if pi is not None else None
        name_score = sim_matrix[gi][pi] if pi is not None else 0.0

        sub_scores = score_subfields(
            pred_item, gold_item,
            field_prefix=field_prefix,
            engine=engine,
            skip_keys={name_key} | skip_keys,
        )
        item_avg = avg([v["score"] for v in sub_scores.values()])
        all_item_avgs.append(item_avg)

        per_item_results.append({
            "name":            gold_item.get(name_key),
            "matched_to":      pred_list[pi].get(name_key) if pred_item else None,
            "name_score":      round(name_score, 4),
            "matched":         pred_item is not None,
            "subfield_scores": {k: v["score"] for k, v in sub_scores.items()},
            "subfield_details": sub_scores,
            "item_avg":        item_avg,
        })

    # Penalise hallucinated pred items
    unmatched_pred = [pred_list[j] for j in range(len(pred_list)) if j not in matched_pred]
    for _ in unmatched_pred:
        all_item_avgs.append(0.0)

    unmatched_gold = [gold_list[i] for i in range(len(gold_list)) if i not in matched_gold]

    return {
        "per_item":       per_item_results,
        "unmatched_gold": [g.get(name_key) for g in unmatched_gold],
        "unmatched_pred": [p.get(name_key) for p in unmatched_pred],
        "field_avg":      avg(all_item_avgs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Per-section scorers
# ══════════════════════════════════════════════════════════════════════════════

def score_labs(pred_inv: dict, gold_inv: dict,
               engine: JudgeEngine, match_threshold: float) -> dict:
    return match_and_score_list(
        pred_inv.get("labs", []), gold_inv.get("labs", []),
        name_key="lab_investigation_name",
        field_prefix="labs",
        engine=engine, match_threshold=match_threshold,
    )


def score_imaging(pred_inv: dict, gold_inv: dict,
                  engine: JudgeEngine, match_threshold: float) -> dict:
    return match_and_score_list(
        _aug_imaging(pred_inv.get("imaging", [])),
        _aug_imaging(gold_inv.get("imaging", [])),
        name_key="_key",
        field_prefix="imaging",
        engine=engine, match_threshold=match_threshold,
    )


def score_monitoring(pred_mon: list, gold_mon: list,
                     engine: JudgeEngine, match_threshold: float) -> dict:
    return match_and_score_list(
        pred_mon or [], gold_mon or [],
        name_key="monitoring_parameter",
        field_prefix="monitoring",
        engine=engine, match_threshold=match_threshold,
    )


def score_medical(pred_tx: dict, gold_tx: dict,
                  engine: JudgeEngine, match_threshold: float) -> dict:
    return match_and_score_list(
        pred_tx.get("medical", []), gold_tx.get("medical", []),
        name_key="name",
        field_prefix="medical",
        engine=engine, match_threshold=match_threshold,
    )


def score_conservative(pred_tx: dict, gold_tx: dict,
                        engine: JudgeEngine) -> dict:
    """
    Score conservative treatment flat list fields.
    Lists are joined with ' | ' and judged as a single unit using the
    flat_list rubric group.
    """
    pred_c = pred_tx.get("conservative") or {}
    gold_c = gold_tx.get("conservative") or {}

    method_result    = engine.judge(
        "flat_list",
        _join_list(pred_c.get("conservative_method")),
        _join_list(gold_c.get("conservative_method")),
    )
    lifestyle_result = engine.judge(
        "flat_list",
        _join_list(pred_c.get("lifestyle_habit_modifications")),
        _join_list(gold_c.get("lifestyle_habit_modifications")),
    )

    field_avg = avg([method_result["score"], lifestyle_result["score"]])

    return {
        "conservative_method":           method_result,
        "lifestyle_habit_modifications": lifestyle_result,
        "field_avg":                     field_avg,
    }


def score_follow_up(pred_fu: list, gold_fu: list,
                    engine: JudgeEngine) -> dict:
    """
    Positional matching — index i in pred matches index i in gold.
    Each sub-field judged with the appropriate rubric group.
    """
    pred_fu = pred_fu or []
    gold_fu = gold_fu or []

    if not gold_fu and not pred_fu:
        return {"per_item": [], "field_avg": 1.0}

    max_len  = max(len(pred_fu), len(gold_fu))
    per_item = []
    all_avgs = []

    for i in range(max_len):
        p = pred_fu[i] if i < len(pred_fu) else {}
        g = gold_fu[i] if i < len(gold_fu) else {}
        sub = score_subfields(p, g, field_prefix="follow_up", engine=engine)
        item_avg = avg([v["score"] for v in sub.values()])
        all_avgs.append(item_avg)
        per_item.append({
            "subfield_scores":  {k: v["score"] for k, v in sub.items()},
            "subfield_details": sub,
            "item_avg":         item_avg,
        })

    return {"per_item": per_item, "field_avg": avg(all_avgs)}


def score_referral(pred_ref: list, gold_ref: list,
                   engine: JudgeEngine, match_threshold: float) -> dict:
    return match_and_score_list(
        pred_ref or [], gold_ref or [],
        name_key="specialty_or_doctor",
        field_prefix="referral",
        engine=engine, match_threshold=match_threshold,
    )


def score_single(field_name: str, pred_val, gold_val,
                 engine: JudgeEngine) -> dict:
    """Judge a single flat string field."""
    group  = get_group(field_name)
    result = engine.judge(group, _val_to_str(pred_val), _val_to_str(gold_val))
    return {"field_avg": result["score"], "detail": result}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Confusion detection
# ══════════════════════════════════════════════════════════════════════════════

_CONFUSION_FIELD_NAMES = {
    "patient_education", "when_to_seek_medical_care",
    "conservative_method", "lifestyle_habit_modifications",
    "prevention", "complementary_therapies",
    "labs", "imaging", "imaging_modality", "imaging_site",
    "tissue_sampling", "monitoring",
    "medical_names", "medical_routes", "medical_dosage_forms",
    "external_equipment", "referral", "follow_up",
}


def _extract_field_text(field_name: str, extraction: dict) -> str | None:
    """Extract all text from a named field into a single string."""
    inv  = extraction.get("investigations", {}) or {}
    tx   = extraction.get("treatment", {}) or {}
    cons = tx.get("conservative") or {}

    field_map = {
        "patient_education":             extraction.get("patient_education"),
        "when_to_seek_medical_care":     extraction.get("when_to_seek_medical_care"),
        "conservative_method":           cons.get("conservative_method") or [],
        "lifestyle_habit_modifications": cons.get("lifestyle_habit_modifications") or [],
        "prevention":                    tx.get("prevention") or [],
        "complementary_therapies":       tx.get("complementary_therapies") or [],
        "labs":     [i.get("lab_investigation_name", "") for i in (inv.get("labs") or [])],
        "imaging":  [f"{i.get('imaging_modality','')} {i.get('site','')}".strip()
                     for i in (inv.get("imaging") or [])],
        "imaging_modality": [i.get("imaging_modality", "") for i in (inv.get("imaging") or [])],
        "imaging_site":     [i.get("site", "")             for i in (inv.get("imaging") or [])],
        "tissue_sampling":  [i.get("tissue_sampling_method", i.get("name", ""))
                             for i in (inv.get("tissue_sampling") or [])],
        "monitoring":       [i.get("monitoring_parameter", "") for i in (extraction.get("monitoring") or [])],
        "medical_names":    [i.get("name", "") for i in (tx.get("medical") or [])],
        "medical_routes":   [i.get("route", "") for i in (tx.get("medical") or []) if i.get("route")],
        "medical_dosage_forms": [i.get("dosage_form", "") for i in (tx.get("medical") or [])
                                 if i.get("dosage_form")],
        "external_equipment": [i.get("name", i.get("equipment_name", ""))
                               for i in (tx.get("external_equipment") or [])],
        "referral":  [f"{i.get('specialty_or_doctor','')} {i.get('aim_of_referral','')}".strip()
                      for i in (extraction.get("referral") or [])],
        "follow_up": [f"{i.get('scheduled_follow_up_time','')} {i.get('aim_of_follow_up','')}".strip()
                      for i in (extraction.get("follow_up") or [])],
    }

    content = field_map.get(field_name)
    if content is None:
        return None
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        joined = " ".join(str(x).strip() for x in content if x).strip()
        return joined or None
    return None


CONFUSION_PAIRS = [
    ("medical_routes",          "medical_dosage_forms",          "medical: route ↔ dosage_form"),
    ("imaging_modality",        "imaging_site",                  "imaging: modality ↔ site"),
    ("follow_up",               "referral",                      "follow_up ↔ referral"),
    ("labs",                    "monitoring",                    "labs ↔ monitoring"),
    ("imaging",                 "monitoring",                    "imaging ↔ monitoring"),
    ("prevention",              "patient_education",             "prevention ↔ patient_education"),
    ("conservative_method",     "medical_names",                 "conservative_method ↔ medical"),
    ("medical_names",           "external_equipment",            "medical ↔ external_equipment"),
    ("imaging",                 "labs",                          "imaging ↔ labs"),
    ("tissue_sampling",         "imaging",                       "tissue_sampling ↔ imaging"),
    ("imaging",                 "referral",                      "imaging ↔ referral"),
    ("labs",                    "referral",                      "labs ↔ referral"),
    ("prevention",              "conservative_method",           "prevention ↔ conservative_method"),
    ("prevention",              "lifestyle_habit_modifications",  "prevention ↔ lifestyle"),
    ("prevention",              "complementary_therapies",       "prevention ↔ complementary_therapies"),
    ("conservative_method",     "lifestyle_habit_modifications",  "conservative_method ↔ lifestyle"),
    ("conservative_method",     "complementary_therapies",       "conservative_method ↔ complementary_therapies"),
    ("lifestyle_habit_modifications", "complementary_therapies", "lifestyle ↔ complementary_therapies"),
]


def detect_confusion(pred_extraction: dict, gold_extraction: dict,
                     engine: JudgeEngine,
                     threshold: float = 0.5) -> dict:
    """Run all confusion pair checks using the judge model."""
    pair_results, flagged = [], []

    for field_a, field_b, description in CONFUSION_PAIRS:
        gold_a = _extract_field_text(field_a, gold_extraction)
        pred_b = _extract_field_text(field_b, pred_extraction)
        gold_b = _extract_field_text(field_b, gold_extraction)
        pred_a = _extract_field_text(field_a, pred_extraction)

        # A→B: does gold's field_A content appear in pred's field_B?
        score_a_in_b = engine.score("name", gold_a, pred_b) if gold_a and pred_b else 0.0
        # B→A: does gold's field_B content appear in pred's field_A?
        score_b_in_a = engine.score("name", gold_b, pred_a) if gold_b and pred_a else 0.0

        is_flagged = score_a_in_b >= threshold or score_b_in_a >= threshold
        result = {
            "pair":    description,
            "field_a": field_a, "field_b": field_b,
            "a_in_b":  round(score_a_in_b, 4),
            "b_in_a":  round(score_b_in_a, 4),
            "flagged": is_flagged,
        }
        pair_results.append(result)
        if is_flagged:
            flagged.append(description)

    return {"pairs": pair_results, "flagged": flagged, "any_flagged": len(flagged) > 0}


def aggregate_confusion(case_confusions: list, case_ids: list) -> dict:
    acc = defaultdict(lambda: {"a_in_b": [], "b_in_a": [], "flagged_count": 0,
                                "total": 0, "flagged_row_ids": []})
    for cc, row_id in zip(case_confusions, case_ids):
        for pr in cc["pairs"]:
            d = pr["pair"]
            acc[d]["a_in_b"].append(pr["a_in_b"])
            acc[d]["b_in_a"].append(pr["b_in_a"])
            acc[d]["total"] += 1
            if pr["flagged"]:
                acc[d]["flagged_count"] += 1
                acc[d]["flagged_row_ids"].append(row_id)

    summary = {}
    for desc, data in acc.items():
        total = data["total"]
        summary[desc] = {
            "flagged_cases":   data["flagged_count"],
            "total_cases":     total,
            "flagged_pct":     round(data["flagged_count"] / total * 100, 1) if total else 0.0,
            "avg_a_in_b":      avg(data["a_in_b"]),
            "avg_b_in_a":      avg(data["b_in_a"]),
            "flagged_row_ids": data["flagged_row_ids"],
        }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Case-level and dataset-level evaluation
# ══════════════════════════════════════════════════════════════════════════════

def score_case(pred_extraction: dict, gold_extraction: dict,
               engine: JudgeEngine,
               match_threshold: float = 0.5,
               run_confusion: bool = False,
               confusion_threshold: float = 0.5) -> dict:
    """Score one case across all extraction fields using the judge model."""
    pred_inv = pred_extraction.get("investigations", {}) or {}
    gold_inv = gold_extraction.get("investigations", {}) or {}
    pred_tx  = pred_extraction.get("treatment", {}) or {}
    gold_tx  = gold_extraction.get("treatment", {}) or {}

    fields = {
        "labs":       score_labs(pred_inv, gold_inv, engine, match_threshold),
        "imaging":    score_imaging(pred_inv, gold_inv, engine, match_threshold),
        "monitoring": score_monitoring(
                          pred_extraction.get("monitoring"),
                          gold_extraction.get("monitoring"),
                          engine, match_threshold),
        "medical":    score_medical(pred_tx, gold_tx, engine, match_threshold),
        "conservative": score_conservative(pred_tx, gold_tx, engine),
        "follow_up":  score_follow_up(
                          pred_extraction.get("follow_up"),
                          gold_extraction.get("follow_up"),
                          engine),
        "referral":   score_referral(
                          pred_extraction.get("referral"),
                          gold_extraction.get("referral"),
                          engine, match_threshold),
        "patient_education": score_single(
                          "patient_education",
                          pred_extraction.get("patient_education"),
                          gold_extraction.get("patient_education"),
                          engine),
        "when_to_seek_medical_care": score_single(
                          "when_to_seek_medical_care",
                          pred_extraction.get("when_to_seek_medical_care"),
                          gold_extraction.get("when_to_seek_medical_care"),
                          engine),
    }

    field_scores = {name: r["field_avg"] for name, r in fields.items()}
    overall      = avg(field_scores.values())

    result = {
        "field_scores":  field_scores,
        "field_details": fields,
        "overall":       overall,
    }

    if run_confusion:
        result["confusion"] = detect_confusion(
            pred_extraction, gold_extraction,
            engine=engine, threshold=confusion_threshold,
        )

    return result


def evaluate_dataset(pred_list: list, gold_list: list,
                     engine: JudgeEngine,
                     match_threshold: float = 0.5,
                     run_confusion: bool = False,
                     confusion_threshold: float = 0.5) -> dict:
    """Evaluate the full dataset."""
    gold_by_id        = {item["row_index"]: item for item in gold_list}
    case_results      = []
    field_accumulator = defaultdict(list)
    case_confusions   = []

    for pred_item in tqdm(pred_list, desc="Evaluating cases", unit="case"):
        row_id = pred_item["original_row_id"]

        if pred_item.get("failed"):
            print(f"[SKIP] row_id={row_id} marked as failed.")
            continue
        if row_id not in gold_by_id:
            print(f"[WARN] No gold for row_id={row_id}, skipping.")
            continue

        gold_item = gold_by_id[row_id]
        result    = score_case(
            pred_item["extraction"], gold_item["extraction"],
            engine=engine,
            match_threshold=match_threshold,
            run_confusion=run_confusion,
            confusion_threshold=confusion_threshold,
        )
        result["row_id"] = row_id
        case_results.append(result)

        for field, score in result["field_scores"].items():
            field_accumulator[field].append(score)

        if run_confusion and "confusion" in result:
            case_confusions.append(result["confusion"])

    dataset_field_scores = {f: avg(s) for f, s in field_accumulator.items()}
    dataset_overall      = avg(dataset_field_scores.values())

    output = {
        "dataset_field_scores": dataset_field_scores,
        "dataset_overall":      dataset_overall,
        "per_case":             case_results,
    }

    if run_confusion and case_confusions:
        case_ids = [c["row_id"] for c in case_results if "confusion" in c]
        output["confusion_summary"] = aggregate_confusion(case_confusions, case_ids)

    return output


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Reporting
# ══════════════════════════════════════════════════════════════════════════════

_CONSERVATIVE_SUBFIELDS = ("conservative_method", "lifestyle_habit_modifications")


def print_report(results: dict,
                 model_id: str,
                 run_confusion: bool = False,
                 confusion_threshold: float = 0.5,
                 match_threshold: float = 0.5) -> None:

    W = 72

    print()
    print("=" * W)
    print("  MEDICAL EXTRACTION EVALUATION REPORT  (LLM-as-Judge / Rubric)")
    print("=" * W)
    print(f"  Cases evaluated  : {len(results['per_case'])}")
    print(f"  Judge model      : {model_id}")
    print(f"  Scoring scale    : 0.0 / 0.5 / 1.0  (rubric-based)")
    print(f"  Match threshold  : {match_threshold}")
    if run_confusion:
        print(f"  Confusion detect : ENABLED  (threshold={confusion_threshold})")
    print("=" * W)

    # ── Dataset field scores ───────────────────────────────────────────────
    print(f"\n  {'Field':<35} {'Score':>8}")
    print(f"  {'─'*35} {'─'*8}")
    for field, score in results["dataset_field_scores"].items():
        print(f"  {field:<35} {score:>8.4f}")
    print(f"  {'─'*35} {'─'*8}")
    print(f"  {'OVERALL':<35} {results['dataset_overall']:>8.4f}")
    print()

    # ── Conservative breakdown ─────────────────────────────────────────────
    print("─" * W)
    print("  CONSERVATIVE TREATMENT  (flat_list rubric group)")
    print("─" * W)
    for case in results["per_case"]:
        row_id = case["row_id"]
        cons   = case["field_details"].get("conservative", {})
        print(f"\n  Case row_id={row_id}")
        for sub in _CONSERVATIVE_SUBFIELDS:
            sub_result = cons.get(sub, {})
            score      = sub_result.get("score", "—")
            parse_ok   = sub_result.get("parse_ok", True)
            label      = sub.replace("_", " ")
            flag       = "" if parse_ok else "  [parse fail]"
            print(f"    {label:<40} Score={score}{flag}")

    # ── Per-case summary table ─────────────────────────────────────────────
    print()
    print("─" * W)
    print("  PER-CASE SUMMARY")
    print("─" * W)

    fields_order = list(results["dataset_field_scores"].keys())
    header = f"  {'Row ID':<8} {'Overall':>8}  " + \
             "  ".join(f"{f[:7]:>7}" for f in fields_order)
    print(header)
    print(f"  {'─'*8} {'─'*8}  " + "  ".join("─" * 7 for _ in fields_order))

    for case in results["per_case"]:
        field_vals = "  ".join(
            f"{case['field_scores'].get(f, 0.0):>7.4f}" for f in fields_order
        )
        print(f"  {case['row_id']:<8} {case['overall']:>8.4f}  {field_vals}")

    # ── Parse failure summary ──────────────────────────────────────────────
    n_failed = sum(
        1
        for case in results["per_case"]
        for field_result in case.get("field_details", {}).values()
        for detail in _iter_details(field_result)
        if not detail.get("parse_ok", True)
    )
    if n_failed:
        print()
        print(f"  [WARN] {n_failed} judgement(s) failed to parse — "
              f"defaulted to 0.0. Consider increasing --max-new-tokens.")

    # ── Confusion report ───────────────────────────────────────────────────
    if run_confusion and "confusion_summary" in results:
        print()
        print("=" * W)
        print(f"  FIELD CONFUSION REPORT  (judge threshold={confusion_threshold})")
        print("=" * W)
        print(f"  {'Confusion Pair':<48} {'Flag%':>6}  {'A→B':>6}  {'B→A':>6}  {'Cases':>6}")
        print(f"  {'─'*48} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

        summary = results["confusion_summary"]
        for desc, data in sorted(summary.items(),
                                  key=lambda x: x[1]["flagged_pct"], reverse=True):
            marker  = "  !" if data["flagged_cases"] > 0 else "   "
            row_ids = data.get("flagged_row_ids", [])
            print(f"{marker} {desc:<48} {data['flagged_pct']:>5.1f}%  "
                  f"{data['avg_a_in_b']:>6.4f}  {data['avg_b_in_a']:>6.4f}  "
                  f"{data['flagged_cases']:>3}/{data['total_cases']:<3}")
            if row_ids:
                print(f"       row_ids: [{', '.join(str(r) for r in row_ids)}]")

        print()
        print("  PER-CASE FLAGS")
        print(f"  {'─'*48}")
        for case in results["per_case"]:
            if not case.get("confusion", {}).get("any_flagged"):
                continue
            print(f"  row_id={case['row_id']}")
            for f in case["confusion"]["flagged"]:
                print(f"    ! {f}")

    print()
    print("=" * W)
    print()


def _iter_details(field_result: dict):
    """Yield all leaf judgement dicts from a field result for parse-fail counting."""
    if "detail" in field_result:
        yield field_result["detail"]
    if "conservative_method" in field_result:
        yield field_result["conservative_method"]
    if "lifestyle_habit_modifications" in field_result:
        yield field_result["lifestyle_habit_modifications"]
    for item in field_result.get("per_item", []):
        for v in item.get("subfield_details", {}).values():
            yield v


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate medical extraction using LLM-as-judge with rubric-based 0/0.5/1 scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_extraction_llm_judge.py --pred pred.json --gold gold.json
  python evaluate_extraction_llm_judge.py --pred pred.json --gold gold.json \\
      --output results.json
  python evaluate_extraction_llm_judge.py --pred pred.json --gold gold.json \\
      --confusion --confusion-threshold 0.5 --batch-size 4
  python evaluate_extraction_llm_judge.py --pred pred.json --gold gold.json \\
      --cache-judgements cache.json --device cuda
  python evaluate_extraction_llm_judge.py --pred pred.json --gold gold.json \\
      --max-new-tokens 32 --match-threshold 0.5
        """,
    )
    parser.add_argument("--pred",                  required=True)
    parser.add_argument("--gold",                  required=True)
    parser.add_argument("--output",                default=None)
    parser.add_argument("--model-id",              default="google/medgemma-4b-it")
    parser.add_argument("--batch-size",            type=int, default=4)
    parser.add_argument("--max-new-tokens",        type=int, default=16,
                        help="Tokens to generate per judgement (default: 16). "
                             "Output is just '{\"score\": X}' so 16 is sufficient.")
    parser.add_argument("--device",               default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--match-threshold",       type=float, default=0.5)
    parser.add_argument("--confusion",             action="store_true")
    parser.add_argument("--confusion-threshold",   type=float, default=0.5)
    parser.add_argument("--cache-judgements",      default=None,
                        help="Path to save/load judgement cache (.json). "
                             "If file exists, loads it; otherwise judges and saves.")
    parser.add_argument("--hf-token",             default=None,
                        help="HuggingFace token for gated repos. "
                             "Falls back to HF_TOKEN env variable.")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    print(f"\nLoading predictions  : {args.pred}")
    print(f"Loading gold         : {args.gold}")
    print(f"Judge model          : {args.model_id}")
    print(f"Device               : {args.device}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Max new tokens       : {args.max_new_tokens}")
    print(f"Match threshold      : {args.match_threshold}")
    if args.confusion:
        print(f"Confusion detection  : enabled (threshold={args.confusion_threshold})")

    with open(args.pred) as f:
        pred_data = json.load(f)
    with open(args.gold) as f:
        gold_data = json.load(f)

    engine = JudgeEngine(
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        hf_token=hf_token,
    )

    cache_exists = args.cache_judgements and os.path.exists(args.cache_judgements)

    if cache_exists:
        print(f"\nLoading judgement cache from: {args.cache_judgements}")
        engine.load_cache(args.cache_judgements)
    else:
        print("\nCollecting all (field_group, pred, gold) triples ...")
        triples = collect_all_triples(pred_data, gold_data)
        print(f"Unique triples: {len(triples)}")
        engine.build_cache(triples)
        if args.cache_judgements:
            engine.save_cache(args.cache_judgements)

    results = evaluate_dataset(
        pred_data, gold_data,
        engine=engine,
        match_threshold=args.match_threshold,
        run_confusion=args.confusion,
        confusion_threshold=args.confusion_threshold,
    )

    print_report(
        results,
        model_id=args.model_id,
        run_confusion=args.confusion,
        confusion_threshold=args.confusion_threshold,
        match_threshold=args.match_threshold,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full results saved → {args.output}\n")


if __name__ == "__main__":
    main()


"""
python evaluate_extraction_llm_judge.py \\
    --pred extracted_plan_google_gemini-2.0-flash-001_pydantic_normalized.json \\
    --gold extractions_annotator_2_normalized.json \\
    --output annotator_2_gemini_2_flash_llm_judge_eval_results.json \\
    --model-id google/medgemma-4b-it \\
    --batch-size 4 \\
    --device cuda \\
    --match-threshold 0.5 \\
    --confusion \\
    --confusion-threshold 0.6 \\
    --cache-judgements medgemma_judge_cache.json
"""