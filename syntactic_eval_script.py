"""
Medical Treatment Plan Extraction — Evaluation Script
======================================================
Compares model extraction output against human annotator (gold) annotations.

─────────────────────────────────────────────────────────────────────────────
SCORING STRATEGY BY FIELD TYPE
─────────────────────────────────────────────────────────────────────────────

  Structured item lists  (labs, imaging, medications, monitoring, referrals)
  ──────────────────────
    • Items matched by name using greedy chrF similarity (fuzzy, not exact)
    • Each matched pair → sub-fields scored individually with chrF
    • Unmatched gold items (missed) → 0
    • Unmatched pred items (hallucinated) → 0
    • field_avg = avg(all item scores including unmatched)

  Flat string lists  (conservative_method, lifestyle_habit_modifications)
  ─────────────────
    • Scored with Coverage F1 (the new metric developed for this task)
    • Handles merges, splits, reordering without penalizing granularity
    • Uses LCS token coverage with grammatical glue removal
    • Reports precision, recall, F1 separately

  Single string fields  (patient_education, when_to_seek_medical_care,
                         follow_up sub-fields, referral aim)
  ─────────────────────
    • Scored with chrF (character n-gram F-score)

─────────────────────────────────────────────────────────────────────────────
COVERAGE F1 METRIC  (for flat string lists)
─────────────────────────────────────────────────────────────────────────────

  gold_ref = concatenation of all gold items
  pred_ref = concatenation of all pred items

  Precision = avg( token_coverage(p, gold_ref) for p in pred_list )
  Recall    = avg( token_coverage(g, pred_ref) for g in gold_list )
  F1        = 2 * P * R / (P + R)

  token_coverage(query, ref) = LCS(query_tokens, ref_tokens) / len(query_tokens)

  Grammatical glue removed before scoring (articles, prepositions, auxiliaries,
  pronouns, conjunctions) while preserving negations (not, no, never, without),
  quantities (plenty, very), frequencies (regularly, daily), and modals.

─────────────────────────────────────────────────────────────────────────────
CONFUSION DETECTION  (optional, enabled with --confusion)
─────────────────────────────────────────────────────────────────────────────

  For each confusion pair (field_A, field_B), after normal evaluation:
    1. Extract content tokens from gold's field_A
    2. Measure their coverage against pred's field_B  (and vice versa)
    3. A confusion is flagged when coverage > --confusion-threshold

  Confusion pairs checked:
    • medical.route         ↔ medical.dosage_form      (intra-item swap)
    • imaging.modality      ↔ imaging.site              (intra-item swap)
    • follow_up             ↔ referral
    • labs                  ↔ monitoring
    • imaging               ↔ monitoring
    • prevention            ↔ patient_education
    • conservative_method   ↔ medical (names)
    • medical               ↔ external_equipment
    • imaging               ↔ labs
    • tissue_sampling       ↔ imaging
    • imaging               ↔ referral
    • labs                  ↔ referral
    • prevention ↔ conservative_method ↔ lifestyle ↔ complementary_therapies

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

  python evaluate_extraction.py --pred predictions.json --gold annotations.json
  python evaluate_extraction.py --pred predictions.json --gold annotations.json \\
      --output results.json --no-stopwords
  python evaluate_extraction.py --pred predictions.json --gold annotations.json \\
      --confusion --confusion-threshold 0.5
"""

import json
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from sacrebleu.metrics import CHRF


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — chrF scorer  (used for structured sub-fields and single strings)
# ══════════════════════════════════════════════════════════════════════════════

_chrf = CHRF()


def chrf_score(pred: str | None, gold: str | None) -> float:
    """
    Character n-gram F-score between two strings, normalised to [0, 1].

    Both None/empty  → 1.0  (both agree nothing is here)
    One side missing → 0.0
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pred = str(pred).strip().lower()
    gold = str(gold).strip().lower()
    if pred == gold:
        return 1.0
    return round(_chrf.sentence_score(pred, [gold]).score / 100.0, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Coverage F1 metric  (used for flat string lists)
# ══════════════════════════════════════════════════════════════════════════════

GRAMMATICAL_GLUE = {
    # Articles
    "a", "an", "the",
    # Coordinating conjunctions that cause merge/split artifacts
    "and", "but",
    # Prepositions
    "of", "in", "on", "at", "by", "from", "with", "to", "for", "as",
    # Relative pronouns
    "that", "which", "who", "whom",
    # Personal pronouns
    "you", "your", "we", "our", "they", "their",
    "it", "its", "i", "my", "he", "she", "his", "her",
    # Demonstratives
    "this", "these", "those",
    # Contraction artifacts (it's → [it, s], don't → [don, t])
    "s", "t",
}


def _filter_glue(tokens: list) -> list:
    """Remove grammatical glue tokens, preserve semantically important words."""
    return [t for t in tokens if t not in GRAMMATICAL_GLUE]


def token_coverage(query: str, ref: str, use_stopwords: bool = True) -> float:
    """
    Fraction of query's tokens found in ref as a subsequence (via LCS).

    Args:
        query         : string to measure coverage of
        ref           : reference pool to search within
        use_stopwords : if True, remove grammatical glue before measuring

    Returns:
        float in [0.0, 1.0]
    """
    q_tokens = query.lower().split()
    r_tokens = ref.lower().split()

    if use_stopwords:
        q_tokens = _filter_glue(q_tokens)
        r_tokens = _filter_glue(r_tokens)

    if not q_tokens and not r_tokens:
        return 1.0
    if not q_tokens or not r_tokens:
        return 0.0

    matcher = SequenceMatcher(None, q_tokens, r_tokens, autojunk=False)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    return round(lcs_len / len(q_tokens), 4)


def coverage_f1(pred_list: list, gold_list: list,
                use_stopwords: bool = True) -> dict:
    """
    Coverage Precision, Recall, and F1 for two flat string lists.
    """
    pred_list = [str(x).strip().lower() for x in (pred_list or [])]
    gold_list = [str(x).strip().lower() for x in (gold_list or [])]

    if not gold_list and not pred_list:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "precision_per_item": [], "recall_per_item": []}
    if not gold_list:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                "precision_per_item": [(p, 0.0) for p in pred_list],
                "recall_per_item": []}
    if not pred_list:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0,
                "precision_per_item": [],
                "recall_per_item": [(g, 0.0) for g in gold_list]}

    gold_ref = " ".join(gold_list)
    pred_ref = " ".join(pred_list)

    precision_per_item = [
        (p, token_coverage(p, gold_ref, use_stopwords)) for p in pred_list
    ]
    recall_per_item = [
        (g, token_coverage(g, pred_ref, use_stopwords)) for g in gold_list
    ]

    precision = round(sum(s for _, s in precision_per_item) / len(precision_per_item), 4)
    recall    = round(sum(s for _, s in recall_per_item)    / len(recall_per_item),    4)
    f1 = round(2 * precision * recall / (precision + recall), 4) \
        if (precision + recall) > 0 else 0.0

    return {
        "precision":          precision,
        "recall":             recall,
        "f1":                 f1,
        "precision_per_item": precision_per_item,
        "recall_per_item":    recall_per_item,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def avg(values) -> float:
    """Average of a collection, ignoring None. Returns 0.0 if empty."""
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def score_subfields(pred_item: dict | None, gold_item: dict,
                    skip_keys: set = None) -> dict:
    """
    Score each sub-field of a matched item pair using chrF.
    pred_item = None when the gold item was unmatched (model missed it).
    """
    skip_keys = skip_keys or set()
    scores = {}

    for key, gold_val in gold_item.items():
        if key in skip_keys:
            continue

        pred_val = pred_item.get(key) if pred_item else None

        if isinstance(gold_val, list):
            gold_str = " | ".join(str(v) for v in gold_val) if gold_val else None
            if isinstance(pred_val, list):
                pred_str = " | ".join(str(v) for v in pred_val) if pred_val else None
            else:
                pred_str = str(pred_val) if pred_val else None
        else:
            gold_str = str(gold_val) if gold_val is not None else None
            pred_str = str(pred_val) if pred_val is not None else None

        scores[key] = chrf_score(pred_str, gold_str)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Structured item list matcher
# ══════════════════════════════════════════════════════════════════════════════

def match_and_score_list(pred_list: list, gold_list: list,
                         name_key: str,
                         skip_keys: set = None,
                         match_threshold: float = 0.3) -> dict:
    """
    Match pred items to gold items by best chrF on name_key, then score
    sub-fields of matched pairs with chrF.
    """
    skip_keys = skip_keys or set()
    pred_list = pred_list or []
    gold_list = gold_list or []

    if not gold_list and not pred_list:
        return {"per_item": [], "unmatched_gold": [], "unmatched_pred": [],
                "field_avg": 1.0}
    if not gold_list:
        return {"per_item": [], "unmatched_gold": [],
                "unmatched_pred": [str(p.get(name_key, "")) for p in pred_list],
                "field_avg": 0.0}

    gold_names = [str(g.get(name_key, "")).strip().lower() for g in gold_list]
    pred_names = [str(p.get(name_key, "")).strip().lower() for p in pred_list]

    sim_matrix = [
        [chrf_score(pred_names[j], gold_names[i]) for j in range(len(pred_list))]
        for i in range(len(gold_list))
    ]

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

    all_item_avgs = []
    per_item_results = []

    for gi, gold_item in enumerate(gold_list):
        pi        = matched_gold.get(gi, None)
        pred_item = pred_list[pi] if pi is not None else None
        name_score = sim_matrix[gi][pi] if pi is not None else 0.0

        sub_scores = score_subfields(pred_item, gold_item,
                                     skip_keys={name_key} | skip_keys)
        item_avg = avg(sub_scores.values())
        all_item_avgs.append(item_avg)

        per_item_results.append({
            "name":            gold_item.get(name_key),
            "matched_to":      pred_list[pi].get(name_key) if pred_item else None,
            "name_chrf":       round(name_score, 4),
            "matched":         pred_item is not None,
            "subfield_scores": sub_scores,
            "item_avg":        item_avg,
        })

    unmatched_pred_items = [pred_list[j] for j in range(len(pred_list))
                            if j not in matched_pred]
    for _ in unmatched_pred_items:
        all_item_avgs.append(0.0)

    unmatched_gold_items = [gold_list[i] for i in range(len(gold_list))
                            if i not in matched_gold]

    return {
        "per_item":       per_item_results,
        "unmatched_gold": [g.get(name_key) for g in unmatched_gold_items],
        "unmatched_pred": [p.get(name_key) for p in unmatched_pred_items],
        "field_avg":      avg(all_item_avgs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Per-section scorers
# ══════════════════════════════════════════════════════════════════════════════

def score_labs(pred_inv: dict, gold_inv: dict) -> dict:
    return match_and_score_list(
        pred_inv.get("labs", []),
        gold_inv.get("labs", []),
        name_key="lab_investigation_name",
    )


def score_imaging(pred_inv: dict, gold_inv: dict) -> dict:
    def augment(items):
        result = []
        for item in (items or []):
            item = dict(item)
            modality = item.get("imaging_modality", "")
            site     = item.get("site", "")
            item["_key"] = f"{modality} {site}".strip().lower()
            result.append(item)
        return result

    return match_and_score_list(
        augment(pred_inv.get("imaging", [])),
        augment(gold_inv.get("imaging", [])),
        name_key="_key",
    )


def score_monitoring(pred_mon: list, gold_mon: list) -> dict:
    return match_and_score_list(
        pred_mon or [], gold_mon or [],
        name_key="monitoring_parameter",
    )


def score_medical(pred_tx: dict, gold_tx: dict) -> dict:
    return match_and_score_list(
        pred_tx.get("medical", []),
        gold_tx.get("medical", []),
        name_key="name",
    )


def score_conservative(pred_tx: dict, gold_tx: dict,
                        use_stopwords: bool = True) -> dict:
    pred_c = pred_tx.get("conservative") or {}
    gold_c = gold_tx.get("conservative") or {}

    method_result = coverage_f1(
        pred_c.get("conservative_method"),
        gold_c.get("conservative_method"),
        use_stopwords=use_stopwords,
    )
    lifestyle_result = coverage_f1(
        pred_c.get("lifestyle_habit_modifications"),
        gold_c.get("lifestyle_habit_modifications"),
        use_stopwords=use_stopwords,
    )

    field_avg = avg([method_result["f1"], lifestyle_result["f1"]])

    return {
        "conservative_method":           method_result,
        "lifestyle_habit_modifications": lifestyle_result,
        "field_avg":                     field_avg,
    }


def score_follow_up(pred_fu: list, gold_fu: list) -> dict:
    pred_fu = pred_fu or []
    gold_fu = gold_fu or []

    if not gold_fu and not pred_fu:
        return {"per_item": [], "field_avg": 1.0}

    max_len = max(len(pred_fu), len(gold_fu))
    per_item = []
    all_avgs = []

    for i in range(max_len):
        p = pred_fu[i] if i < len(pred_fu) else {}
        g = gold_fu[i] if i < len(gold_fu) else {}
        sub = score_subfields(p, g)
        item_avg = avg(sub.values())
        all_avgs.append(item_avg)
        per_item.append({"subfield_scores": sub, "item_avg": item_avg})

    return {"per_item": per_item, "field_avg": avg(all_avgs)}


def score_referral(pred_ref: list, gold_ref: list) -> dict:
    return match_and_score_list(
        pred_ref or [], gold_ref or [],
        name_key="specialty_or_doctor",
    )


def score_patient_education(pred_val, gold_val) -> dict:
    score = chrf_score(
        str(pred_val) if pred_val else None,
        str(gold_val) if gold_val else None,
    )
    return {"field_avg": score}


def score_when_to_seek(pred_val, gold_val) -> dict:
    score = chrf_score(
        str(pred_val) if pred_val else None,
        str(gold_val) if gold_val else None,
    )
    return {"field_avg": score}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Confusion detection
# ══════════════════════════════════════════════════════════════════════════════

def _extract_string_tokens(field_name: str, extraction: dict) -> str:
    """
    Extract a single flat string of all text content from a field,
    suitable for token_coverage comparison.

    Handles:
      - flat string fields (patient_education, when_to_seek_medical_care)
      - list-of-strings fields (conservative_method, lifestyle_habit_modifications,
                                prevention, complementary_therapies)
      - list-of-dicts fields (labs, imaging, monitoring, medical, referral,
                              follow_up, external_equipment, tissue_sampling)
      - nested dict fields (conservative, treatment sub-dicts)

    Returns a single lowercased string of all text tokens joined by spaces.
    """
    inv  = extraction.get("investigations", {}) or {}
    tx   = extraction.get("treatment", {}) or {}
    cons = tx.get("conservative") or {}

    # Map field names to their content
    field_map = {
        # ── flat strings ────────────────────────────────────────────────────
        "patient_education":        extraction.get("patient_education"),
        "when_to_seek_medical_care": extraction.get("when_to_seek_medical_care"),

        # ── list of strings ─────────────────────────────────────────────────
        "conservative_method":      cons.get("conservative_method") or [],
        "lifestyle_habit_modifications": cons.get("lifestyle_habit_modifications") or [],
        "prevention":               tx.get("prevention") or [],
        "complementary_therapies":  tx.get("complementary_therapies") or [],

        # ── list of dicts: extract key text fields ───────────────────────────
        "labs": [
            item.get("lab_investigation_name", "")
            for item in (inv.get("labs") or [])
        ],
        "imaging": [
            f"{item.get('imaging_modality', '')} {item.get('site', '')}".strip()
            for item in (inv.get("imaging") or [])
        ],
        "imaging_modality": [
            item.get("imaging_modality", "")
            for item in (inv.get("imaging") or [])
        ],
        "imaging_site": [
            item.get("site", "")
            for item in (inv.get("imaging") or [])
        ],
        "tissue_sampling": [
            item.get("tissue_sampling_method", item.get("name", ""))
            for item in (inv.get("tissue_sampling") or [])
        ],
        "monitoring": [
            item.get("monitoring_parameter", "")
            for item in (extraction.get("monitoring") or [])
        ],
        "medical_names": [
            item.get("name", "")
            for item in (tx.get("medical") or [])
        ],
        "medical_routes": [
            item.get("route", "")
            for item in (tx.get("medical") or [])
            if item.get("route")
        ],
        "medical_dosage_forms": [
            item.get("dosage_form", "")
            for item in (tx.get("medical") or [])
            if item.get("dosage_form")
        ],
        "external_equipment": [
            item.get("name", item.get("equipment_name", ""))
            for item in (tx.get("external_equipment") or [])
        ],
        "referral": [
            f"{item.get('specialty_or_doctor', '')} {item.get('aim_of_referral', '')}".strip()
            for item in (extraction.get("referral") or [])
        ],
        "follow_up": [
            f"{item.get('scheduled_follow_up_time', '')} {item.get('aim_of_follow_up', '')}".strip()
            for item in (extraction.get("follow_up") or [])
        ],
    }

    content = field_map.get(field_name)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip().lower()
    if isinstance(content, list):
        return " ".join(str(x).strip().lower() for x in content if x).strip()
    return ""


def _confusion_score(gold_field_a: str, pred_field_b: str,
                     gold_extraction: dict, pred_extraction: dict,
                     use_stopwords: bool = True) -> float:
    """
    Measure how much of gold's field_A content appears in pred's field_B.

    A high score means the model put field_A content into field_B instead.
    Returns token_coverage(gold_A_tokens, pred_B_tokens).
    """
    gold_a = _extract_string_tokens(gold_field_a, gold_extraction)
    pred_b = _extract_string_tokens(pred_field_b, pred_extraction)

    if not gold_a and not pred_b:
        return 0.0   # nothing to confuse — not flagged
    if not gold_a or not pred_b:
        return 0.0

    return token_coverage(gold_a, pred_b, use_stopwords=use_stopwords)


# All confusion pairs as (field_A, field_B, description)
# Each pair is checked bidirectionally:
#   A→B: gold A content found in pred B  (model put A's content into B)
#   B→A: gold B content found in pred A  (model put B's content into A)
CONFUSION_PAIRS = [
    # Intra-item field swaps (within same structured item)
    ("medical_routes",          "medical_dosage_forms",   "medical: route ↔ dosage_form"),
    ("imaging_modality",        "imaging_site",           "imaging: modality ↔ site"),

    # Cross-section confusions
    ("follow_up",               "referral",               "follow_up ↔ referral"),
    ("labs",                    "monitoring",             "labs ↔ monitoring"),
    ("imaging",                 "monitoring",             "imaging ↔ monitoring"),
    ("prevention",              "patient_education",      "prevention ↔ patient_education"),
    ("conservative_method",     "medical_names",          "conservative_method ↔ medical"),
    ("medical_names",           "external_equipment",     "medical ↔ external_equipment"),
    ("imaging",                 "labs",                   "imaging ↔ labs"),
    ("tissue_sampling",         "imaging",                "tissue_sampling ↔ imaging"),
    ("imaging",                 "referral",               "imaging ↔ referral"),
    ("labs",                    "referral",               "labs ↔ referral"),

    # Treatment/lifestyle cluster  (each pair within the cluster)
    ("prevention",              "conservative_method",    "prevention ↔ conservative_method"),
    ("prevention",              "lifestyle_habit_modifications", "prevention ↔ lifestyle"),
    ("prevention",              "complementary_therapies","prevention ↔ complementary_therapies"),
    ("conservative_method",     "lifestyle_habit_modifications", "conservative_method ↔ lifestyle"),
    ("conservative_method",     "complementary_therapies","conservative_method ↔ complementary_therapies"),
    ("lifestyle_habit_modifications", "complementary_therapies", "lifestyle ↔ complementary_therapies"),
]


def detect_confusion(pred_extraction: dict, gold_extraction: dict,
                     threshold: float = 0.5,
                     use_stopwords: bool = True) -> dict:
    """
    Run all confusion pair checks for a single case.

    For each pair (A, B):
      - A→B score: gold A content found in pred B
      - B→A score: gold B content found in pred A
      - flagged if either direction exceeds threshold

    Returns:
        pairs     : list of per-pair results (always, even if not flagged)
        flagged   : list of pair descriptions where confusion was detected
        any_flagged : bool
    """
    pair_results = []
    flagged      = []

    for field_a, field_b, description in CONFUSION_PAIRS:
        score_a_in_b = _confusion_score(field_a, field_b,
                                        gold_extraction, pred_extraction,
                                        use_stopwords=use_stopwords)
        score_b_in_a = _confusion_score(field_b, field_a,
                                        gold_extraction, pred_extraction,
                                        use_stopwords=use_stopwords)

        is_flagged = score_a_in_b >= threshold or score_b_in_a >= threshold

        result = {
            "pair":         description,
            "field_a":      field_a,
            "field_b":      field_b,
            "a_in_b":       round(score_a_in_b, 4),   # gold A content in pred B
            "b_in_a":       round(score_b_in_a, 4),   # gold B content in pred A
            "flagged":      is_flagged,
        }
        pair_results.append(result)
        if is_flagged:
            flagged.append(description)

    return {
        "pairs":       pair_results,
        "flagged":     flagged,
        "any_flagged": len(flagged) > 0,
    }


def aggregate_confusion(case_confusions: list, case_ids: list) -> dict:
    """
    Aggregate per-case confusion results into dataset-level statistics.

    For each pair: reports how many cases were flagged, the avg scores
    in each direction, and the list of row_ids where the confusion was flagged.

    Args:
        case_confusions : list of per-case confusion dicts (from detect_confusion)
        case_ids        : list of row_ids in the same order as case_confusions
    """
    pair_accumulator = defaultdict(lambda: {
        "a_in_b": [], "b_in_a": [], "flagged_count": 0,
        "total": 0, "flagged_row_ids": []
    })

    for case_confusion, row_id in zip(case_confusions, case_ids):
        for pair_result in case_confusion["pairs"]:
            desc = pair_result["pair"]
            pair_accumulator[desc]["a_in_b"].append(pair_result["a_in_b"])
            pair_accumulator[desc]["b_in_a"].append(pair_result["b_in_a"])
            pair_accumulator[desc]["total"] += 1
            if pair_result["flagged"]:
                pair_accumulator[desc]["flagged_count"] += 1
                pair_accumulator[desc]["flagged_row_ids"].append(row_id)

    summary = {}
    for desc, data in pair_accumulator.items():
        total = data["total"]
        summary[desc] = {
            "flagged_cases":    data["flagged_count"],
            "total_cases":      total,
            "flagged_pct":      round(data["flagged_count"] / total * 100, 1) if total else 0.0,
            "avg_a_in_b":       avg(data["a_in_b"]),
            "avg_b_in_a":       avg(data["b_in_a"]),
            "flagged_row_ids":  data["flagged_row_ids"],
        }

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Case-level and dataset-level evaluation
# ══════════════════════════════════════════════════════════════════════════════

def score_case(pred_extraction: dict, gold_extraction: dict,
               use_stopwords: bool = True,
               run_confusion: bool = False,
               confusion_threshold: float = 0.5) -> dict:
    """
    Score one case across all extraction fields.
    Optionally runs confusion detection if run_confusion=True.
    """
    pred_inv = pred_extraction.get("investigations", {})
    gold_inv = gold_extraction.get("investigations", {})
    pred_tx  = pred_extraction.get("treatment", {})
    gold_tx  = gold_extraction.get("treatment", {})

    fields = {
        "labs":       score_labs(pred_inv, gold_inv),
        "imaging":    score_imaging(pred_inv, gold_inv),
        "monitoring": score_monitoring(
                          pred_extraction.get("monitoring"),
                          gold_extraction.get("monitoring")),
        "medical":    score_medical(pred_tx, gold_tx),
        "conservative": score_conservative(pred_tx, gold_tx,
                                           use_stopwords=use_stopwords),
        "follow_up":  score_follow_up(
                          pred_extraction.get("follow_up"),
                          gold_extraction.get("follow_up")),
        "referral":   score_referral(
                          pred_extraction.get("referral"),
                          gold_extraction.get("referral")),
        "patient_education": score_patient_education(
                          pred_extraction.get("patient_education"),
                          gold_extraction.get("patient_education")),
        "when_to_seek_medical_care": score_when_to_seek(
                          pred_extraction.get("when_to_seek_medical_care"),
                          gold_extraction.get("when_to_seek_medical_care")),
    }

    field_scores = {name: result["field_avg"] for name, result in fields.items()}
    overall      = avg(field_scores.values())

    result = {
        "field_scores":  field_scores,
        "field_details": fields,
        "overall":       overall,
    }

    if run_confusion:
        result["confusion"] = detect_confusion(
            pred_extraction, gold_extraction,
            threshold=confusion_threshold,
            use_stopwords=use_stopwords,
        )

    return result


def evaluate_dataset(pred_list: list, gold_list: list,
                     use_stopwords: bool = True,
                     run_confusion: bool = False,
                     confusion_threshold: float = 0.5) -> dict:
    """
    Evaluate the full dataset.
    """
    gold_by_id = {item["row_index"]: item for item in gold_list}

    case_results      = []
    field_accumulator = defaultdict(list)
    case_confusions   = []

    for pred_item in pred_list:
        row_id = pred_item["original_row_id"]

        if pred_item.get("failed"):
            print(f"[SKIP] row_id={row_id} marked as failed, skipping.")
            continue

        if row_id not in gold_by_id:
            print(f"[WARN] No gold found for row_id={row_id}, skipping.")
            continue

        gold_item = gold_by_id[row_id]
        result    = score_case(
            pred_item["extraction"], gold_item["extraction"],
            use_stopwords=use_stopwords,
            run_confusion=run_confusion,
            confusion_threshold=confusion_threshold,
        )
        result["row_id"] = row_id
        case_results.append(result)

        for field, score in result["field_scores"].items():
            field_accumulator[field].append(score)

        if run_confusion and "confusion" in result:
            case_confusions.append(result["confusion"])

    dataset_field_scores = {
        field: avg(scores) for field, scores in field_accumulator.items()
    }
    dataset_overall = avg(dataset_field_scores.values())

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
# SECTION 8 — Reporting
# ══════════════════════════════════════════════════════════════════════════════

_COVERAGE_FIELDS     = {"conservative"}
_CONSERVATIVE_SUBFIELDS = {"conservative_method", "lifestyle_habit_modifications"}


def print_report(results: dict, run_confusion: bool = False,
                 confusion_threshold: float = 0.5) -> None:
    """Print a human-readable evaluation report to stdout."""

    W = 72

    print()
    print("=" * W)
    print("  MEDICAL EXTRACTION EVALUATION REPORT")
    print("=" * W)
    print(f"  Cases evaluated : {len(results['per_case'])}")
    print(f"  String lists    : Coverage F1 (LCS + glue removal)")
    print(f"  Structured items: chrF matching + sub-field chrF scoring")
    if run_confusion:
        print(f"  Confusion detect: ENABLED  (threshold={confusion_threshold})")
    print("=" * W)

    # ── Dataset-level field scores ─────────────────────────────────────────
    print(f"\n  {'Field':<35} {'Score':>8}  Metric")
    print(f"  {'─'*35} {'─'*8}  {'─'*20}")
    for field, score in results["dataset_field_scores"].items():
        metric = "Coverage F1" if field in _COVERAGE_FIELDS else "chrF"
        print(f"  {field:<35} {score:>8.4f}  {metric}")
    print(f"  {'─'*35} {'─'*8}")
    print(f"  {'OVERALL':<35} {results['dataset_overall']:>8.4f}")
    print()

    # ── Conservative field breakdown ──────────────────────────────────────
    print("─" * W)
    print("  CONSERVATIVE TREATMENT  (Coverage F1 detail)")
    print("─" * W)
    for case in results["per_case"]:
        row_id = case["row_id"]
        cons   = case["field_details"].get("conservative", {})
        print(f"\n  Case row_id={row_id}")
        for sub in _CONSERVATIVE_SUBFIELDS:
            sub_result = cons.get(sub, {})
            p  = sub_result.get("precision", "—")
            r  = sub_result.get("recall",    "—")
            f1 = sub_result.get("f1",        "—")
            p_items = sub_result.get("precision_per_item", [])
            r_items = sub_result.get("recall_per_item",    [])
            label = sub.replace("_", " ")
            print(f"    {label}")
            print(f"      P={p:.4f}  R={r:.4f}  F1={f1:.4f}" if isinstance(f1, float)
                  else f"      {p} / {r} / {f1}")
            if p_items:
                print(f"      Precision per pred item:")
                for item, score in p_items:
                    print(f"        [{score:.2f}]  \"{item}\"")
            if r_items:
                print(f"      Recall per gold item:")
                for item, score in r_items:
                    print(f"        [{score:.2f}]  \"{item}\"")

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

    # ── Confusion detection report ─────────────────────────────────────────
    if run_confusion and "confusion_summary" in results:
        print()
        print("=" * W)
        print(f"  FIELD CONFUSION REPORT  (threshold={confusion_threshold})")
        print("=" * W)
        print(f"  {'Confusion Pair':<48} {'Flag%':>6}  {'A→B':>6}  {'B→A':>6}  {'Cases':>6}")
        print(f"  {'─'*48} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

        summary = results["confusion_summary"]
        # Sort by flagged_pct descending so worst offenders appear first
        sorted_pairs = sorted(summary.items(),
                              key=lambda x: x[1]["flagged_pct"], reverse=True)

        for desc, data in sorted_pairs:
            flagged_pct = data["flagged_pct"]
            avg_a_in_b  = data["avg_a_in_b"]
            avg_b_in_a  = data["avg_b_in_a"]
            flagged     = data["flagged_cases"]
            total       = data["total_cases"]
            row_ids     = data.get("flagged_row_ids", [])
            # Highlight pairs with any flagged cases
            marker = "  !" if flagged > 0 else "   "
            print(f"{marker} {desc:<48} {flagged_pct:>5.1f}%  "
                  f"{avg_a_in_b:>6.4f}  {avg_b_in_a:>6.4f}  "
                  f"{flagged:>3}/{total:<3}")
            if row_ids:
                ids_str = ", ".join(str(r) for r in row_ids)
                print(f"       row_ids: [{ids_str}]")

        # Per-case flagged list
        print()
        print("  PER-CASE FLAGS")
        print(f"  {'─'*48}")
        for case in results["per_case"]:
            if not case.get("confusion", {}).get("any_flagged"):
                continue
            row_id  = case["row_id"]
            flagged = case["confusion"]["flagged"]
            print(f"  row_id={row_id}")
            for f in flagged:
                print(f"    ! {f}")

    print()
    print("=" * W)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate medical treatment plan extraction against human annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_extraction.py --pred pred.json --gold gold.json
  python evaluate_extraction.py --pred pred.json --gold gold.json --output results.json
  python evaluate_extraction.py --pred pred.json --gold gold.json --no-stopwords
  python evaluate_extraction.py --pred pred.json --gold gold.json --confusion
  python evaluate_extraction.py --pred pred.json --gold gold.json --confusion --confusion-threshold 0.6
        """,
    )
    parser.add_argument("--pred",                  required=True,
                        help="Path to model predictions JSON")
    parser.add_argument("--gold",                  required=True,
                        help="Path to annotator gold JSON")
    parser.add_argument("--output",                default=None,
                        help="Optional path to save full results as JSON")
    parser.add_argument("--no-stopwords",          action="store_true",
                        help="Disable grammatical glue removal for Coverage F1")
    parser.add_argument("--confusion",             action="store_true",
                        help="Enable field confusion detection")
    parser.add_argument("--confusion-threshold",   type=float, default=0.5,
                        help="Coverage threshold to flag a confusion (default: 0.5)")
    args = parser.parse_args()

    use_stopwords = not args.no_stopwords

    print(f"\nLoading predictions : {args.pred}")
    print(f"Loading gold        : {args.gold}")
    print(f"Glue removal        : {'enabled' if use_stopwords else 'disabled'}")
    if args.confusion:
        print(f"Confusion detection : enabled  (threshold={args.confusion_threshold})")

    with open(args.pred) as f:
        pred_data = json.load(f)
    with open(args.gold) as f:
        gold_data = json.load(f)

    results = evaluate_dataset(
        pred_data, gold_data,
        use_stopwords=use_stopwords,
        run_confusion=args.confusion,
        confusion_threshold=args.confusion_threshold,
    )
    print_report(results,
                 run_confusion=args.confusion,
                 confusion_threshold=args.confusion_threshold)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full results saved to: {args.output}\n")


if __name__ == "__main__":
    main()


"""
python syntactic_eval_script.py \
    --pred extracted_plan_google_gemini-2.0-flash-001_pydantic_normalized.json \
    --gold extractions_annotator_2_normalized.json \
    --output annotator_2_gemini_2_flash_eval_results.json \
    --confusion \
    --confusion-threshold 0.5
"""