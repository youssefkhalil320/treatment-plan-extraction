import json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map of Unicode smart/curly quotes → straight ASCII equivalents
QUOTE_MAP = str.maketrans({
    '\u201c': '"',  # "  left double quotation mark
    '\u201d': '"',  # "  right double quotation mark
    '\u201e': '"',  # „  double low-9 quotation mark
    '\u201f': '"',  # ‟  double high-reversed-9 quotation mark
    '\u2018': "'",  # '  left single quotation mark
    '\u2019': "'",  # '  right single quotation mark
    '\u201a': "'",  # ‚  single low-9 quotation mark
    '\u201b': "'",  # ‛  single high-reversed-9 quotation mark
    '\u00ab': '"',  # «  left-pointing double angle quotation mark
    '\u00bb': '"',  # »  right-pointing double angle quotation mark
    '\u2039': "'",  # ‹  single left-pointing angle quotation mark
    '\u203a': "'",  # ›  single right-pointing angle quotation mark
})

# Grammatical glue words to strip at comparison time only.
# Never removed from the stored normalized JSON —
# only used ephemerally inside preprocess_for_comparison().
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

# ---------------------------------------------------------------------------
# Canonical key mapping
# ---------------------------------------------------------------------------
# Maps every known casing variant → canonical lowercase key.
# Applied as the very first step so all downstream logic uses
# consistent key names regardless of source annotator/model.

CANONICAL_KEYS = {
    # Top-level extraction fields
    "INVESTIGATIONS":                       "investigations",
    "MONITORING":                           "monitoring",
    "Monitoring":                           "monitoring",
    "TREATMENT":                            "treatment",
    "FOLLOW_UP":                            "follow_up",
    "PATIENT_EDUCATION":                    "patient_education",
    "REFERRAL":                             "referral",
    "WHEN_TO_SEEK_MEDICAL_CARE":            "when_to_seek_medical_care",

    # Investigations sub-fields
    "Labs":                                 "labs",
    "Imaging":                              "imaging",
    "Nerve_Muscle_Conduction_Studies":      "nerve_muscle_conduction_studies",
    "Tissue_Sampling":                      "tissue_sampling",
    "Tissue_sampling":                      "tissue_sampling",
    # FIX (Req 6): rename monitoring_routine_investigations → monitoring
    # (lifting to top-level is handled separately in normalize_extraction)
    "monitoring_routine_investigations":    "monitoring",
    "Monitoring_Routine_Investigations":    "monitoring",

    # Treatment sub-fields
    "Prevention":                           "prevention",
    "Conservative":                         "conservative",
    "Medical":                              "medical",
    "Surgical":                             "surgical",
    "COMPLEMENTARY_THERAPIES":              "complementary_therapies",
    "EXTERNAL_EQUIPMENT":                   "external_equipment",
}

# ---------------------------------------------------------------------------
# Fields to drop from specific sections (Req 4)
# ---------------------------------------------------------------------------

FIELDS_TO_DROP = {
    "follow_up":       {"follow_up_ordered"},
    "referral":        {"referral_ordered"},
    "tissue_sampling": {"tissue_sampling_ordered"},
}


def canonicalize_keys(obj):
    """
    Recursively rename all keys in a dict/list structure using CANONICAL_KEYS.
    Keys not present in the map are lowercased as a safe fallback.
    When two keys in the same dict map to the same canonical key, the one
    with more data (non-null, non-empty) wins; the other is discarded.
    """
    if isinstance(obj, dict):
        merged = {}
        for k, v in obj.items():
            canonical = CANONICAL_KEYS.get(k, k.lower())
            v = canonicalize_keys(v)
            if canonical not in merged:
                merged[canonical] = v
            else:
                merged[canonical] = _richer(merged[canonical], v)
        return merged
    elif isinstance(obj, list):
        return [canonicalize_keys(i) for i in obj]
    return obj


def _richer(a, b):
    """
    Return whichever of a or b is considered 'richer' (has more data).
    Priority: non-None > non-empty-list > non-empty-dict > the other.
    If both are equally rich, a (the incumbent) wins.
    """
    def score(x):
        if x is None:
            return 0
        if isinstance(x, list):
            return len(x)
        if isinstance(x, dict):
            return sum(1 for v in x.values() if v is not None)
        if isinstance(x, str):
            return len(x.strip())
        return 1  # bool, int, float — always has content
    return b if score(b) > score(a) else a


# ---------------------------------------------------------------------------
# String normalization  (stored — applied to the JSON output)
# ---------------------------------------------------------------------------

def normalize_string(s):
    """
    - Strip leading/trailing whitespace
    - Normalize smart/curly quotes to straight ASCII quotes
    - Lowercase
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = s.translate(QUOTE_MAP)
    s = s.lower()
    return s


def normalize_strings_in_obj(obj):
    """
    Recursively apply normalize_string() to every string leaf.
    Non-string scalars (int, float, bool, None) are left untouched.
    """
    if isinstance(obj, dict):
        return {k: normalize_strings_in_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_strings_in_obj(i) for i in obj]
    elif isinstance(obj, str):
        return normalize_string(obj)
    return obj


# ---------------------------------------------------------------------------
# Comparison preprocessing  (ephemeral — never written back to disk)
# ---------------------------------------------------------------------------

def remove_stopwords(s):
    """
    Remove GRAMMATICAL_GLUE words from an already-normalized string.
    Input assumed to be already lowercased and stripped.
    """
    if not isinstance(s, str):
        return s
    tokens = s.split()
    filtered = [tok for tok in tokens if tok not in GRAMMATICAL_GLUE]
    return " ".join(filtered)


def preprocess_for_comparison(obj):
    """
    Recursively apply stopword removal to all string values.
    Call at comparison time only on already-normalized data.
    Never store the result of this function back to disk.
    """
    if isinstance(obj, dict):
        return {k: preprocess_for_comparison(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [preprocess_for_comparison(i) for i in obj]
    elif isinstance(obj, str):
        return remove_stopwords(obj)
    return obj


# ---------------------------------------------------------------------------
# Structural normalization helpers
# ---------------------------------------------------------------------------

def all_none(obj):
    """Return True if obj is None, or a dict whose every value is None."""
    if obj is None:
        return True
    if isinstance(obj, dict):
        return all(v is None for v in obj.values())
    return False


def normalize_list_field(items):
    """
    - None → []
    - Filter out fully-null dicts
    """
    if items is None:
        return []
    if isinstance(items, list):
        return [item for item in items if not all_none(item)]
    return items


def wrap_as_list(value):
    """
    FIX (Req 2): Ensure a field is always a list of dicts.
    - None       → []
    - dict {}    → [{}]  (plain object → wrap in list)
    - list [..] → as-is  (already correct)
    Fully-null dicts are filtered out after wrapping.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        return [item for item in value if not all_none(item)]
    return value


def drop_fields(items, fields_to_drop):
    """
    FIX (Req 4): Remove specified keys from every dict in a list.
    fields_to_drop: set of key names to remove.
    """
    if not isinstance(items, list):
        return items
    result = []
    for item in items:
        if isinstance(item, dict):
            item = {k: v for k, v in item.items() if k not in fields_to_drop}
        result.append(item)
    return result


def normalize_side_effects(value):
    """side_effects_contraindications: None → [], list → unchanged."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return value


def normalize_imaging(imaging):
    """
    - None → []
    - Drop items where all non-contrast fields are null AND contrast is
      null or "no" / "No" (case-insensitive). These represent phantom
      imaging entries where the annotator only set contrast to No/null
      without entering any actual imaging data.
    """
    if imaging is None:
        return []
    if not isinstance(imaging, list):
        return imaging

    result = []
    for item in imaging:
        if item is None:
            continue
        non_contrast = {k: v for k, v in item.items() if k != "contrast"}
        contrast = item.get("contrast")
        contrast_lower = contrast.lower() if isinstance(contrast, str) else contrast
        if all(v is None for v in non_contrast.values()) and contrast_lower in (None, "no"):
            continue
        result.append(item)
    return result


def normalize_conservative(conservative):
    """
    - None / empty list → []
    - Unwrap single-element list → dict
    - Fully-null dict → []
    - conservative_method: None → []
    - lifestyle_habit_modifications: None → []
    """
    if conservative is None:
        return []

    if isinstance(conservative, list):
        items = [i for i in conservative if i is not None]
        if not items:
            return []
        conservative = items[0]

    if isinstance(conservative, dict):
        if all_none(conservative):
            return []

        cm = conservative.get("conservative_method")
        if cm is None:
            conservative["conservative_method"] = []

        lhm = conservative.get("lifestyle_habit_modifications")
        if lhm is None:
            conservative["lifestyle_habit_modifications"] = []

        return conservative

    return conservative


def normalize_medical_list(items):
    """
    Standard list normalization + ensure side_effects_contraindications
    is always a list (never null).
    """
    items = normalize_list_field(items)
    for item in items:
        if isinstance(item, dict):
            item["side_effects_contraindications"] = normalize_side_effects(
                item.get("side_effects_contraindications")
            )
    return items


def lift_monitoring_from_investigations(extraction, investigations):
    """
    FIX (Req 6): If investigations contains a 'monitoring' key
    (originally 'monitoring_routine_investigations', already renamed by
    canonicalize_keys), merge it into the top-level extraction['monitoring']
    and remove it from investigations.
    """
    nested_monitoring = investigations.pop("monitoring", None)
    if nested_monitoring:
        nested_monitoring = normalize_list_field(nested_monitoring)
        existing = normalize_list_field(extraction.get("monitoring"))
        # Merge: combine both lists, deduplicate by content
        combined = existing + [
            item for item in nested_monitoring if item not in existing
        ]
        extraction["monitoring"] = combined
    return extraction, investigations


# ---------------------------------------------------------------------------
# Core extraction normalizer
# ---------------------------------------------------------------------------

def normalize_extraction(extraction):
    """
    Full normalization pipeline for a single extraction dict:
      1. Canonicalize all key names (merge duplicates, prefer richer value)
      2. Structural normalization (null coercion, list unwrapping, field drops, etc.)
      3. Text normalization (strip, quote normalization, lowercase)
    """
    # ── STEP 1: Canonicalize keys ────────────────────────────────────────────
    extraction = canonicalize_keys(extraction)

    # ── STEP 2: Structural normalization ────────────────────────────────────

    # INVESTIGATIONS
    investigations = extraction.get("investigations", {}) or {}
    investigations["labs"] = normalize_list_field(
        investigations.get("labs")
    )
    investigations["imaging"] = normalize_imaging(
        investigations.get("imaging")
    )
    investigations["nerve_muscle_conduction_studies"] = normalize_list_field(
        investigations.get("nerve_muscle_conduction_studies")
    )
    # tissue_sampling: normalize list + drop tissue_sampling_ordered (Req 4)
    investigations["tissue_sampling"] = drop_fields(
        normalize_list_field(investigations.get("tissue_sampling")),
        FIELDS_TO_DROP["tissue_sampling"]
    )

    # FIX (Req 6): lift monitoring out of investigations → top-level
    extraction, investigations = lift_monitoring_from_investigations(extraction, investigations)
    extraction["investigations"] = investigations

    # MONITORING (top-level)
    extraction["monitoring"] = normalize_list_field(
        extraction.get("monitoring")
    )

    # TREATMENT
    treatment = extraction.get("treatment", {}) or {}
    treatment["prevention"] = normalize_list_field(
        treatment.get("prevention")
    )
    treatment["conservative"] = normalize_conservative(
        treatment.get("conservative")
    )
    treatment["medical"] = normalize_medical_list(
        treatment.get("medical")
    )
    treatment["surgical"] = normalize_list_field(
        treatment.get("surgical")
    )
    treatment["complementary_therapies"] = normalize_list_field(
        treatment.get("complementary_therapies")
    )
    treatment["external_equipment"] = normalize_list_field(
        treatment.get("external_equipment")
    )
    extraction["treatment"] = treatment

    # FOLLOW_UP
    # FIX (Req 2): wrap plain dict → list
    # FIX (Req 3): keep only scheduled_follow_up_time + aim_of_follow_up
    # FIX (Req 4): drop follow_up_ordered
    follow_up = wrap_as_list(extraction.get("follow_up"))
    follow_up = drop_fields(follow_up, FIELDS_TO_DROP["follow_up"])
    # Req 3: ensure only the two target fields are kept, defaulting to None
    normalized_follow_up = []
    for item in follow_up:
        if isinstance(item, dict):
            normalized_follow_up.append({
                "scheduled_follow_up_time": item.get("scheduled_follow_up_time"),
                "aim_of_follow_up":         item.get("aim_of_follow_up"),
            })
    extraction["follow_up"] = normalized_follow_up

    # REFERRAL
    # FIX (Req 2): wrap plain dict → list
    # FIX (Req 4): drop referral_ordered
    referral = wrap_as_list(extraction.get("referral"))
    referral = drop_fields(referral, FIELDS_TO_DROP["referral"])
    extraction["referral"] = referral

    # ── STEP 3: Text normalization (strip, quotes, lowercase) ────────────────
    extraction = normalize_strings_in_obj(extraction)

    return extraction


# ---------------------------------------------------------------------------
# Record and file entry points
# ---------------------------------------------------------------------------

def normalize_record(record):
    """Normalize a single top-level record."""
    extraction = record.get("extraction", {})
    record["extraction"] = normalize_extraction(extraction)
    return record


def normalize_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        normalized = [normalize_record(r) for r in data]
    else:
        normalized = normalize_record(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(f"Done. Normalized output written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python normalize_annotators_files.py input.json output.json")
        sys.exit(1)

    normalize_file(sys.argv[1], sys.argv[2])


"""
python normalize_annotators_files.py extracted_plan_all_google_gemini-2.0-flash-001_pydantic.json extracted_plan_all_google_gemini-2.0-flash-001_pydantic_normalized.json

python normalize_annotators_files.py all_annotations_merged.json all_annotations_merged_normalized.json

python normalize_annotators_files.py extracted_plan_all_openai_gpt-4o_pydantic.json extracted_plan_all_openai_gpt-4o_pydantic_normalized.json
"""