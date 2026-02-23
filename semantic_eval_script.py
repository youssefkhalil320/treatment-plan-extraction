"""
Medical Treatment Plan Extraction — Evaluation Script (MedGemma Embeddings)
============================================================================
Compares model extraction output against human annotator (gold) annotations.

Replaces all chrF scoring with MedGemma-4b-it embedding cosine similarity.
The model is loaded once at startup, all strings across the full dataset are
collected, deduplicated, batch-embedded, and cached in memory. Evaluation then
runs entirely from the cache — zero additional model calls.

─────────────────────────────────────────────────────────────────────────────
SCORING STRATEGY BY FIELD TYPE
─────────────────────────────────────────────────────────────────────────────

  Structured item lists  (labs, imaging, medications, monitoring, referrals)
  ──────────────────────
    • Items matched by name using greedy cosine similarity
    • Similarity threshold: 0.65 — below this, treated as unrelated (score 0.0)
    • Each matched pair → sub-fields scored individually with cosine similarity
    • Unmatched gold items (missed) → 0
    • Unmatched pred items (hallucinated) → 0
    • field_avg = avg(all item scores including unmatched)

  Flat string lists  (conservative_method, lifestyle_habit_modifications)
  ─────────────────
    • Each item embedded individually
    • Soft matching Coverage F1:
        Precision = avg( max cosine(p, g) for g in gold_list  for p in pred_list )
        Recall    = avg( max cosine(g, p) for p in pred_list  for g in gold_list )
        F1        = harmonic mean(P, R)
    • Threshold 0.65 applies: cosine < 0.65 treated as 0.0

  Single string fields  (patient_education, when_to_seek_medical_care,
                         follow_up sub-fields, referral aim)
  ─────────────────────
    • Direct cosine similarity between embeddings

─────────────────────────────────────────────────────────────────────────────
EMBEDDING DETAILS
─────────────────────────────────────────────────────────────────────────────

  Model  : google/medgemma-4b-it  (or any local path)
  Layer  : last hidden state, mean-pooled over non-padding tokens
  Norm   : L2-normalized → cosine similarity = dot product
  Null   : (None, None) → 1.0 | (None, X) or (X, None) → 0.0
  Thresh : cosine < 0.65 → 0.0  (unrelated medical text baseline gate)

─────────────────────────────────────────────────────────────────────────────
CONFUSION DETECTION  (optional, enabled with --confusion)
─────────────────────────────────────────────────────────────────────────────

  Same confusion pairs as original script.
  Coverage replaced by: cosine similarity between mean-pooled field embeddings.
  Threshold still controlled by --confusion-threshold (default 0.5).

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

  python evaluate_extraction_medgemma.py --pred predictions.json --gold annotations.json
  python evaluate_extraction_medgemma.py --pred predictions.json --gold annotations.json \\
      --output results.json --embedding-model google/medgemma-4b-it
  python evaluate_extraction_medgemma.py --pred predictions.json --gold annotations.json \\
      --confusion --confusion-threshold 0.5 --batch-size 64
  python evaluate_extraction_medgemma.py --pred predictions.json --gold annotations.json \\
      --cache-embeddings embeddings_cache.pt
"""

import json
import argparse
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Embedding engine
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Loads MedGemma-4b-it once, batch-embeds all strings, caches results.

    Usage pattern:
        engine = EmbeddingEngine(model_name, device, batch_size)
        engine.build_cache(list_of_all_strings)
        score = engine.similarity(str_a, str_b)
    """

    def __init__(self,
                 model_name: str = "google/medgemma-4b-it",
                 device: str = "auto",
                 batch_size: int = 32):

        self.model_name = model_name
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}

        # ── device resolution ────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[EmbeddingEngine] Loading tokenizer from: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"[EmbeddingEngine] Loading model → {self.device}")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[EmbeddingEngine] Model ready.")

    # ── core embedding ────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings in one forward pass.

        Strategy:
          1. Tokenize with padding + truncation (max 512 tokens)
          2. Forward pass → last_hidden_state  [B, seq_len, hidden_dim]
          3. Mean-pool over non-padding token positions
          4. L2-normalize each vector → cosine sim = dot product later

        Returns:
            np.ndarray of shape [len(texts), hidden_dim], float32
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        # last_hidden_state: [B, seq_len, H]
        hidden = output.last_hidden_state
        attention_mask = encoded["attention_mask"]  # [B, seq_len]

        # Mean pool: sum(hidden * mask) / sum(mask)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        sum_hidden = (hidden * mask_expanded).sum(dim=1)       # [B, H]
        sum_mask   = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        pooled     = sum_hidden / sum_mask                      # [B, H]

        # L2 normalize
        norms  = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        normed = (pooled / norms).cpu().float().numpy()         # [B, H]

        return normed

    # ── cache management ──────────────────────────────────────────────────────

    def build_cache(self, strings: list[str]) -> None:
        """
        Embed all unique non-empty strings and populate the cache.

        All strings from the entire dataset should be passed here in one call
        so that batching is maximally efficient — the model runs exactly once.
        """
        unique = list({s for s in strings if s and s.strip()})
        if not unique:
            print("[EmbeddingEngine] No strings to embed.")
            return

        print(f"[EmbeddingEngine] Embedding {len(unique)} unique strings "
              f"(batch_size={self.batch_size}) ...")

        all_embeddings = []
        batches = range(0, len(unique), self.batch_size)
        for i in tqdm(batches, desc="Embedding batches", unit="batch"):
            batch = unique[i : i + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)  # [N, H]
        for string, vec in zip(unique, all_embeddings):
            self._cache[string] = vec

        print(f"[EmbeddingEngine] Cache built: {len(self._cache)} entries.")

    def save_cache(self, path: str) -> None:
        """Persist embedding cache to disk as a .pt file."""
        torch.save(self._cache, path)
        print(f"[EmbeddingEngine] Cache saved → {path}")

    def load_cache(self, path: str) -> None:
        """Load a previously saved embedding cache from disk."""
        self._cache = torch.load(path, map_location="cpu")
        print(f"[EmbeddingEngine] Cache loaded: {len(self._cache)} entries from {path}")

    # ── similarity ────────────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> np.ndarray | None:
        """Return cached embedding for text, or None if not found."""
        if not text or not text.strip():
            return None
        return self._cache.get(text.strip().lower())

    def similarity(self, pred: str | None, gold: str | None,
                   threshold: float = 0.65) -> float:
        """
        Cosine similarity between two strings via cached embeddings.

        Null contract (preserved from original chrF version):
            (None, None) → 1.0   both agree nothing is here
            (None, X)    → 0.0   one side missing
            (X, None)    → 0.0

        Threshold gate:
            cosine < threshold → 0.0  (treat as unrelated)
        """
        # Null contract
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0

        pred_clean = str(pred).strip().lower()
        gold_clean = str(gold).strip().lower()

        # Exact match shortcut (also handles identical strings not yet embedded)
        if pred_clean == gold_clean:
            return 1.0

        pred_vec = self.get_embedding(pred_clean)
        gold_vec = self.get_embedding(gold_clean)

        # Fallback if embedding missing (shouldn't happen after build_cache)
        if pred_vec is None or gold_vec is None:
            return 0.0

        cosine = float(np.dot(pred_vec, gold_vec))  # already L2-normalized

        return round(cosine if cosine >= threshold else 0.0, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Embedding Coverage F1  (replaces token LCS Coverage F1)
# ══════════════════════════════════════════════════════════════════════════════

def embedding_coverage_f1(pred_list: list,
                           gold_list: list,
                           engine: EmbeddingEngine,
                           threshold: float = 0.65) -> dict:
    """
    Soft-matching Coverage Precision, Recall, and F1 for two flat string lists.

    For each pred item  → best cosine match across all gold items  = precision score
    For each gold item  → best cosine match across all pred items  = recall score

    This preserves the spirit of the original Coverage F1 (handles splits/merges)
    but uses semantic similarity rather than LCS token overlap.

    Edge cases mirror original:
        both empty      → P=1, R=1, F1=1
        pred empty only → P=1, R=0, F1=0
        gold empty only → P=0, R=1, F1=0
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

    # Build similarity matrix [len(pred), len(gold)]
    sim_matrix = np.array([
        [engine.similarity(p, g, threshold=threshold)
         for g in gold_list]
        for p in pred_list
    ])  # shape: [P, G]

    # Precision: each pred matched to best gold
    precision_per_item = [
        (pred_list[i], float(sim_matrix[i].max()))
        for i in range(len(pred_list))
    ]

    # Recall: each gold matched to best pred
    recall_per_item = [
        (gold_list[j], float(sim_matrix[:, j].max()))
        for j in range(len(gold_list))
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
# SECTION 3 — String collection pass  (feeds build_cache)
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_strings(pred_list: list, gold_list: list) -> list[str]:
    """
    Walk every extraction in pred and gold, collect every string value
    that will be embedded during evaluation.

    This single pass ensures build_cache() gets ALL strings upfront so
    the model runs exactly once over the dataset.
    """
    strings = set()

    def _collect_extraction(ext: dict) -> None:
        if not ext:
            return

        inv  = ext.get("investigations", {}) or {}
        tx   = ext.get("treatment", {}) or {}
        cons = tx.get("conservative") or {}

        # ── investigations ────────────────────────────────────────────────
        for lab in (inv.get("labs") or []):
            _collect_item(lab)

        for img in (inv.get("imaging") or []):
            _collect_item(img)
            # synthetic key used in matching
            key = f"{img.get('imaging_modality', '')} {img.get('site', '')}".strip().lower()
            if key:
                strings.add(key)

        for ts in (inv.get("tissue_sampling") or []):
            _collect_item(ts)

        # ── monitoring ────────────────────────────────────────────────────
        for mon in (ext.get("monitoring") or []):
            _collect_item(mon)

        # ── treatment: medical ────────────────────────────────────────────
        for med in (tx.get("medical") or []):
            _collect_item(med)

        # ── treatment: conservative ───────────────────────────────────────
        for item in (cons.get("conservative_method") or []):
            strings.add(str(item).strip().lower())
        for item in (cons.get("lifestyle_habit_modifications") or []):
            strings.add(str(item).strip().lower())

        # ── treatment: other list fields ──────────────────────────────────
        for item in (tx.get("prevention") or []):
            strings.add(str(item).strip().lower())
        for item in (tx.get("complementary_therapies") or []):
            strings.add(str(item).strip().lower())
        for eq in (tx.get("external_equipment") or []):
            _collect_item(eq)

        # ── follow_up ─────────────────────────────────────────────────────
        for fu in (ext.get("follow_up") or []):
            _collect_item(fu)

        # ── referral ──────────────────────────────────────────────────────
        for ref in (ext.get("referral") or []):
            _collect_item(ref)

        # ── flat string fields ────────────────────────────────────────────
        for field in ("patient_education", "when_to_seek_medical_care"):
            val = ext.get(field)
            if val:
                strings.add(str(val).strip().lower())

    def _collect_item(item: dict) -> None:
        """Collect all string leaf values from a dict item."""
        if not item:
            return
        for val in item.values():
            if isinstance(val, str) and val.strip():
                strings.add(val.strip().lower())
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, str) and v.strip():
                        strings.add(v.strip().lower())

    for entry in pred_list:
        _collect_extraction(entry.get("extraction", {}))
    for entry in gold_list:
        _collect_extraction(entry.get("extraction", {}))

    return [s for s in strings if s]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def avg(values) -> float:
    """Average of a collection, ignoring None. Returns 0.0 if empty."""
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def score_subfields(pred_item: dict | None,
                    gold_item: dict,
                    engine: EmbeddingEngine,
                    skip_keys: set = None,
                    threshold: float = 0.65) -> dict:
    """
    Score each sub-field of a matched item pair using embedding cosine similarity.
    pred_item = None when the gold item was unmatched (model missed it).
    """
    skip_keys = skip_keys or set()
    scores = {}

    for key, gold_val in gold_item.items():
        if key in skip_keys:
            continue

        pred_val = pred_item.get(key) if pred_item else None

        # Flatten lists to strings for embedding
        if isinstance(gold_val, list):
            gold_str = " ".join(str(v) for v in gold_val) if gold_val else None
            if isinstance(pred_val, list):
                pred_str = " ".join(str(v) for v in pred_val) if pred_val else None
            else:
                pred_str = str(pred_val).strip() if pred_val else None
        else:
            gold_str = str(gold_val).strip() if gold_val is not None else None
            pred_str = str(pred_val).strip() if pred_val is not None else None

        scores[key] = engine.similarity(pred_str, gold_str, threshold=threshold)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Structured item list matcher  (greedy cosine instead of chrF)
# ══════════════════════════════════════════════════════════════════════════════

def match_and_score_list(pred_list: list,
                         gold_list: list,
                         name_key: str,
                         engine: EmbeddingEngine,
                         skip_keys: set = None,
                         threshold: float = 0.65) -> dict:
    """
    Match pred items to gold items by best cosine similarity on name_key,
    then score sub-fields of matched pairs with embedding similarity.

    Matching threshold: cosine >= threshold to be eligible for matching.
    Items below threshold are treated as unrelated (score 0.0).
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

    # Build cosine similarity matrix for names
    sim_matrix = [
        [engine.similarity(pred_names[j], gold_names[i], threshold=threshold)
         for j in range(len(pred_list))]
        for i in range(len(gold_list))
    ]

    # Greedy matching — same logic as original, cosine replaces chrF
    candidates = [
        (sim_matrix[i][j], i, j)
        for i in range(len(gold_list))
        for j in range(len(pred_list))
        if sim_matrix[i][j] > 0.0  # threshold already applied inside similarity()
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

        sub_scores = score_subfields(
            pred_item, gold_item, engine,
            skip_keys={name_key} | skip_keys,
            threshold=threshold,
        )
        item_avg = avg(sub_scores.values())
        all_item_avgs.append(item_avg)

        per_item_results.append({
            "name":            gold_item.get(name_key),
            "matched_to":      pred_list[pi].get(name_key) if pred_item else None,
            "name_cosine":     round(name_score, 4),
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
# SECTION 6 — Per-section scorers
# ══════════════════════════════════════════════════════════════════════════════

def score_labs(pred_inv: dict, gold_inv: dict,
               engine: EmbeddingEngine, threshold: float) -> dict:
    return match_and_score_list(
        pred_inv.get("labs", []),
        gold_inv.get("labs", []),
        name_key="lab_investigation_name",
        engine=engine, threshold=threshold,
    )


def score_imaging(pred_inv: dict, gold_inv: dict,
                  engine: EmbeddingEngine, threshold: float) -> dict:
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
        engine=engine, threshold=threshold,
    )


def score_monitoring(pred_mon: list, gold_mon: list,
                     engine: EmbeddingEngine, threshold: float) -> dict:
    return match_and_score_list(
        pred_mon or [], gold_mon or [],
        name_key="monitoring_parameter",
        engine=engine, threshold=threshold,
    )


def score_medical(pred_tx: dict, gold_tx: dict,
                  engine: EmbeddingEngine, threshold: float) -> dict:
    return match_and_score_list(
        pred_tx.get("medical", []),
        gold_tx.get("medical", []),
        name_key="name",
        engine=engine, threshold=threshold,
    )


def score_conservative(pred_tx: dict, gold_tx: dict,
                        engine: EmbeddingEngine,
                        threshold: float) -> dict:
    pred_c = pred_tx.get("conservative") or {}
    gold_c = gold_tx.get("conservative") or {}

    method_result = embedding_coverage_f1(
        pred_c.get("conservative_method"),
        gold_c.get("conservative_method"),
        engine=engine, threshold=threshold,
    )
    lifestyle_result = embedding_coverage_f1(
        pred_c.get("lifestyle_habit_modifications"),
        gold_c.get("lifestyle_habit_modifications"),
        engine=engine, threshold=threshold,
    )

    field_avg = avg([method_result["f1"], lifestyle_result["f1"]])

    return {
        "conservative_method":           method_result,
        "lifestyle_habit_modifications": lifestyle_result,
        "field_avg":                     field_avg,
    }


def score_follow_up(pred_fu: list, gold_fu: list,
                    engine: EmbeddingEngine, threshold: float) -> dict:
    """
    Positional matching preserved from original — index i in pred matches index i
    in gold. Embedding similarity replaces chrF for sub-field scoring.
    """
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
        sub = score_subfields(p, g, engine, threshold=threshold)
        item_avg = avg(sub.values())
        all_avgs.append(item_avg)
        per_item.append({"subfield_scores": sub, "item_avg": item_avg})

    return {"per_item": per_item, "field_avg": avg(all_avgs)}


def score_referral(pred_ref: list, gold_ref: list,
                   engine: EmbeddingEngine, threshold: float) -> dict:
    return match_and_score_list(
        pred_ref or [], gold_ref or [],
        name_key="specialty_or_doctor",
        engine=engine, threshold=threshold,
    )


def score_patient_education(pred_val, gold_val,
                             engine: EmbeddingEngine,
                             threshold: float) -> dict:
    score = engine.similarity(
        str(pred_val).strip() if pred_val else None,
        str(gold_val).strip() if gold_val else None,
        threshold=threshold,
    )
    return {"field_avg": score}


def score_when_to_seek(pred_val, gold_val,
                       engine: EmbeddingEngine,
                       threshold: float) -> dict:
    score = engine.similarity(
        str(pred_val).strip() if pred_val else None,
        str(gold_val).strip() if gold_val else None,
        threshold=threshold,
    )
    return {"field_avg": score}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Confusion detection  (token_coverage → embedding cosine)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_field_text(field_name: str, extraction: dict) -> str:
    """
    Extract all text from a named field into a single lowercased string.
    Identical mapping to original script — just returns a flat string
    that we then embed as a whole for confusion scoring.
    """
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
                     engine: EmbeddingEngine,
                     confusion_threshold: float) -> float:
    """
    Measure semantic similarity between gold's field_A and pred's field_B.

    A high score → the model placed field_A's content into field_B instead.
    Uses full-field embedding similarity (embed the whole concatenated text).
    Returns 0.0 if either field is empty (nothing to confuse).
    """
    gold_a = _extract_field_text(gold_field_a, gold_extraction)
    pred_b = _extract_field_text(pred_field_b, pred_extraction)

    if not gold_a or not pred_b:
        return 0.0

    # For confusion we use raw cosine (no threshold gate) so we get a
    # continuous signal even for borderline cases
    gold_vec = engine.get_embedding(gold_a)
    pred_vec = engine.get_embedding(pred_b)

    if gold_vec is None or pred_vec is None:
        # Strings might be too long or not in cache; embed on-the-fly
        vecs = engine._embed_batch([gold_a, pred_b])
        gold_vec, pred_vec = vecs[0], vecs[1]

    cosine = float(np.dot(gold_vec, pred_vec))
    return round(max(0.0, cosine), 4)


# All confusion pairs — identical to original
CONFUSION_PAIRS = [
    ("medical_routes",          "medical_dosage_forms",         "medical: route ↔ dosage_form"),
    ("imaging_modality",        "imaging_site",                 "imaging: modality ↔ site"),
    ("follow_up",               "referral",                     "follow_up ↔ referral"),
    ("labs",                    "monitoring",                   "labs ↔ monitoring"),
    ("imaging",                 "monitoring",                   "imaging ↔ monitoring"),
    ("prevention",              "patient_education",            "prevention ↔ patient_education"),
    ("conservative_method",     "medical_names",                "conservative_method ↔ medical"),
    ("medical_names",           "external_equipment",           "medical ↔ external_equipment"),
    ("imaging",                 "labs",                         "imaging ↔ labs"),
    ("tissue_sampling",         "imaging",                      "tissue_sampling ↔ imaging"),
    ("imaging",                 "referral",                     "imaging ↔ referral"),
    ("labs",                    "referral",                     "labs ↔ referral"),
    ("prevention",              "conservative_method",          "prevention ↔ conservative_method"),
    ("prevention",              "lifestyle_habit_modifications", "prevention ↔ lifestyle"),
    ("prevention",              "complementary_therapies",      "prevention ↔ complementary_therapies"),
    ("conservative_method",     "lifestyle_habit_modifications", "conservative_method ↔ lifestyle"),
    ("conservative_method",     "complementary_therapies",      "conservative_method ↔ complementary_therapies"),
    ("lifestyle_habit_modifications", "complementary_therapies","lifestyle ↔ complementary_therapies"),
]


def detect_confusion(pred_extraction: dict, gold_extraction: dict,
                     engine: EmbeddingEngine,
                     threshold: float = 0.5) -> dict:
    """Run all confusion pair checks for a single case."""
    pair_results = []
    flagged      = []

    for field_a, field_b, description in CONFUSION_PAIRS:
        score_a_in_b = _confusion_score(field_a, field_b,
                                        gold_extraction, pred_extraction,
                                        engine, threshold)
        score_b_in_a = _confusion_score(field_b, field_a,
                                        gold_extraction, pred_extraction,
                                        engine, threshold)

        is_flagged = score_a_in_b >= threshold or score_b_in_a >= threshold

        result = {
            "pair":    description,
            "field_a": field_a,
            "field_b": field_b,
            "a_in_b":  round(score_a_in_b, 4),
            "b_in_a":  round(score_b_in_a, 4),
            "flagged": is_flagged,
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
    """Aggregate per-case confusion results into dataset-level statistics."""
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
            "flagged_cases":   data["flagged_count"],
            "total_cases":     total,
            "flagged_pct":     round(data["flagged_count"] / total * 100, 1) if total else 0.0,
            "avg_a_in_b":      avg(data["a_in_b"]),
            "avg_b_in_a":      avg(data["b_in_a"]),
            "flagged_row_ids": data["flagged_row_ids"],
        }

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Case-level and dataset-level evaluation
# ══════════════════════════════════════════════════════════════════════════════

def score_case(pred_extraction: dict, gold_extraction: dict,
               engine: EmbeddingEngine,
               threshold: float = 0.65,
               run_confusion: bool = False,
               confusion_threshold: float = 0.5) -> dict:
    """Score one case across all extraction fields."""
    pred_inv = pred_extraction.get("investigations", {}) or {}
    gold_inv = gold_extraction.get("investigations", {}) or {}
    pred_tx  = pred_extraction.get("treatment", {}) or {}
    gold_tx  = gold_extraction.get("treatment", {}) or {}

    fields = {
        "labs":       score_labs(pred_inv, gold_inv, engine, threshold),
        "imaging":    score_imaging(pred_inv, gold_inv, engine, threshold),
        "monitoring": score_monitoring(
                          pred_extraction.get("monitoring"),
                          gold_extraction.get("monitoring"),
                          engine, threshold),
        "medical":    score_medical(pred_tx, gold_tx, engine, threshold),
        "conservative": score_conservative(pred_tx, gold_tx, engine, threshold),
        "follow_up":  score_follow_up(
                          pred_extraction.get("follow_up"),
                          gold_extraction.get("follow_up"),
                          engine, threshold),
        "referral":   score_referral(
                          pred_extraction.get("referral"),
                          gold_extraction.get("referral"),
                          engine, threshold),
        "patient_education": score_patient_education(
                          pred_extraction.get("patient_education"),
                          gold_extraction.get("patient_education"),
                          engine, threshold),
        "when_to_seek_medical_care": score_when_to_seek(
                          pred_extraction.get("when_to_seek_medical_care"),
                          gold_extraction.get("when_to_seek_medical_care"),
                          engine, threshold),
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
            engine=engine,
            threshold=confusion_threshold,
        )

    return result


def evaluate_dataset(pred_list: list, gold_list: list,
                     engine: EmbeddingEngine,
                     threshold: float = 0.65,
                     run_confusion: bool = False,
                     confusion_threshold: float = 0.5) -> dict:
    """Evaluate the full dataset."""
    gold_by_id = {item["row_index"]: item for item in gold_list}

    case_results      = []
    field_accumulator = defaultdict(list)
    case_confusions   = []

    for pred_item in tqdm(pred_list, desc="Evaluating cases", unit="case"):
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
            engine=engine,
            threshold=threshold,
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
# SECTION 9 — Reporting  (identical structure to original)
# ══════════════════════════════════════════════════════════════════════════════

_COVERAGE_FIELDS        = {"conservative"}
_CONSERVATIVE_SUBFIELDS = {"conservative_method", "lifestyle_habit_modifications"}


def print_report(results: dict,
                 run_confusion: bool = False,
                 confusion_threshold: float = 0.5,
                 threshold: float = 0.65) -> None:
    """Print a human-readable evaluation report to stdout."""

    W = 72

    print()
    print("=" * W)
    print("  MEDICAL EXTRACTION EVALUATION REPORT  (MedGemma Embeddings)")
    print("=" * W)
    print(f"  Cases evaluated  : {len(results['per_case'])}")
    print(f"  Scoring metric   : Cosine similarity (MedGemma-4b-it embeddings)")
    print(f"  Similarity thresh: {threshold}  (below → 0.0)")
    print(f"  List fields      : Embedding Coverage F1 (soft max-cosine matching)")
    print(f"  Structured items : Greedy cosine matching + sub-field cosine")
    if run_confusion:
        print(f"  Confusion detect : ENABLED  (threshold={confusion_threshold})")
    print("=" * W)

    # ── Dataset-level field scores ─────────────────────────────────────────
    print(f"\n  {'Field':<35} {'Score':>8}  Metric")
    print(f"  {'─'*35} {'─'*8}  {'─'*25}")
    for field, score in results["dataset_field_scores"].items():
        metric = "Embedding Coverage F1" if field in _COVERAGE_FIELDS \
                 else "Cosine Similarity"
        print(f"  {field:<35} {score:>8.4f}  {metric}")
    print(f"  {'─'*35} {'─'*8}")
    print(f"  {'OVERALL':<35} {results['dataset_overall']:>8.4f}")
    print()

    # ── Conservative field breakdown ──────────────────────────────────────
    print("─" * W)
    print("  CONSERVATIVE TREATMENT  (Embedding Coverage F1 detail)")
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
        sorted_pairs = sorted(summary.items(),
                              key=lambda x: x[1]["flagged_pct"], reverse=True)

        for desc, data in sorted_pairs:
            flagged_pct = data["flagged_pct"]
            avg_a_in_b  = data["avg_a_in_b"]
            avg_b_in_a  = data["avg_b_in_a"]
            flagged     = data["flagged_cases"]
            total       = data["total_cases"]
            row_ids     = data.get("flagged_row_ids", [])
            marker = "  !" if flagged > 0 else "   "
            print(f"{marker} {desc:<48} {flagged_pct:>5.1f}%  "
                  f"{avg_a_in_b:>6.4f}  {avg_b_in_a:>6.4f}  "
                  f"{flagged:>3}/{total:<3}")
            if row_ids:
                ids_str = ", ".join(str(r) for r in row_ids)
                print(f"       row_ids: [{ids_str}]")

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
# SECTION 10 — CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate medical treatment plan extraction using MedGemma embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_extraction_medgemma.py --pred pred.json --gold gold.json
  python evaluate_extraction_medgemma.py --pred pred.json --gold gold.json \\
      --output results.json --embedding-model google/medgemma-4b-it
  python evaluate_extraction_medgemma.py --pred pred.json --gold gold.json \\
      --confusion --confusion-threshold 0.5
  python evaluate_extraction_medgemma.py --pred pred.json --gold gold.json \\
      --cache-embeddings my_cache.pt
  python evaluate_extraction_medgemma.py --pred pred.json --gold gold.json \\
      --batch-size 64 --device cuda
        """,
    )
    parser.add_argument("--pred",                 required=True,
                        help="Path to model predictions JSON")
    parser.add_argument("--gold",                 required=True,
                        help="Path to annotator gold JSON")
    parser.add_argument("--output",               default=None,
                        help="Optional path to save full results as JSON")
    parser.add_argument("--embedding-model",      default="google/medgemma-4b-it",
                        help="HuggingFace model name or local path (default: google/medgemma-4b-it)")
    parser.add_argument("--batch-size",           type=int, default=32,
                        help="Embedding batch size (default: 32)")
    parser.add_argument("--device",               default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for inference (default: auto)")
    parser.add_argument("--threshold",            type=float, default=0.65,
                        help="Cosine similarity threshold — below this scores 0.0 (default: 0.65)")
    parser.add_argument("--confusion",            action="store_true",
                        help="Enable field confusion detection")
    parser.add_argument("--confusion-threshold",  type=float, default=0.5,
                        help="Cosine threshold to flag a confusion (default: 0.5)")
    parser.add_argument("--cache-embeddings",     default=None,
                        help="Path to save/load embedding cache (.pt file). "
                             "If file exists, load it; otherwise embed and save.")
    args = parser.parse_args()

    print(f"\nLoading predictions  : {args.pred}")
    print(f"Loading gold         : {args.gold}")
    print(f"Embedding model      : {args.embedding_model}")
    print(f"Device               : {args.device}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Similarity threshold : {args.threshold}")
    if args.confusion:
        print(f"Confusion detection  : enabled  (threshold={args.confusion_threshold})")

    with open(args.pred) as f:
        pred_data = json.load(f)
    with open(args.gold) as f:
        gold_data = json.load(f)

    # ── Load or build embedding engine ────────────────────────────────────
    engine = EmbeddingEngine(
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    import os
    cache_exists = args.cache_embeddings and os.path.exists(args.cache_embeddings)

    if cache_exists:
        print(f"\nLoading embedding cache from: {args.cache_embeddings}")
        engine.load_cache(args.cache_embeddings)
    else:
        print("\nCollecting all strings from dataset for batch embedding ...")
        all_strings = collect_all_strings(pred_data, gold_data)
        print(f"Total unique strings to embed: {len(all_strings)}")
        engine.build_cache(all_strings)

        if args.cache_embeddings:
            engine.save_cache(args.cache_embeddings)

    # ── Run evaluation ────────────────────────────────────────────────────
    results = evaluate_dataset(
        pred_data, gold_data,
        engine=engine,
        threshold=args.threshold,
        run_confusion=args.confusion,
        confusion_threshold=args.confusion_threshold,
    )

    print_report(
        results,
        run_confusion=args.confusion,
        confusion_threshold=args.confusion_threshold,
        threshold=args.threshold,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full results saved to: {args.output}\n")


if __name__ == "__main__":
    main()



"""
python semantic_eval_script.py \
    --pred extracted_plan_google_gemini-2.0-flash-001_pydantic_normalized.json \
    --gold extractions_annotator_2_normalized.json \
    --output annotator_2_gemini_2_flash_semantic_eval_results_medgemma.json \
    --embedding-model google/medgemma-4b-it \
    --batch-size 32 \
    --device cuda \
    --threshold 0.65 \
    --confusion \
    --confusion-threshold 0.5 \
    --cache-embeddings dataset_embeddings.pt
"""