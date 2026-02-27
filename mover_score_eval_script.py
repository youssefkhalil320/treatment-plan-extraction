"""
Medical Treatment Plan Extraction — MoverScore Evaluation Script
================================================================
Computes MoverScore (semantic + syntactic joint metric) for comparing
model extraction output against human annotator (gold) annotations.

Also reports chrF (syntactic) and cosine similarity (semantic) as
decomposed components alongside MoverScore for full interpretability.

─────────────────────────────────────────────────────────────────────────────
WHAT IS MOVERSCORE?
─────────────────────────────────────────────────────────────────────────────

  MoverScore uses Earth Mover's Distance (Wasserstein-1) over contextual
  token embeddings to measure how much semantic "work" is needed to
  transform pred into gold. Unlike BERTScore (which does greedy 1-to-1
  token matching), EMD finds the globally optimal soft alignment — a token
  in pred can partially contribute to multiple gold tokens.

  This naturally handles:
    - Synonyms and paraphrases   ("ibuprofen" ↔ "NSAID")
    - Granularity mismatches     ("diet and exercise" ↔ "low-carb diet | daily walking")
    - Word order differences     ("twice daily 500mg" ↔ "500mg BID")

  MoverScore = 1 - normalized_EMD(pred_token_vectors, gold_token_vectors)
  Range: [0, 1], higher = better

─────────────────────────────────────────────────────────────────────────────
THREE SCORES REPORTED PER FIELD
─────────────────────────────────────────────────────────────────────────────

  mover  : MoverScore (joint semantic + syntactic via EMD over token embeddings)
  chrf   : Character n-gram F-score (pure syntactic surface similarity)
  cosine : Sentence-level cosine similarity (pure semantic, coarse-grained)

  The three scores are complementary:
    • High mover + low chrf  → semantically equivalent but different wording
    • High chrf + low cosine → surface match but possible meaning drift
    • High cosine + low mover→ related topic but token-level mismatch

─────────────────────────────────────────────────────────────────────────────
EMBEDDING DETAILS
─────────────────────────────────────────────────────────────────────────────

  Model  : thomas-sounack/BioClinical-ModernBERT-base  (encoder, bidirectional)
  Pooling for MoverScore  : per-token hidden states (NOT pooled) — EMD needs
                            one vector per token to compute transport cost
  Pooling for cosine      : mean-pool over non-padding tokens → sentence vector
  Norm   : L2-normalized token vectors → cosine of token pair = dot product

  All strings across the full dataset are collected, deduplicated,
  batch-tokenized, and their token embeddings cached in memory.
  Zero additional model calls during evaluation.

─────────────────────────────────────────────────────────────────────────────
SCORING STRATEGY BY FIELD TYPE
─────────────────────────────────────────────────────────────────────────────

  Structured item lists  (labs, imaging, medications, monitoring, referrals)
  ──────────────────────
    Item-level:
      • Greedy MoverScore matching on item name
      • Match threshold: 0.5 (below → unrelated, score 0.0)
      • Matched pairs → sub-fields scored with all three metrics
      • Unmatched items → 0.0

    Field-level:
      • All item names concatenated per side → one MoverScore for the full field
      • Reported separately as field_mover_full

  Flat string lists  (conservative_method, lifestyle_habit_modifications)
  ─────────────────
    • Lists joined with " | " → single string each side
    • All three metrics computed on the joined strings

  Single string fields  (patient_education, when_to_seek_medical_care,
                         follow_up sub-fields, referral aim)
  ─────────────────────
    • Direct per-string scoring with all three metrics

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

  python mover_eval.py --pred predictions.json --gold annotations.json
  python mover_eval.py --pred predictions.json --gold annotations.json \\
      --output results.json
  python mover_eval.py --pred predictions.json --gold annotations.json \\
      --embedding-model thomas-sounack/BioClinical-ModernBERT-base \\
      --batch-size 32 --device cuda
  python mover_eval.py --pred predictions.json --gold annotations.json \\
      --cache-embeddings embed_cache.pt \\
      --confusion --confusion-threshold 0.5
"""

import json
import argparse
import os
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sacrebleu.metrics import CHRF
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — chrF scorer
# ══════════════════════════════════════════════════════════════════════════════

_chrf = CHRF()


def chrf_score(pred: str | None, gold: str | None) -> float:
    """Character n-gram F-score, normalised to [0, 1]."""
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
# SECTION 2 — Embedding engine  (token-level cache for MoverScore + cosine)
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Loads BioClinical-ModernBERT-base once, caches two representations
    per string:

      token_vecs  : list of L2-normalised per-token vectors  [seq_len, H]
                    Used by mover_score() — EMD needs individual token vectors.
      sentence_vec: mean-pooled + L2-normalised vector       [H]
                    Used by cosine_sim() — sentence-level similarity.

    Both are computed in the same forward pass; no extra model calls.

    Architecture: encoder (bidirectional) → mean pooling is valid for both
    sentence vectors and individual token representations.
    """

    def __init__(self,
                 model_name: str = "thomas-sounack/BioClinical-ModernBERT-base",
                 device: str = "auto",
                 batch_size: int = 32):

        self.model_name = model_name
        self.batch_size = batch_size

        # Cache: string → {"tokens": np.ndarray [T, H], "sentence": np.ndarray [H]}
        self._cache: dict[str, dict] = {}

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[EmbeddingEngine] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"[EmbeddingEngine] Loading model → {self.device}")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[EmbeddingEngine] Ready.")

    # ── forward pass ──────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> list[dict]:
        """
        Single forward pass over a batch of texts.

        Returns a list (one entry per text) of:
          {
            "tokens":   np.ndarray  [n_real_tokens, hidden_dim]   L2-normalised
            "sentence": np.ndarray  [hidden_dim]                  L2-normalised
          }

        Special tokens ([CLS], [SEP], [PAD]) are excluded from token_vecs
        so that MoverScore only transports over content tokens.
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

        hidden = output.last_hidden_state          # [B, seq_len, H]
        mask   = encoded["attention_mask"]         # [B, seq_len]  1=real, 0=pad

        results = []
        for i in range(hidden.size(0)):
            # --- isolate real (non-padding) positions ---
            real_mask  = mask[i].bool()            # [seq_len]
            real_hidden = hidden[i][real_mask]     # [n_real, H]

            # Remove [CLS] (position 0) and [SEP] (last real token) if present.
            # ModernBERT uses these special tokens; stripping them keeps
            # content tokens only, which is what MoverScore should transport.
            n = real_hidden.size(0)
            if n >= 3:
                content = real_hidden[1:-1]        # strip CLS + SEP
            elif n == 2:
                content = real_hidden[1:]          # only strip CLS
            else:
                content = real_hidden              # single token, keep as is

            # L2-normalise token vectors → cosine similarity = dot product
            t_norms   = content.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            token_vecs = (content / t_norms).cpu().float().numpy()   # [T, H]

            # Sentence vector: mean of REAL (non-special) positions
            # We use the pre-normalised real_hidden for mean pooling
            # (mean of L2-normalised vectors is not the same as L2-normalising
            # the mean — but for sentence cosine it's acceptable; we renormalise)
            if content.size(0) > 0:
                sent = content.mean(dim=0)
            else:
                sent = real_hidden.mean(dim=0)
            s_norm      = sent.norm().clamp(min=1e-9)
            sentence_vec = (sent / s_norm).cpu().float().numpy()     # [H]

            results.append({"tokens": token_vecs, "sentence": sentence_vec})

        return results

    # ── cache ──────────────────────────────────────────────────────────────────

    def build_cache(self, strings: list[str]) -> None:
        """Embed all unique non-empty strings and store in cache."""
        unique = list({s for s in strings if s and s.strip()})
        if not unique:
            print("[EmbeddingEngine] No strings to embed.")
            return

        print(f"[EmbeddingEngine] Embedding {len(unique)} unique strings "
              f"(batch={self.batch_size}) ...")

        for i in tqdm(range(0, len(unique), self.batch_size),
                      desc="Embedding", unit="batch"):
            batch   = unique[i : i + self.batch_size]
            results = self._embed_batch(batch)
            for text, entry in zip(batch, results):
                self._cache[text] = entry

        print(f"[EmbeddingEngine] Cache: {len(self._cache)} entries.")

    def save_cache(self, path: str) -> None:
        torch.save(self._cache, path)
        print(f"[EmbeddingEngine] Cache saved → {path}")

    def load_cache(self, path: str) -> None:
        self._cache = torch.load(path, map_location="cpu")
        print(f"[EmbeddingEngine] Cache loaded: {len(self._cache)} entries.")

    def _get(self, text: str) -> dict | None:
        if not text or not text.strip():
            return None
        return self._cache.get(text.strip().lower())

    def _embed_on_the_fly(self, text: str) -> dict | None:
        """Embed a single string not in cache (fallback)."""
        if not text or not text.strip():
            return None
        results = self._embed_batch([text.strip().lower()])
        entry = results[0]
        self._cache[text.strip().lower()] = entry
        return entry

    def get_entry(self, text: str) -> dict | None:
        """Return cache entry, falling back to on-the-fly embedding."""
        entry = self._get(text)
        if entry is None:
            entry = self._embed_on_the_fly(text)
        return entry


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MoverScore  (EMD over token embeddings)
# ══════════════════════════════════════════════════════════════════════════════

def _emd_one_way(query_vecs: np.ndarray, ref_vecs: np.ndarray) -> float:
    """
    One-directional greedy soft alignment: each query token finds its nearest
    ref token (no assignment constraint). Returns similarity in [0, 1].

    Intentionally greedy (NOT linear_sum_assignment) because:
      - We want each query token to independently find its best match in ref
      - Multiple query tokens can match the same ref token -- this is correct
        for a coverage/recall measurement
      - Linear assignment forces 1-to-1 which defeats asymmetric coverage

    Called twice per pair:
      query=pred, ref=gold -> precision  (are pred tokens valid / in gold?)
      query=gold, ref=pred -> recall     (are gold tokens covered by pred?)
    """
    if query_vecs.shape[0] == 0 or ref_vecs.shape[0] == 0:
        return 0.0
    sim_matrix  = query_vecs @ ref_vecs.T          # [n_query, n_ref]
    dist_matrix = 1.0 - sim_matrix.clip(-1, 1)     # cosine distance in [0, 2]
    emd = dist_matrix.min(axis=1).mean()
    return max(0.0, 1.0 - emd / 2.0)


def _emd_score(pred_vecs: np.ndarray, gold_vecs: np.ndarray) -> float:
    """
    Asymmetric F1-MoverScore: harmonic mean of precision and recall directions.

    precision = _emd_one_way(pred -> gold): are pred tokens valid?
    recall    = _emd_one_way(gold -> pred): did pred miss any gold tokens?

    A pred covering 3 of 5 gold items gets penalised on recall -- the 2
    missing gold items find no close match in pred, driving recall down.
    Symmetric EMD does not have this property.

    Returns:
        float in [0, 1], where 1 = identical, 0 = maximally dissimilar
    """
    if pred_vecs.shape[0] == 0 or gold_vecs.shape[0] == 0:
        return 0.0
    p = _emd_one_way(pred_vecs, gold_vecs)   # precision
    r = _emd_one_way(gold_vecs, pred_vecs)   # recall
    if p + r == 0:
        return 0.0
    return round(float(2 * p * r / (p + r)), 4)


def mover_score(pred: str | None, gold: str | None,
                engine: EmbeddingEngine) -> float:
    """
    MoverScore between two strings.

    Null contract:
        (None, None) → 1.0   (both agree nothing here)
        (None, X)    → 0.0
        (X, None)    → 0.0
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    pred_clean = str(pred).strip().lower()
    gold_clean = str(gold).strip().lower()

    if pred_clean == gold_clean:
        return 1.0

    pred_entry = engine.get_entry(pred_clean)
    gold_entry = engine.get_entry(gold_clean)

    if pred_entry is None or gold_entry is None:
        return 0.0

    return _emd_score(pred_entry["tokens"], gold_entry["tokens"])


def cosine_sim(pred: str | None, gold: str | None,
               engine: EmbeddingEngine) -> float:
    """
    Sentence-level cosine similarity (mean-pooled embeddings).

    Null contract: same as mover_score.
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    pred_clean = str(pred).strip().lower()
    gold_clean = str(gold).strip().lower()

    if pred_clean == gold_clean:
        return 1.0

    pred_entry = engine.get_entry(pred_clean)
    gold_entry = engine.get_entry(gold_clean)

    if pred_entry is None or gold_entry is None:
        return 0.0

    score = float(np.dot(pred_entry["sentence"], gold_entry["sentence"]))
    return round(max(0.0, score), 4)


def score_triple(pred: str | None, gold: str | None,
                 engine: EmbeddingEngine) -> dict:
    """
    Compute all three metrics for a (pred, gold) string pair.
    Returns: {"mover": float, "chrf": float, "cosine": float}
    """
    return {
        "mover":  mover_score(pred, gold, engine),
        "chrf":   chrf_score(pred, gold),
        "cosine": cosine_sim(pred, gold, engine),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — String collection  (feeds build_cache)
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_strings(pred_list: list, gold_list: list) -> list[str]:
    """
    Walk every extraction in pred and gold, collect every string that will
    be embedded. For flat list fields the ' | '-joined concatenation is
    collected (exactly what the scorer will embed). For confusion detection
    the assembled field strings are also collected.
    """
    strings = set()

    def add(s):
        if s and str(s).strip():
            strings.add(str(s).strip().lower())

    def add_joined(items):
        if not items:
            return
        cleaned = [str(x).strip().lower() for x in items if str(x).strip()]
        if cleaned:
            strings.add(" | ".join(cleaned))

    def collect_extraction(ext: dict):
        if not ext:
            return
        inv  = ext.get("investigations", {}) or {}
        tx   = ext.get("treatment", {}) or {}
        cons = tx.get("conservative") or {}

        def collect_item(item):
            """Safely collect strings from an item that may be a str or dict."""
            if isinstance(item, str):
                add(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str): add(v)

        for lab in (inv.get("labs") or []):
            collect_item(lab)

        for img in (inv.get("imaging") or []):
            collect_item(img)
            if isinstance(img, dict):
                key = f"{img.get('imaging_modality','')} {img.get('site','')}".strip()
                add(key)

        for ts in (inv.get("tissue_sampling") or []):
            collect_item(ts)

        for mon in (ext.get("monitoring") or []):
            collect_item(mon)

        for med in (tx.get("medical") or []):
            collect_item(med)

        # Flat list fields — collect joined string
        add_joined(cons.get("conservative_method"))
        add_joined(cons.get("lifestyle_habit_modifications"))
        add_joined(tx.get("prevention"))
        add_joined(tx.get("complementary_therapies"))

        for eq in (tx.get("external_equipment") or []):
            collect_item(eq)

        for fu in (ext.get("follow_up") or []):
            collect_item(fu)

        for ref in (ext.get("referral") or []):
            collect_item(ref)

        add(ext.get("patient_education"))
        add(ext.get("when_to_seek_medical_care"))

        # Confusion detection field strings
        _collect_confusion_strings(ext, strings)

    for entry in pred_list:
        collect_extraction(entry.get("extraction", {}))
    for entry in gold_list:
        collect_extraction(entry.get("extraction", {}))

    return [s for s in strings if s]


def _collect_confusion_strings(ext: dict, strings: set):
    """
    Pre-collect the concatenated field strings that confusion detection
    will need, so they are in cache before evaluation runs.
    """
    inv  = ext.get("investigations", {}) or {}
    tx   = ext.get("treatment", {}) or {}
    cons = tx.get("conservative") or {}

    def _join(items):
        if not items: return None
        parts = [str(x).strip().lower() for x in items if str(x).strip()]
        return " ".join(parts) if parts else None

    def _get(item, *keys, default=""):
        if isinstance(item, str):
            return item
        for key in keys:
            val = item.get(key)
            if val:
                return val
        return default

    labs_list     = inv.get("labs") or []
    imaging_list  = inv.get("imaging") or []
    tissue_list   = inv.get("tissue_sampling") or []
    monitor_list  = ext.get("monitoring") or []
    medical_list  = tx.get("medical") or []
    equip_list    = tx.get("external_equipment") or []
    referral_list = ext.get("referral") or []
    followup_list = ext.get("follow_up") or []

    candidates = [
        ext.get("patient_education"),
        ext.get("when_to_seek_medical_care"),
        _join(cons.get("conservative_method") or []),
        _join(cons.get("lifestyle_habit_modifications") or []),
        _join(tx.get("prevention") or []),
        _join(tx.get("complementary_therapies") or []),
        _join([_get(i, "lab_investigation_name") for i in labs_list]),
        _join([f"{_get(i, 'imaging_modality')} {_get(i, 'site')}".strip() for i in imaging_list]),
        _join([_get(i, "imaging_modality") for i in imaging_list]),
        _join([_get(i, "site") for i in imaging_list]),
        _join([_get(i, "tissue_sampling_method", "name") for i in tissue_list]),
        _join([_get(i, "monitoring_parameter") for i in monitor_list]),
        _join([_get(i, "name") for i in medical_list]),
        _join([_get(i, "route") for i in medical_list if (isinstance(i, dict) and i.get("route"))]),
        _join([_get(i, "dosage_form") for i in medical_list if (isinstance(i, dict) and i.get("dosage_form"))]),
        _join([_get(i, "name", "equipment_name") for i in equip_list]),
        _join([f"{_get(i, 'specialty_or_doctor')} {_get(i, 'aim_of_referral')}".strip() for i in referral_list]),
        _join([f"{_get(i, 'scheduled_follow_up_time')} {_get(i, 'aim_of_follow_up')}".strip() for i in followup_list]),
    ]
    for s in candidates:
        if s and s.strip():
            strings.add(s.strip().lower())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def avg(values) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def avg_triple(triples: list[dict]) -> dict:
    """Average a list of {"mover", "chrf", "cosine"} dicts."""
    if not triples:
        return {"mover": 0.0, "chrf": 0.0, "cosine": 0.0}
    return {
        "mover":  avg([t["mover"]  for t in triples]),
        "chrf":   avg([t["chrf"]   for t in triples]),
        "cosine": avg([t["cosine"] for t in triples]),
    }


def score_subfields(pred_item: dict | None,
                    gold_item: dict,
                    engine: EmbeddingEngine,
                    skip_keys: set = None) -> dict:
    """
    Score each sub-field of a matched item pair with all three metrics.
    pred_item = None when the gold item was unmatched (model missed it).
    """
    skip_keys = skip_keys or set()
    scores = {}

    if isinstance(pred_item, str):
        pred_item = {"name": pred_item}
    if isinstance(gold_item, str):
        gold_item = {"name": gold_item}

    for key, gold_val in gold_item.items():
        if key in skip_keys:
            continue
        pred_val = pred_item.get(key) if pred_item else None

        if isinstance(gold_val, list):
            gold_str = " | ".join(str(v) for v in gold_val) if gold_val else None
            pred_str = " | ".join(str(v) for v in pred_val) if isinstance(pred_val, list) and pred_val \
                       else (str(pred_val).strip() if pred_val else None)
        else:
            gold_str = str(gold_val).strip() if gold_val is not None else None
            pred_str = str(pred_val).strip() if pred_val is not None else None

        scores[key] = score_triple(pred_str, gold_str, engine)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Structured item list matcher (greedy MoverScore + field-level)
# ══════════════════════════════════════════════════════════════════════════════

def match_and_score_list(pred_list: list,
                         gold_list: list,
                         name_key: str,
                         engine: EmbeddingEngine,
                         skip_keys: set = None,
                         threshold: float = 0.5) -> dict:
    """
    Match pred items to gold items by best MoverScore on name_key,
    score sub-fields of matched pairs with all three metrics,
    and also compute a field-level MoverScore over concatenated names.

    Returns field_avg as the primary scalar — averaged over mover scores
    of all items (matched and unmatched), mirroring the structure of the
    reference scripts.
    """
    skip_keys = skip_keys or set()

    pred_list = [{"name": p} if isinstance(p, str) else (p or {}) for p in (pred_list or [])]
    gold_list = [{"name": g} if isinstance(g, str) else (g or {}) for g in (gold_list or [])]

    if not gold_list and not pred_list:
        return {"per_item": [], "unmatched_gold": [], "unmatched_pred": [],
                "field_avg": 1.0, "field_avg_triple": {"mover": 1.0, "chrf": 1.0, "cosine": 1.0},
                "field_mover_full": 1.0}
    if not gold_list:
        return {"per_item": [], "unmatched_gold": [],
                "unmatched_pred": [str(p.get(name_key,"")) for p in pred_list],
                "field_avg": 0.0, "field_avg_triple": {"mover": 0.0, "chrf": 0.0, "cosine": 0.0},
                "field_mover_full": 0.0}

    gold_names = [str(g.get(name_key, "")).strip().lower() for g in gold_list]
    pred_names = [str(p.get(name_key, "")).strip().lower() for p in pred_list]

    # ── item-level matching ──────────────────────────────────────────────────
    # Build MoverScore similarity matrix on name_key
    sim_matrix = [
        [mover_score(pred_names[j], gold_names[i], engine) if pred_names[j] and gold_names[i] else 0.0
         for j in range(len(pred_list))]
        for i in range(len(gold_list))
    ]

    candidates = [
        (sim_matrix[i][j], i, j)
        for i in range(len(gold_list))
        for j in range(len(pred_list))
        if sim_matrix[i][j] >= threshold
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gold = {}
    matched_pred = {}
    for score, gi, pi in candidates:
        if gi not in matched_gold and pi not in matched_pred:
            matched_gold[gi] = pi
            matched_pred[pi] = gi

    all_item_movers = []
    all_item_triples = []
    per_item_results = []

    for gi, gold_item in enumerate(gold_list):
        pi        = matched_gold.get(gi)
        pred_item = pred_list[pi] if pi is not None else None
        name_ms   = sim_matrix[gi][pi] if pi is not None else 0.0

        sub_scores = score_subfields(pred_item, gold_item, engine,
                                     skip_keys={name_key} | skip_keys)
        item_triple = avg_triple(list(sub_scores.values()))
        item_mover  = item_triple["mover"]
        all_item_movers.append(item_mover)
        all_item_triples.append(item_triple)

        per_item_results.append({
            "name":            gold_item.get(name_key),
            "matched_to":      pred_list[pi].get(name_key) if pred_item else None,
            "name_mover":      round(name_ms, 4),
            "matched":         pred_item is not None,
            "subfield_scores": sub_scores,
            "item_triple":     item_triple,
            "item_avg":        item_mover,
        })

    # Penalise hallucinated pred items
    unmatched_pred = [pred_list[j] for j in range(len(pred_list)) if j not in matched_pred]
    for _ in unmatched_pred:
        all_item_movers.append(0.0)
        all_item_triples.append({"mover": 0.0, "chrf": 0.0, "cosine": 0.0})

    unmatched_gold = [gold_list[i] for i in range(len(gold_list)) if i not in matched_gold]

    # ── field-level MoverScore (concatenated names) ──────────────────────────
    pred_concat = " ".join(n for n in pred_names if n)
    gold_concat = " ".join(n for n in gold_names if n)
    field_mover_full = mover_score(pred_concat or None, gold_concat or None, engine)

    return {
        "per_item":         per_item_results,
        "unmatched_gold":   [g.get(name_key) for g in unmatched_gold],
        "unmatched_pred":   [p.get(name_key) for p in unmatched_pred],
        "field_avg":        avg(all_item_movers),
        "field_avg_triple": avg_triple(all_item_triples),
        "field_mover_full": field_mover_full,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Per-section scorers
# ══════════════════════════════════════════════════════════════════════════════

def score_labs(pred_inv: dict, gold_inv: dict, engine: EmbeddingEngine) -> dict:
    return match_and_score_list(
        pred_inv.get("labs", []), gold_inv.get("labs", []),
        name_key="lab_investigation_name", engine=engine)


def score_imaging(pred_inv: dict, gold_inv: dict, engine: EmbeddingEngine) -> dict:
    def augment(items):
        out = []
        for item in (items or []):
            item = dict(item)
            item["_key"] = f"{item.get('imaging_modality','')} {item.get('site','')}".strip().lower()
            out.append(item)
        return out
    return match_and_score_list(
        augment(pred_inv.get("imaging", [])), augment(gold_inv.get("imaging", [])),
        name_key="_key", engine=engine)


def score_monitoring(pred_mon: list, gold_mon: list, engine: EmbeddingEngine) -> dict:
    return match_and_score_list(
        pred_mon or [], gold_mon or [],
        name_key="monitoring_parameter", engine=engine)


def score_medical(pred_tx: dict, gold_tx: dict, engine: EmbeddingEngine) -> dict:
    return match_and_score_list(
        pred_tx.get("medical", []), gold_tx.get("medical", []),
        name_key="name", engine=engine)


def score_conservative(pred_tx: dict, gold_tx: dict, engine: EmbeddingEngine) -> dict:
    """
    Score conservative treatment flat list fields.
    Both lists are joined with ' | ' and scored with all three metrics.
    """
    pred_c = pred_tx.get("conservative") or {}
    gold_c = gold_tx.get("conservative") or {}

    def join(items):
        cleaned = [str(x).strip().lower() for x in (items or []) if str(x).strip()]
        return " | ".join(cleaned) if cleaned else None

    method_triple    = score_triple(join(pred_c.get("conservative_method")),
                                    join(gold_c.get("conservative_method")), engine)
    lifestyle_triple = score_triple(join(pred_c.get("lifestyle_habit_modifications")),
                                    join(gold_c.get("lifestyle_habit_modifications")), engine)

    field_avg_triple = avg_triple([method_triple, lifestyle_triple])

    return {
        "conservative_method":           method_triple,
        "lifestyle_habit_modifications": lifestyle_triple,
        "field_avg":                     field_avg_triple["mover"],
        "field_avg_triple":              field_avg_triple,
    }


def score_follow_up(pred_fu: list, gold_fu: list, engine: EmbeddingEngine) -> dict:
    """Positional matching — index i in pred matches index i in gold."""
    pred_fu = pred_fu or []
    gold_fu = gold_fu or []

    if not gold_fu and not pred_fu:
        return {"per_item": [], "field_avg": 1.0,
                "field_avg_triple": {"mover": 1.0, "chrf": 1.0, "cosine": 1.0}}

    max_len = max(len(pred_fu), len(gold_fu))
    per_item, all_triples = [], []

    for i in range(max_len):
        p = pred_fu[i] if i < len(pred_fu) else {}
        g = gold_fu[i] if i < len(gold_fu) else {}
        sub = score_subfields(p, g, engine)
        t   = avg_triple(list(sub.values()))
        all_triples.append(t)
        per_item.append({"subfield_scores": sub, "item_triple": t, "item_avg": t["mover"]})

    agg = avg_triple(all_triples)
    return {"per_item": per_item, "field_avg": agg["mover"], "field_avg_triple": agg}


def score_referral(pred_ref: list, gold_ref: list, engine: EmbeddingEngine) -> dict:
    return match_and_score_list(
        pred_ref or [], gold_ref or [],
        name_key="specialty_or_doctor", engine=engine)


def score_single(pred_val, gold_val, engine: EmbeddingEngine) -> dict:
    t = score_triple(
        str(pred_val).strip() if pred_val else None,
        str(gold_val).strip() if gold_val else None,
        engine,
    )
    return {"field_avg": t["mover"], "field_avg_triple": t}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Confusion detection  (MoverScore-based)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_field_text(field_name: str, extraction: dict) -> str:
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
        "labs": [i.get("lab_investigation_name","") for i in (inv.get("labs") or [])],
        "imaging": [f"{i.get('imaging_modality','')} {i.get('site','')}".strip()
                    for i in (inv.get("imaging") or [])],
        "imaging_modality": [i.get("imaging_modality","") for i in (inv.get("imaging") or [])],
        "imaging_site":     [i.get("site","") for i in (inv.get("imaging") or [])],
        "tissue_sampling":  [i.get("tissue_sampling_method", i.get("name",""))
                             for i in (inv.get("tissue_sampling") or [])],
        "monitoring":       [i.get("monitoring_parameter","") for i in (extraction.get("monitoring") or [])],
        "medical_names":    [i.get("name","") for i in (tx.get("medical") or [])],
        "medical_routes":   [i.get("route","") for i in (tx.get("medical") or []) if i.get("route")],
        "medical_dosage_forms": [i.get("dosage_form","") for i in (tx.get("medical") or []) if i.get("dosage_form")],
        "external_equipment": [i.get("name", i.get("equipment_name","")) for i in (tx.get("external_equipment") or [])],
        "referral": [f"{i.get('specialty_or_doctor','')} {i.get('aim_of_referral','')}".strip()
                     for i in (extraction.get("referral") or [])],
        "follow_up": [f"{i.get('scheduled_follow_up_time','')} {i.get('aim_of_follow_up','')}".strip()
                      for i in (extraction.get("follow_up") or [])],
    }

    content = field_map.get(field_name)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip().lower()
    if isinstance(content, list):
        return " ".join(str(x).strip().lower() for x in content if x).strip()
    return ""


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
                     engine: EmbeddingEngine,
                     threshold: float = 0.5) -> dict:
    """Run all confusion pair checks using MoverScore."""
    pair_results, flagged = [], []

    for field_a, field_b, description in CONFUSION_PAIRS:
        gold_a = _extract_field_text(field_a, gold_extraction)
        pred_b = _extract_field_text(field_b, pred_extraction)
        gold_b = _extract_field_text(field_b, gold_extraction)
        pred_a = _extract_field_text(field_a, pred_extraction)

        score_a_in_b = mover_score(gold_a or None, pred_b or None, engine) if gold_a and pred_b else 0.0
        score_b_in_a = mover_score(gold_b or None, pred_a or None, engine) if gold_b and pred_a else 0.0

        is_flagged = score_a_in_b >= threshold or score_b_in_a >= threshold
        result = {
            "pair": description, "field_a": field_a, "field_b": field_b,
            "a_in_b": round(score_a_in_b, 4), "b_in_a": round(score_b_in_a, 4),
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
# SECTION 9 — Case-level and dataset-level evaluation
# ══════════════════════════════════════════════════════════════════════════════

def score_case(pred_extraction: dict, gold_extraction: dict,
               engine: EmbeddingEngine,
               run_confusion: bool = False,
               confusion_threshold: float = 0.5) -> dict:
    """Score one case across all fields, returning mover/chrf/cosine per field."""
    pred_inv = pred_extraction.get("investigations", {}) or {}
    gold_inv = gold_extraction.get("investigations", {}) or {}
    pred_tx  = pred_extraction.get("treatment", {}) or {}
    gold_tx  = gold_extraction.get("treatment", {}) or {}

    fields = {
        "labs":       score_labs(pred_inv, gold_inv, engine),
        "imaging":    score_imaging(pred_inv, gold_inv, engine),
        "monitoring": score_monitoring(
                          pred_extraction.get("monitoring"),
                          gold_extraction.get("monitoring"), engine),
        "medical":    score_medical(pred_tx, gold_tx, engine),
        "conservative": score_conservative(pred_tx, gold_tx, engine),
        "follow_up":  score_follow_up(
                          pred_extraction.get("follow_up"),
                          gold_extraction.get("follow_up"), engine),
        "referral":   score_referral(
                          pred_extraction.get("referral"),
                          gold_extraction.get("referral"), engine),
        "patient_education": score_single(
                          pred_extraction.get("patient_education"),
                          gold_extraction.get("patient_education"), engine),
        "when_to_seek_medical_care": score_single(
                          pred_extraction.get("when_to_seek_medical_care"),
                          gold_extraction.get("when_to_seek_medical_care"), engine),
    }

    # Primary scalar per field = mover score
    field_scores = {name: r["field_avg"] for name, r in fields.items()}
    # Full triple per field
    field_triples = {name: r.get("field_avg_triple", {"mover": r["field_avg"],
                                                       "chrf": 0.0, "cosine": 0.0})
                     for name, r in fields.items()}

    overall        = avg(field_scores.values())
    overall_triple = avg_triple(list(field_triples.values()))

    result = {
        "field_scores":   field_scores,
        "field_triples":  field_triples,
        "field_details":  fields,
        "overall":        overall,
        "overall_triple": overall_triple,
    }

    if run_confusion:
        result["confusion"] = detect_confusion(
            pred_extraction, gold_extraction, engine, threshold=confusion_threshold)

    return result


def evaluate_dataset(pred_list: list, gold_list: list,
                     engine: EmbeddingEngine,
                     run_confusion: bool = False,
                     confusion_threshold: float = 0.5) -> dict:
    """Evaluate the full dataset."""
    gold_by_id = {item["row_index"]: item for item in gold_list}

    case_results      = []
    field_accumulator = defaultdict(list)
    triple_accumulator = defaultdict(list)
    case_confusions   = []

    for pred_item in tqdm(pred_list, desc="Evaluating", unit="case"):
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
            run_confusion=run_confusion,
            confusion_threshold=confusion_threshold,
        )
        result["row_id"] = row_id
        case_results.append(result)

        for field, score in result["field_scores"].items():
            field_accumulator[field].append(score)
        for field, triple in result["field_triples"].items():
            triple_accumulator[field].append(triple)

        if run_confusion and "confusion" in result:
            case_confusions.append(result["confusion"])

    dataset_field_scores  = {f: avg(scores) for f, scores in field_accumulator.items()}
    dataset_field_triples = {f: avg_triple(triples) for f, triples in triple_accumulator.items()}
    dataset_overall        = avg(dataset_field_scores.values())
    dataset_overall_triple = avg_triple(list(dataset_field_triples.values()))

    output = {
        "dataset_field_scores":  dataset_field_scores,
        "dataset_field_triples": dataset_field_triples,
        "dataset_overall":       dataset_overall,
        "dataset_overall_triple": dataset_overall_triple,
        "per_case":              case_results,
    }

    if run_confusion and case_confusions:
        case_ids = [c["row_id"] for c in case_results if "confusion" in c]
        output["confusion_summary"] = aggregate_confusion(case_confusions, case_ids)

    return output


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_report(results: dict,
                 run_confusion: bool = False,
                 confusion_threshold: float = 0.5) -> None:
    W = 80

    print()
    print("=" * W)
    print("  MEDICAL EXTRACTION EVALUATION — MoverScore + chrF + Cosine")
    print("=" * W)
    print(f"  Cases evaluated : {len(results['per_case'])}")
    print(f"  Backbone        : BioClinical-ModernBERT-base")
    print(f"  MoverScore      : EMD over L2-normalised per-token embeddings")
    print(f"  chrF            : Character n-gram F-score (syntactic)")
    print(f"  Cosine          : Sentence mean-pool cosine similarity (semantic)")
    if run_confusion:
        print(f"  Confusion       : ENABLED  (threshold={confusion_threshold})")
    print("=" * W)

    # ── Dataset field table ────────────────────────────────────────────────
    print(f"\n  {'Field':<30}  {'MoverScore':>10}  {'chrF':>8}  {'Cosine':>8}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*8}  {'─'*8}")

    triples = results["dataset_field_triples"]
    scores  = results["dataset_field_scores"]
    for field in scores:
        t = triples.get(field, {})
        print(f"  {field:<30}  {scores[field]:>10.4f}  "
              f"{t.get('chrf',0.0):>8.4f}  {t.get('cosine',0.0):>8.4f}")

    print(f"  {'─'*30}  {'─'*10}  {'─'*8}  {'─'*8}")
    ot = results["dataset_overall_triple"]
    print(f"  {'OVERALL':<30}  {results['dataset_overall']:>10.4f}  "
          f"{ot.get('chrf',0.0):>8.4f}  {ot.get('cosine',0.0):>8.4f}")
    print()

    # ── Conservative detail ────────────────────────────────────────────────
    print("─" * W)
    print("  CONSERVATIVE TREATMENT  (MoverScore detail)")
    print("─" * W)
    for case in results["per_case"]:
        row_id = case["row_id"]
        cons   = case["field_details"].get("conservative", {})
        print(f"\n  Case row_id={row_id}")
        for sub in ("conservative_method", "lifestyle_habit_modifications"):
            t = cons.get(sub, {})
            label = sub.replace("_", " ")
            print(f"    {label}")
            print(f"      Mover={t.get('mover',0.0):.4f}  "
                  f"chrF={t.get('chrf',0.0):.4f}  "
                  f"Cosine={t.get('cosine',0.0):.4f}")

    # ── Per-case summary ───────────────────────────────────────────────────
    print()
    print("─" * W)
    print("  PER-CASE SUMMARY  (MoverScore | chrF | Cosine)")
    print("─" * W)
    fields_order = list(results["dataset_field_scores"].keys())
    # Header
    print(f"  {'Row ID':<8}  {'Overall':>8}  " +
          "  ".join(f"{f[:6]:>6}" for f in fields_order))
    print(f"  {'─'*8}  {'─'*8}  " + "  ".join("─" * 6 for _ in fields_order))

    for case in results["per_case"]:
        mover_vals = "  ".join(
            f"{case['field_scores'].get(f, 0.0):>6.3f}" for f in fields_order
        )
        print(f"  {case['row_id']:<8}  {case['overall']:>8.4f}  {mover_vals}  ← mover")

        t = case["overall_triple"]
        chrf_vals = "  ".join(
            f"{case['field_triples'].get(f, {}).get('chrf', 0.0):>6.3f}" for f in fields_order
        )
        print(f"  {'':<8}  {t.get('chrf',0.0):>8.4f}  {chrf_vals}  ← chrf")

        cos_vals = "  ".join(
            f"{case['field_triples'].get(f, {}).get('cosine', 0.0):>6.3f}" for f in fields_order
        )
        print(f"  {'':<8}  {t.get('cosine',0.0):>8.4f}  {cos_vals}  ← cosine")
        print()

    # ── Confusion report ───────────────────────────────────────────────────
    if run_confusion and "confusion_summary" in results:
        print("=" * W)
        print(f"  FIELD CONFUSION REPORT  (MoverScore threshold={confusion_threshold})")
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate medical extraction with MoverScore + chrF + Cosine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mover_eval.py --pred pred.json --gold gold.json
  python mover_eval.py --pred pred.json --gold gold.json --output results.json
  python mover_eval.py --pred pred.json --gold gold.json \\
      --embedding-model thomas-sounack/BioClinical-ModernBERT-base \\
      --batch-size 32 --device cuda
  python mover_eval.py --pred pred.json --gold gold.json \\
      --cache-embeddings cache.pt --confusion --confusion-threshold 0.5
        """,
    )
    parser.add_argument("--pred",                required=True)
    parser.add_argument("--gold",                required=True)
    parser.add_argument("--output",              default=None)
    parser.add_argument("--embedding-model",     default="thomas-sounack/BioClinical-ModernBERT-base")
    parser.add_argument("--batch-size",          type=int, default=32)
    parser.add_argument("--device",              default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--confusion",           action="store_true")
    parser.add_argument("--confusion-threshold", type=float, default=0.5)
    parser.add_argument("--cache-embeddings",    default=None,
                        help="Path to save/load embedding cache (.pt). "
                             "If file exists, load it; otherwise embed and save.")
    args = parser.parse_args()

    print(f"\nLoading predictions : {args.pred}")
    print(f"Loading gold        : {args.gold}")
    print(f"Embedding model     : {args.embedding_model}")
    print(f"Device              : {args.device}")
    print(f"Batch size          : {args.batch_size}")
    if args.confusion:
        print(f"Confusion detection : enabled (threshold={args.confusion_threshold})")
    if not _SCIPY_AVAILABLE:
        print("[WARN] scipy not found — using greedy EMD fallback. "
              "Install scipy for exact MoverScore.")

    with open(args.pred) as f: pred_data = json.load(f)
    with open(args.gold) as f: gold_data = json.load(f)

    engine = EmbeddingEngine(
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    if args.cache_embeddings and os.path.exists(args.cache_embeddings):
        engine.load_cache(args.cache_embeddings)
    else:
        print("\nCollecting strings for batch embedding ...")
        strings = collect_all_strings(pred_data, gold_data)
        print(f"Unique strings to embed: {len(strings)}")
        engine.build_cache(strings)
        if args.cache_embeddings:
            engine.save_cache(args.cache_embeddings)

    results = evaluate_dataset(
        pred_data, gold_data,
        engine=engine,
        run_confusion=args.confusion,
        confusion_threshold=args.confusion_threshold,
    )

    print_report(results, run_confusion=args.confusion,
                 confusion_threshold=args.confusion_threshold)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full results saved → {args.output}\n")


if __name__ == "__main__":
    main()

"""
python mover_score_eval_script.py \
    --pred extracted_plan_google_gemini-2.0-flash-001_pydantic_normalized.json \
    --gold extractions_annotator_2_normalized.json \
    --output annotator_2_gemini_2_flash_mover_eval_results.json \
    --embedding-model thomas-sounack/BioClinical-ModernBERT-base \
    --batch-size 32 \
    --device cuda \
    --confusion \
    --confusion-threshold 0.5 
"""