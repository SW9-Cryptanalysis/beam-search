import os
import math
import json
import pickle
from collections import Counter, defaultdict
from typing import Dict, Tuple, Iterable
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
BOOKS_DIR = "./books"          # folder containing .txt files
N = 5                          # n-gram
BOUNDARY = "$"                 # boundary symbol
MODEL_PKL = "char6gram_model.pkl"
META_JSON = "char6gram_meta.json"


# ==============================
# TEXT PREPROCESSING
# ==============================
def iter_book_files(dirpath: str) -> Iterable[str]:
    """Yield all .txt files under dirpath."""
    for fname in sorted(os.listdir(dirpath)):
        if fname.lower().endswith(".txt"):
            yield os.path.join(dirpath, fname)


def normalize_text(s: str) -> str:
    """Keep only lowercase letters and underscores (spaces)."""
    s = s.lower()
    # Convert spaces to underscore for boundary awareness
    s = s.replace(" ", "_")
    return "".join(ch for ch in s if ('a' <= ch <= 'z') or ch == "_")


# ==============================
# BUILD COUNTS
# ==============================
def build_ngram_counts(dirpath: str, n: int = N) -> Tuple[Dict[int, Counter], int]:
    """
    Count all n-grams up to order n from the text files.
    Returns:
        counts: {order -> Counter of ngrams}
        total_chars: total letters processed
    """
    counts = {k: Counter() for k in range(1, n + 1)}
    total_chars = 0

    book_files = list(iter_book_files(dirpath))
    print(f"Found {len(book_files)} book files in {dirpath}")

    for path in tqdm(book_files, desc="Building n-gram counts", unit="file"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        text = normalize_text(text)
        if not text:
            continue

        # Pad with boundary markers
        pad = BOUNDARY * (n - 1)
        seq = pad + text + BOUNDARY
        total_chars += len(text)
        L = len(seq)

        for i in range(L):
            for k in range(1, n + 1):
                if i + k <= L:
                    g = seq[i:i + k]
                    counts[k][g] += 1

    return counts, total_chars


# ==============================
# MODIFIED KNESER–NEY
# ==============================
def build_conditional_probs_KN_fast(counts, n=N):
    """
    Much faster iterative approximation of Modified Kneser–Ney.
    Computes discounts & backoff weights bottom-up.
    """
    from collections import defaultdict
    import math, numpy as np

    cond_probs = {}
    vocab = set(c for gram in counts[1] for c in gram)

    # ---- Precompute discount constants ----
    def calc_discount(N1, N2, N3p):
        Y = N1 / (N1 + 2 * N2) if (N1 + 2 * N2) > 0 else 0
        D1 = 1 - 2 * Y * N2 / N1 if N1 else 0.75
        D2 = 2 - 3 * Y * N3p / N2 if N2 else 1.0
        D3 = 3 - 4 * Y * N3p / N3p if N3p else 1.0
        return D1, D2, D3

    N1 = sum(1 for c in counts[n].values() if c == 1)
    N2 = sum(1 for c in counts[n].values() if c == 2)
    N3p = sum(1 for c in counts[n].values() if c >= 3)
    D1, D2, D3p = calc_discount(N1, N2, N3p)

    # ---- 1-gram continuation probs ----
    cont_counts = defaultdict(set)
    for gram in counts[2]:
        hist, nxt = gram[:-1], gram[-1]
        cont_counts[nxt].add(hist)
    cont = {c: len(v) for c, v in cont_counts.items()}
    denom = sum(cont.values())
    P_cont = {c: cont[c] / denom for c in vocab}

    # ---- Iterate from 2-grams → n-grams ----
    backoff = P_cont
    for order in range(2, n + 1):
        ngrams = counts[order]
        hist_counts = defaultdict(int)
        follow_types = defaultdict(set)
        for g, c in ngrams.items():
            hist_counts[g[:-1]] += c
            follow_types[g[:-1]].add(g[-1])

        probs = {}
        for g, c in ngrams.items():
            hist, nxt = g[:-1], g[-1]
            denom = hist_counts[hist]
            uniq = len(follow_types[hist])
            D = D3p if c >= 3 else D2 if c == 2 else D1
            base = max(c - D, 0) / denom
            lambda_h = (D * uniq) / denom
            p_bo = backoff.get(nxt, 1e-12)
            p = base + lambda_h * p_bo
            if hist not in probs:
                probs[hist] = {}
            probs[hist][nxt] = math.log(max(p, 1e-12))
        backoff = {nxt: math.exp(np.mean([v[nxt] for v in probs.values() if nxt in v])) for nxt in vocab}
        cond_probs.update(probs)

    cond_probs["_vocab"] = "".join(sorted(vocab))
    cond_probs["_n"] = n
    print("✅ Fast Kneser–Ney built.")
    return cond_probs

# ==============================
# SEQUENCE SCORING
# ==============================
def score_sequence(seq: str, cond_probs: Dict[str, Dict[str, float]], n: int = N) -> float:
    """Compute log-probability of a sequence under the model."""
    s = normalize_text(seq)
    pad = BOUNDARY * (n - 1)
    s = pad + s + BOUNDARY
    total_logp = 0.0
    for i in range(n - 1, len(s)):
        hist = s[i - (n - 1):i]
        nxt = s[i]
        if hist in cond_probs and nxt in cond_probs[hist]:
            total_logp += cond_probs[hist][nxt]
        else:
            total_logp += math.log(1e-12)
    return total_logp


# ==============================
# SAVE / LOAD
# ==============================
def save_model(model: Dict[str, Dict[str, float]], model_path=MODEL_PKL, meta_path=META_JSON):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    meta = {"model_path": model_path, "n": model.get("_n", N), "vocab": model.get("_vocab", "")}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Saved model to {model_path}")


def load_model(model_path=MODEL_PKL):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ==============================
# MAIN PIPELINE
# ==============================
def build_and_save(dirpath=BOOKS_DIR, n=N):
    print("📚 Counting n-grams...")
    counts, total_chars = build_ngram_counts(dirpath, n=n)
    print(f"Total characters counted: {total_chars}")

    print("🧮 Building Modified Kneser–Ney model...")
    cond_probs = build_conditional_probs_KN_fast(counts, n=n)
    save_model(cond_probs)

    vocab = cond_probs["_vocab"]
    print(f"Vocab size: {len(vocab)}")
    if n in counts:
        print("Top 6-grams:")
        for g, c in counts[n].most_common(10):
            print(c, g)
    return cond_probs


if __name__ == "__main__":
    if os.path.exists(MODEL_PKL):
        print(f"Model {MODEL_PKL} already exists. Skipping retrain.")
    else:
        build_and_save()