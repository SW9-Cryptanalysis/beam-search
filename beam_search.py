import heapq
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from n_gram_model import score_sequence


# ---------- Shared helpers (used by workers and class) ----------

def decrypt_with_mapping(cipher_tokens, mapping):
    """Apply mapping to ciphertext tokens, replacing unknowns with '?'. """
    return "".join(mapping.get(tok, "?") for tok in cipher_tokens)


def can_extend_with(mapping, next_cipher, candidate_plain, nmax):
    """Check homophonic constraint."""
    used_by_plain = sum(1 for f, e in mapping.items() if e == candidate_plain)
    return used_by_plain < nmax


def score_candidate(mapping, next_cipher, cipher_tokens, model, Ve, nmax):
    """Compute scores for all possible extensions of the mapping."""
    n = model["_n"]
    pad = "$" * (n - 1)

    results = []
    for e in Ve:
        if can_extend_with(mapping, next_cipher, e, nmax):
            new_mapping = dict(mapping)
            new_mapping[next_cipher] = e

            s = 0.0
            s_str = decrypt_with_mapping(cipher_tokens, new_mapping)
            s_str = pad + s_str.replace("?", "$") + "$"

            # Optimistic heuristic: count only fixed n-grams
            for i in range(n - 1, len(s_str)):
                hist = s_str[i - (n - 1):i]
                nxt = s_str[i]
                if "$" not in hist + nxt:
                    if hist in model and nxt in model[hist]:
                        s += model[hist][nxt]
                    else:
                        s += math.log(1e-12)
            results.append((new_mapping, s))
    return results


# ---------- Main class ----------

class BeamSearchCipherSolver:
    def __init__(self, ciphertext, model, beam_size=10000,
                 plaintext_alphabet=None, nmax=1, num_workers=None):
        """
        ciphertext: string of numbers separated by spaces
        model: loaded n-gram model
        beam_size: beam width
        plaintext_alphabet: allowed plaintext symbols
        nmax: max cipher tokens per plaintext letter
        num_workers: number of CPU processes (default = os.cpu_count())
        """
        # --- Parse ciphertext ---
        if isinstance(ciphertext, str):
            self.cipher_tokens = [tok for tok in ciphertext.strip().split() if tok.isdigit()]
        else:
            self.cipher_tokens = list(map(str, ciphertext))
        if not self.cipher_tokens:
            raise ValueError("Ciphertext appears empty or invalid!")

        self.model = model
        self.beam_size = beam_size
        self.nmax = nmax
        self.num_workers = num_workers
        self.Vf = sorted(set(self.cipher_tokens))
        self.Ve = plaintext_alphabet or list(model["_vocab"].replace("$", ""))
        self.unigram_counts = Counter(self.cipher_tokens)
        self.ext_order = [f for f, _ in self.unigram_counts.most_common()]

    # ---------------- Beam search ----------------

    def beam_search(self):
        beam = [({}, 0.0)]
        print(f"Starting beam search with {len(self.Vf)} cipher tokens, "
              f"nmax={self.nmax}, beam={self.beam_size}, workers={self.num_workers or 'auto'}")

        for step, f in enumerate(self.ext_order, 1):
            new_beam = []

            # ---- Parallel scoring ----
            args_list = [(mapping, f, self.cipher_tokens, self.model, self.Ve, self.nmax)
                         for mapping, _ in beam]

            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(score_candidate, *args) for args in args_list]
                for fut in as_completed(futures):
                    for result in fut.result():
                        new_beam.append(result)

            # ---- Select best candidates ----
            if not new_beam:
                print(f"⚠️ No extensions possible at step {step}.")
                break

            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_size]

            best_score = beam[0][1]
            avg_score = best_score / max(1, len(self.cipher_tokens))
            print(f"Step {step}/{len(self.ext_order)} | Candidates={len(new_beam)} | "
                  f"Beam={len(beam)} | Best={best_score:.2f} (avg {avg_score:.3f})")

        best_mapping, best_score = beam[0]
        return best_mapping, best_score

    def decrypt_best(self, best_mapping):
        return decrypt_with_mapping(self.cipher_tokens, best_mapping)