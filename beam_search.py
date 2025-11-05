import heapq
from collections import Counter
import math
from n_gram_model import score_sequence


class BeamSearchCipherSolver:
    def __init__(self, ciphertext, model, beam_size=10000,
                 plaintext_alphabet=None, nmax=1):
        """
        ciphertext: a string of numbers separated by spaces
        model: loaded n-gram model (from n_gram_model.load_model)
        beam_size: number of hypotheses to keep per step (upper limit)
        plaintext_alphabet: allowed plaintext symbols
        nmax: max cipher tokens allowed per plaintext letter
        """
        # --- Parse ciphertext ---
        if isinstance(ciphertext, str):
            self.cipher_tokens = [tok for tok in ciphertext.strip().split() if tok.isdigit()]
        else:
            self.cipher_tokens = list(map(str, ciphertext))
        if not self.cipher_tokens:
            raise ValueError("Ciphertext appears empty or not tokenized correctly!")

        self.model = model
        self.beam_size = beam_size
        self.nmax = nmax
        self.Vf = sorted(set(self.cipher_tokens))   # cipher vocabulary (numbers as strings)
        self.Ve = plaintext_alphabet or list(model["_vocab"].replace("$", ""))
        self.unigram_counts = Counter(self.cipher_tokens)
        self.ext_order = [f for f, _ in self.unigram_counts.most_common()]

    # ---------------- Utility functions ----------------

    def decrypt_with_mapping(self, mapping):
        """Apply current mapping (unknowns → '?')."""
        return ''.join(mapping.get(tok, '?') for tok in self.cipher_tokens)

    def score_partial(self, mapping):
      """
      Optimistic heuristic (Eq. 7 from Nuhn et al. 2013):
      - Score only n-grams that are fully fixed (no '?').
      - Unfixed parts contribute 0 to the total score.
      """
      s = self.decrypt_with_mapping(mapping)
      s = s.replace('?', '$')  # keep consistent padding token

      n = self.model["_n"]  # assuming stored in model
      pad = '$' * (n - 1)
      s = pad + s + '$'

      total_logp = 0.0
      for i in range(n - 1, len(s)):
          hist = s[i - (n - 1):i]
          nxt = s[i]
          # only count fully fixed n-grams (no '$' inside hist+nxt except boundary padding)
          if '$' not in hist + nxt:
              if hist in self.model and nxt in self.model[hist]:
                  total_logp += self.model[hist][nxt]
              else:
                  total_logp += math.log(1e-12)
          else:
              # optimistic: ignore unfinished parts
              total_logp += 0.0

      return total_logp
    
    # ---------------- Extension limits ----------------

    def _can_extend_with(self, mapping, next_cipher, candidate_plain):
        """Homophonic constraint: ≤ nmax cipher tokens per plaintext letter."""
        used_by_plain = sum(1 for f, e in mapping.items() if e == candidate_plain)
        return used_by_plain < self.nmax

    def extend_mapping(self, mapping, next_cipher):
        """Generate valid extensions for next cipher token."""
        for e in self.Ve:
            if self._can_extend_with(mapping, next_cipher, e):
                new_mapping = dict(mapping)
                new_mapping[next_cipher] = e
                yield new_mapping

    # ---------------- Beam search ----------------

    def beam_search(self):
        beam = [({}, 0.0)]  # list of (mapping, score)
        print(f"Starting beam search with {len(self.Vf)} cipher tokens, nmax={self.nmax}, beam={self.beam_size}")

        for step, f in enumerate(self.ext_order, 1):
            new_beam = []

            # Expand all current hypotheses
            for mapping, base_score in beam:
                for new_mapping in self.extend_mapping(mapping, f):
                    s = self.score_partial(new_mapping)
                    new_beam.append((new_mapping, s))

            if not new_beam:
                print(f"⚠️ No extensions possible at step {step}.")
                break

            # Sort all candidates by score descending
            new_beam.sort(key=lambda x: x[1], reverse=True)

            # Limit to beam_size *only if* we have more than that
            if len(new_beam) > self.beam_size:
                beam = new_beam[:self.beam_size]
            else:
                beam = new_beam  # keep all if fewer than beam_size

            best_score = beam[0][1]
            avg_score = best_score / max(1, len(self.cipher_tokens))
            print(f"Step {step}/{len(self.ext_order)} | Candidates={len(new_beam)} | "
                  f"Beam={len(beam)} | Best={best_score:.2f} (avg {avg_score:.3f})")

        best_mapping, best_score = beam[0]
        return best_mapping, best_score

    def decrypt_best(self, best_mapping):
        return self.decrypt_with_mapping(best_mapping)