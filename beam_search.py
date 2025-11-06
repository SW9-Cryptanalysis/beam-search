import heapq
import os
import time
import json
import pickle
import math
from collections import Counter
from n_gram_model import score_sequence


class BeamSearchCipherSolver:
    def __init__(self, ciphertext, model, beam_size=10000,
                 plaintext_alphabet=None, nmax=1,
                 checkpoint_dir="checkpoints", checkpoint_interval=300):
        """
        ciphertext: a string of numbers separated by spaces
        model: loaded n-gram model (from n_gram_model.load_model)
        beam_size: number of hypotheses to keep per step (upper limit)
        plaintext_alphabet: allowed plaintext symbols
        nmax: max cipher tokens allowed per plaintext letter
        checkpoint_dir: folder where checkpoints are stored
        checkpoint_interval: seconds between checkpoints (default: 1 hour)
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
        self.Vf = sorted(set(self.cipher_tokens))
        self.Ve = plaintext_alphabet or list(model["_vocab"].replace("$", ""))
        self.unigram_counts = Counter(self.cipher_tokens)
        self.ext_order = [f for f, _ in self.unigram_counts.most_common()]

        # --- Checkpointing setup ---
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.last_checkpoint_time = time.time()

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
        s = s.replace('?', '$')  # placeholder for unfixed parts

        n = self.model["_n"]
        pad = '$' * (n - 1)
        s = pad + s + '$'

        total_logp = 0.0
        for i in range(n - 1, len(s)):
            hist = s[i - (n - 1):i]
            nxt = s[i]
            if '$' not in hist + nxt:
                if hist in self.model and nxt in self.model[hist]:
                    total_logp += self.model[hist][nxt]
                else:
                    total_logp += math.log(1e-12)
            else:
                total_logp += 0.0  # optimistic scoring
        return total_logp

    # ---------------- Extension limits ----------------

    def _can_extend_with(self, mapping, next_cipher, candidate_plain):
        used_by_plain = sum(1 for f, e in mapping.items() if e == candidate_plain)
        return used_by_plain < self.nmax

    def extend_mapping(self, mapping, next_cipher):
        for e in self.Ve:
            if self._can_extend_with(mapping, next_cipher, e):
                new_mapping = dict(mapping)
                new_mapping[next_cipher] = e
                yield new_mapping

    # ---------------- Checkpointing ----------------

    def _checkpoint_path(self, step):
        return os.path.join(self.checkpoint_dir, f"checkpoint_step{step}.pkl")

    def _save_checkpoint(self, step, beam):
        path = self._checkpoint_path(step)
        state = {
            "step": step,
            "beam": beam,
            "timestamp": time.time(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"💾 Checkpoint saved at step {step} → {path}")

        # keep only latest 3 checkpoints
        ckpts = sorted(
            [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir)
             if f.startswith("checkpoint_step")],
            key=os.path.getmtime, reverse=True
        )
        for old in ckpts[3:]:
            os.remove(old)

    def _load_latest_checkpoint(self):
        ckpts = sorted(
            [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir)
             if f.startswith("checkpoint_step")],
            key=os.path.getmtime, reverse=True
        )
        if not ckpts:
            return None
        path = ckpts[0]
        with open(path, "rb") as f:
            state = pickle.load(f)
        print(f"🔄 Resuming from checkpoint: {path}")
        return state

    # ---------------- Beam search ----------------

    def beam_search(self, resume=True):
        start_time = time.time()
        state = self._load_latest_checkpoint() if resume else None

        if state:
            beam = state["beam"]
            start_step = state["step"] + 1
        else:
            beam = [({}, 0.0)]
            start_step = 1

        print(f"Starting beam search with {len(self.Vf)} cipher tokens, nmax={self.nmax}, beam={self.beam_size}")

        for step in range(start_step, len(self.ext_order) + 1):
            f = self.ext_order[step - 1]
            new_beam = []

            for mapping, base_score in beam:
                for new_mapping in self.extend_mapping(mapping, f):
                    s = self.score_partial(new_mapping)
                    new_beam.append((new_mapping, s))

            if not new_beam:
                print(f"⚠️ No extensions possible at step {step}.")
                break

            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_size]

            best_score = beam[0][1]
            avg_score = best_score / max(1, len(self.cipher_tokens))
            print(f"Step {step}/{len(self.ext_order)} | Candidates={len(new_beam)} | "
                  f"Beam={len(beam)} | Best={best_score:.2f} (avg {avg_score:.3f})")

            # Save checkpoint every hour
            if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
                self._save_checkpoint(step, beam)
                self.last_checkpoint_time = time.time()

        best_mapping, best_score = beam[0]
        print("✅ Beam search complete.")
        return best_mapping, best_score

    def decrypt_best(self, best_mapping):
        return self.decrypt_with_mapping(best_mapping)