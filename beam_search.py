import heapq
import os
import time
import json
import pickle
import math
from collections import Counter
# NEW IMPORTS
import multiprocessing
from multiprocessing import Pool

# =======================================================================
# ==  TOP-LEVEL WORKER FUNCTIONS FOR MULTIPROCESSING                   ==
# =======================================================================

# We use a dictionary to hold variables for each worker process
# This avoids passing the huge 'model' object around
WORKER_VARS = {}

def init_worker(model, cipher_tokens, ve, nmax, n_model):
    """
    Initializer for each pool worker.
    Stores read-only data in a global-like dictionary.
    """
    WORKER_VARS['model'] = model
    WORKER_VARS['cipher_tokens'] = cipher_tokens
    WORKER_VARS['Ve'] = ve
    WORKER_VARS['nmax'] = nmax
    WORKER_VARS['n_model'] = n_model
    # print(f"Worker {os.getpid()} initialized.") # For debugging

def _score_partial_static(mapping):
    """
    A static, standalone version of score_partial for parallel workers.
    It uses the data stored in WORKER_VARS.
    """
    # Access data from the worker's global-like store
    model = WORKER_VARS['model']
    cipher_tokens = WORKER_VARS['cipher_tokens']
    n = WORKER_VARS['n_model']

    # Standalone version of decrypt_with_mapping
    s = ''.join(mapping.get(tok, '?') for tok in cipher_tokens)
    s = s.replace('?', '$')  # placeholder for unfixed parts
    pad = '$' * (n - 1)
    s = pad + s + '$'

    total_logp = 0.0
    for i in range(n - 1, len(s)):
        hist = s[i - (n - 1):i]
        nxt = s[i]
        if '$' not in hist + nxt:
            if hist in model and nxt in model[hist]:
                total_logp += model[hist][nxt]
            else:
                total_logp += math.log(1e-12) # OOV penalty
        else:
            total_logp += 0.0  # optimistic scoring
    return total_logp

def process_one_hypothesis(task):
    """
    This is the main function executed by a worker process.
    It takes one (mapping, base_score) tuple and the next cipher token.
    It returns a list of all scored extensions for that single hypothesis.
    """
    mapping, base_score = task['mapping_score']
    f = task['f'] # The cipher token we are currently assigning

    # Access worker data
    Ve = WORKER_VARS['Ve']
    nmax = WORKER_VARS['nmax']

    results = []
    
    # Standalone version of _can_extend_with and extend_mapping
    for e in Ve:
        # Check _can_extend_with
        used_by_plain = sum(1 for f_tok, e_plain in mapping.items() if e_plain == e)
        
        if used_by_plain < nmax:
            # Create new mapping
            new_mapping = dict(mapping)
            new_mapping[f] = e
            
            # Score it
            s = _score_partial_static(new_mapping)
            results.append((new_mapping, s))
            
    return results

# =======================================================================
# ==  ORIGINAL CLASS (with modified beam_search)                       ==
# =======================================================================

class BeamSearchCipherSolver:
    def __init__(self, ciphertext, model, beam_size=500000,
                 plaintext_alphabet=None, nmax=1,
                 checkpoint_dir="checkpoints", checkpoint_interval=3600):
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

    def beam_search(self, resume=True, num_workers=None):
        start_time = time.time()
        state = self._load_latest_checkpoint() if resume else None

        if state:
            beam = state["beam"]
            start_step = state["step"] + 1
        else:
            beam = [({}, 0.0)]
            start_step = 1

        if num_workers is None:
            num_workers = os.cpu_count() or 1
            print(f"Using default {num_workers} worker processes (all available cores).")
        else:
            print(f"Using {num_workers} worker processes.")

        print(f"Starting beam search with {len(self.Vf)} cipher tokens, nmax={self.nmax}, beam={self.beam_size}")

        # We pass the large, read-only data *once* to the initializer
        init_args = (self.model, self.cipher_tokens, self.Ve, self.nmax, self.model["_n"])
        
        # Use 'spawn' context for better cross-platform compatibility
        context = multiprocessing.get_context('spawn')
        
        # Use 'with' to automatically manage the pool's lifecycle
        with context.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:

            for step in range(start_step, len(self.ext_order) + 1):
                step_start_time = time.time()
                f = self.ext_order[step - 1] # Current cipher token to map
                
                # 1. Create the list of tasks to distribute
                # A task is one item from the old beam
                tasks = [{'mapping_score': mapping_score, 'f': f} for mapping_score in beam]

                # 2. Run the parallel processing
                # pool.map distributes the 'tasks' list to the 'process_one_hypothesis' function
                # This blocks until all tasks are done.
                # list_of_lists will contain one list of results for each task.
                list_of_lists = pool.map(process_one_hypothesis, tasks)

                # 3. Flatten the list of lists into our single new_beam
                new_beam = [item for sublist in list_of_lists for item in sublist]

                if not new_beam:
                    print(f"⚠️ No extensions possible at step {step}.")
                    break

                # 4. Sort and prune
                new_beam.sort(key=lambda x: x[1], reverse=True)
                beam = new_beam[:self.beam_size]
                
                step_duration = time.time() - step_start_time
                best_score = beam[0][1]
                avg_score = best_score / max(1, len(self.cipher_tokens))
                
                print(f"Step {step}/{len(self.ext_order)} | Candidates={len(new_beam)} | "
                      f"Beam={len(beam)} | Best={best_score:.2f} (avg {avg_score:.3f}) | "
                      f"Time={step_duration:.2f}s")

                # Save checkpoint every hour
                if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
                    self._save_checkpoint(step, beam)
                    self.last_checkpoint_time = time.time()

        best_mapping, best_score = beam[0]
        print("✅ Beam search complete.")
        return best_mapping, best_score

    def decrypt_best(self, best_mapping):
        return self.decrypt_with_mapping(best_mapping)