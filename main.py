from n_gram_model import load_model
from beam_search import BeamSearchCipherSolver
import json

print("Loading n-gram model...")
model = load_model("char6gram_model.pkl")

# --- Load ciphertext ---
with open("cipher-30.json", "r", encoding="utf-8") as f:
    data = json.load(f)
cipher_text = data["ciphertext"]

# --- Load ciphertext ---
with open("cipher-30.json", "r", encoding="utf-8") as f:
    data = json.load(f)
plain_text = data["plaintext"]

# --- Configure and run beam search ---
BEAM_WIDTH = 500000  # wider beam for better search coverage
HOMOPHONIC_NMAX = 21

solver = BeamSearchCipherSolver(
    ciphertext=cipher_text,
    model=model,
    beam_size=BEAM_WIDTH,
    nmax=HOMOPHONIC_NMAX
)

print(f"\nStarting beam search (beam={BEAM_WIDTH}, nmax={HOMOPHONIC_NMAX})...")
best_mapping, best_score = solver.beam_search(resume=True)

# --- Print best mapping ---
print("\nBest mapping found:")
for c, p in sorted(best_mapping.items(), key=lambda x: int(x[0])):
    print(f"{c} → {p}")

# --- Show deciphered text ---
plaintext_guess = solver.decrypt_best(best_mapping)
print("\nDeciphered text (partial):")
print(plaintext_guess[:500])

total = len(plain_text)
correct = sum(1 for p, r in zip(plaintext_guess, plain_text) if p == r)
ser = 1 - (correct / total)

print(f"\nSER:{ser}")
