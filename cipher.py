import random

class Cipher:
    def __init__(self):
        self.key = None

    def generate_key(self, alphabet, range_vals=(1,2), weights=None):
        # Default range is only [1]
        if range_vals == (1,2):
            values = [1]
        else:
            # If a tuple or list is provided, convert to list of values
            if isinstance(range_vals, tuple) or isinstance(range_vals, list):
                values = list(range(*range_vals)) if len(range_vals) == 2 else list(range_vals)
            else:
                values = list(range_vals)
        if weights is None:
            weights = [1] * len(values)
        key = {}
        count = 0
        for letter in alphabet:
            key[letter] = []
            num_mappings = random.choices(values, weights=weights[:len(values)], k=1)[0]
            for _ in range(num_mappings):
                key[letter].append(count)
                count += 1
        self.key = key

    def encrypt(self, plaintext):
        if self.key is None:
            raise Exception("The cipher has no key")
        
        plaintext_formatted = [c.upper() for c in plaintext if c.isalpha()]
        ciphertext = [random.choice(self.key[letter]) for letter in plaintext_formatted]
        return ciphertext
    
    def __str__(self):
        return f"Key: {self.key}"