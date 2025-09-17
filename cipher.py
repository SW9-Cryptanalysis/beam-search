import random

class Cipher:
    def __init__(self):
        self.key = None

    def generate_key(self, alphabet, min=1, max=1, weights=None):
        if min > max:
            raise ValueError("min cannot be greater than max")
        values = list(range(min, max + 1))
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