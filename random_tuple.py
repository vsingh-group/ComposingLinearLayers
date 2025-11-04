import random
import hashlib

def generate_unique_tuples(num_tuples, tuple_size, X, Y, seed_string=None):
    if seed_string is not None:
        # Hash the seed string and convert to int
        hash_obj = hashlib.sha256(seed_string.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)  # Reduce to 32-bit int for compatibility
        random.seed(seed)

    if Y - X + 1 < tuple_size:
        raise ValueError("Range too small to generate tuples with unique values.")

    result = []
    for _ in range(num_tuples):
        values = random.sample(range(X, Y + 1), tuple_size)
        result.append(sorted(tuple(values)))

    return result

# Example usage
seed = "hellaswag_chat"
tuples = generate_unique_tuples(num_tuples=5, tuple_size=2, X=3, Y=31, seed_string=seed)
print(seed, tuples)
