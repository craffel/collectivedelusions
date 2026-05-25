import random

# Use a deterministic seed to make it reproducible but pseudo-random
random.seed(42)
selected_index = random.randint(1, 10)
print(f"Selected index: {selected_index}")
