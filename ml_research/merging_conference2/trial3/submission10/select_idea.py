import random

def select_idea():
    # Use seed 42 for a reproducible pseudo-random selection
    random.seed(42)
    selected_index = random.randint(1, 10)
    print(f"Selected Idea Index: {selected_index}")

if __name__ == "__main__":
    select_idea()
