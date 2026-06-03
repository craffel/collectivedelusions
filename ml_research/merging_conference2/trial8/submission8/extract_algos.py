import re

with open("papers/submission9.txt", "r") as f:
    text = f.read()

# Let's search for "Algorithm 1" and print the next 100 lines
print("=== Algorithm 1 and surrounding context in submission9 ===")
matches = [m.start() for m in re.finditer(r"Algorithm 1", text, re.IGNORECASE)]
for m in matches:
    print(text[m:m+1500])
    print("-"*50)

print("\n=== Algorithm 2 and surrounding context in submission9 ===")
matches = [m.start() for m in re.finditer(r"Algorithm 2", text, re.IGNORECASE)]
for m in matches:
    print(text[m:m+1500])
    print("-"*50)

print("\n=== Algorithm 3 and surrounding context in submission9 ===")
matches = [m.start() for m in re.finditer(r"Algorithm 3", text, re.IGNORECASE)]
for m in matches:
    print(text[m:m+1500])
    print("-"*50)
