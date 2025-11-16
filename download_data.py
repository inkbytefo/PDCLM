## Developer: inkbytefo
## Modified: 2025-11-16

from datasets import load_dataset
import os

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)

# Download and prepare wikitext dataset
print("Downloading wikitext dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:1%]")  # %1 subset, ~2M karakter

print(f"Dataset size: {len(dataset)} examples")
print("Saving to data/raw/wikitext_sample.txt...")

with open("data/raw/wikitext_sample.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        if example["text"].strip():
            f.write(example["text"] + "\n")

print("Data download and preparation completed!")
