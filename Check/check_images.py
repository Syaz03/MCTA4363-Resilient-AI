from datasets import load_dataset
import random

ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
samples = []
for item in ds:
    samples.append(item)
    if len(samples) >= 5:
        break

for item in samples:
    img = item["image"]
    print(f"Q: {item['question']}")
    print(f"GT: {item['multiple_choice_answer']}")
    print(f"Image mode: {img.mode}, size: {img.size}")
    print()