import torch
import time
import json
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from codecarbon import EmissionsTracker
from datasets import load_dataset
import random

model_id = "google/gemma-4-E4B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading model — INT8 quantization (less aggressive than NF4)...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0"
)
model.eval()
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Load dataset
print("Loading VQAv2...")
ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
samples = []
for item in ds:
    samples.append(item)
    if len(samples) >= 500:
        break

yes_no = [s for s in samples if s["answer_type"] == "yes/no"][:15]
number = [s for s in samples if s["answer_type"] == "number"][:15]
other  = [s for s in samples if s["answer_type"] == "other"][:20]
stratified = yes_no + number + other
random.shuffle(stratified)
print(f"Dataset ready: {len(stratified)} samples")

def run_inference(item):
    image = item["image"]
    question = item["question"]

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Question: {question} Answer briefly."}
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda:0")

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    ttft = time.perf_counter() - start

    input_len = inputs["input_ids"].shape[-1]
    predicted = processor.decode(
        output[0][input_len:], skip_special_tokens=True
    ).strip()

    return predicted, ttft

print("\n--- EXPERIMENT 003: Selective NF4 (Vision Protected) ---")
tracker = EmissionsTracker(log_level="error")
tracker.start()
results = []

for i, item in enumerate(stratified):
    pred, ttft = run_inference(item)
    gt = item["multiple_choice_answer"]
    correct = gt.lower() in pred.lower()
    results.append({
        "question": item["question"],
        "predicted": pred,
        "ground_truth": gt,
        "answer_type": item["answer_type"],
        "ttft_seconds": round(ttft, 4),
        "correct": correct
    })
    print(f"[{i+1}/50] Q: {item['question'][:40]} | GT: {gt} | Pred: {pred[:40]} | TTFT: {ttft:.2f}s")

emissions = tracker.stop()
accuracy = sum(r["correct"] for r in results) / len(results)
avg_ttft = sum(r["ttft_seconds"] for r in results) / len(results)

report = {
    "model": "gemma-4-E4B-it selective NF4 vision protected",
    "vqa_accuracy": round(accuracy, 4),
    "avg_ttft_seconds": round(avg_ttft, 4),
    "co2_kg": round(emissions, 8),
    "co2_grams_per_sample": round((emissions * 1000) / len(results), 6),
    "results": results
}

with open("003_results.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*45}")
print(f"VQA Accuracy : {accuracy:.2%}")
print(f"Avg TTFT     : {avg_ttft:.3f}s")
print(f"CO2/sample   : {(emissions*1000)/len(results):.4f}g")
print(f"VRAM         : {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"{'='*45}")
print("Saved to 003_results.json")