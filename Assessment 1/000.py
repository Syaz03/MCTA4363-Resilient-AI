import torch
import time
import json
import random
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from codecarbon import EmissionsTracker
from datasets import load_dataset
from difflib import SequenceMatcher

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

NUMBER_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
              "five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}

def is_correct(gt, pred):
    gt = gt.lower().strip()
    pred = pred.lower().strip()
    if not pred:
        return False
    gt_n = NUMBER_MAP.get(gt, gt)
    pred_n = NUMBER_MAP.get(pred, pred)
    if gt_n == pred_n:
        return True
    if gt in pred or pred in gt:
        return True
    ratio = SequenceMatcher(None, gt, pred).ratio()
    return ratio > 0.6

model_id = "google/gemma-4-E4B-it"
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading unquantized model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="/tmp/offload"
)
model.eval()
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

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
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"Question: {question} Answer in one word or short phrase only."}
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda:0")
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    ttft = time.perf_counter() - start
    input_len = inputs["input_ids"].shape[-1]
    pred = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
    return pred, ttft

print("\n--- BASELINE: Gemma 4 E4B Unquantized ---")
tracker = EmissionsTracker(log_level="error")
tracker.start()
results = []

for i, item in enumerate(stratified):
    try:
        pred, ttft = run_inference(item)
        gt = item["multiple_choice_answer"]
        correct = is_correct(gt, pred)
        results.append({
            "question": item["question"],
            "predicted": pred,
            "ground_truth": gt,
            "answer_type": item["answer_type"],
            "ttft_seconds": round(ttft, 4),
            "correct": correct
        })
        icon = "✅" if correct else "❌"
        print(f"[{i+1}/50] Q: {item['question'][:35]} | GT: {gt} | Pred: {pred[:30]} | {icon}")
    except Exception as e:
        print(f"[{i+1}/50] ERROR: {e}")
        results.append({"question": item["question"], "error": str(e), "correct": False})

emissions = tracker.stop()
accuracy = sum(r.get("correct", False) for r in results) / len(results)
avg_ttft = sum(r.get("ttft_seconds", 0) for r in results if "ttft_seconds" in r) / len(results)

report = {
    "model": "gemma-4-E4B-it unquantized bfloat16",
    "vqa_accuracy": round(accuracy, 4),
    "avg_ttft_seconds": round(avg_ttft, 4),
    "co2_kg": round(emissions, 8),
    "co2_grams_per_sample": round((emissions*1000)/len(results), 6),
    "results": results
}

with open("gemma4_baseline_results.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*45}")
print(f"VQA Accuracy : {accuracy:.2%}")
print(f"Avg TTFT     : {avg_ttft:.3f}s")
print(f"CO2/sample   : {(emissions*1000)/len(results):.4f}g")
print(f"{'='*45}")
print("Saved to gemma4_baseline_results.json")