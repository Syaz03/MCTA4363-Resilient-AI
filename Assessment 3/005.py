import subprocess
import time
import json
import random
import tempfile
import os
from difflib import SequenceMatcher
from datasets import load_dataset
from codecarbon import EmissionsTracker

LLAMA_CLI = "/home/muhdirfanrosdin/resAI/llama.cpp/build/bin/llama-mtmd-cli"
MODEL = "/home/muhdirfanrosdin/resAI/MCTA4363-Resilient-AI/models/gemma4-E4B-Q8_0.gguf"
MMPROJ = "/home/muhdirfanrosdin/resAI/MCTA4363-Resilient-AI/models/gemma4-E4B-mmproj.gguf"

NUMBER_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
              "five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}

def is_correct(gt, pred):
    gt = gt.lower().strip()
    pred = pred.lower().strip()
    if "<|channel>" in pred or pred.startswith("*"):
        return False
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

def run_inference(image, question, idx=0):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f.name)
        img_path = f.name

    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "--mmproj", MMPROJ,
        "--image", img_path,
        "-p", f"Question: {question} Answer in one word or short phrase only.",
        "-n", "300",
        "--gpu-layers", "99",
        "--jinja",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ttft = time.perf_counter() - start

    if idx == 0:
        print(f"RAW STDOUT: {repr(result.stdout[:300])}")

    os.unlink(img_path)

    output = result.stdout.strip()

    for marker in ["<|turn>model", "<turn|>", "<|turn>"]:
        output = output.replace(marker, "").strip()

    if "<|channel>thought" in output:
        parts = output.split("<|channel>thought")
        before = parts[0].strip()
        if before:
            output = before
        else:
            after = parts[-1] if len(parts) > 1 else ""
            lines = [l.strip() for l in after.split('\n')
                     if l.strip()
                     and not l.strip().startswith("*")
                     and not l.strip().startswith("Here's")
                     and not l.strip().startswith("Thinking")
                     and not l.strip()[0].isdigit()]
            output = lines[-1] if lines else ""

    if "<|channel>" in output:
        output = output.split("<|channel>")[0].strip()

    if output.startswith("*"):
        output = ""

    lines = [l.strip() for l in output.split('\n') if l.strip()]
    output = lines[-1] if lines else ""

    return output, ttft

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

print("\n--- EXPERIMENT 005: Q8_0 GGUF (llama.cpp) ---")
tracker = EmissionsTracker(log_level="error")
tracker.start()
results = []

for i, item in enumerate(stratified):
    try:
        pred, ttft = run_inference(item["image"], item["question"], idx=i)
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
        print(f"[{i+1}/50] Q: {item['question'][:40]} | GT: {gt} | Pred: {pred[:40]} | TTFT: {ttft:.2f}s | {icon}")
    except Exception as e:
        print(f"[{i+1}/50] ERROR: {e}")
        results.append({"question": item["question"], "error": str(e), "correct": False})

emissions = tracker.stop()
accuracy = sum(r.get("correct", False) for r in results) / len(results)
avg_ttft = sum(r.get("ttft_seconds", 0) for r in results if "ttft_seconds" in r) / len(results)

report = {
    "model": "gemma-4-E4B-it Q8_0 GGUF llama.cpp",
    "vqa_accuracy": round(accuracy, 4),
    "avg_ttft_seconds": round(avg_ttft, 4),
    "co2_kg": round(emissions, 8),
    "co2_grams_per_sample": round((emissions * 1000) / len(results), 6),
    "results": results
}

with open("005_results.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*45}")
print(f"VQA Accuracy : {accuracy:.2%}")
print(f"Avg TTFT     : {avg_ttft:.3f}s")
print(f"CO2/sample   : {(emissions*1000)/len(results):.4f}g")
print(f"{'='*45}")
print("Saved to 005_results.json")