import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset

model_id = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="/tmp/offload"
)
model.eval()

ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
item = next(iter(ds))
image = item["image"]
question = item["question"]
gt = item["multiple_choice_answer"]

print(f"Q: {question}")
print(f"GT: {gt}")

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": f"Question: {question} Answer briefly."}
]}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to("cuda:0")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)

input_len = inputs["input_ids"].shape[-1]
result = processor.decode(output[0][input_len:], skip_special_tokens=True)
print(f"Prediction: {result}")