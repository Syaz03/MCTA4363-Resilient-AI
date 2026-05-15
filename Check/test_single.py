import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset

model_id = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0"
)
model.eval()

# Load one VQAv2 sample
ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
item = next(iter(ds))
image = item["image"]
question = item["question"]
gt = item["multiple_choice_answer"]

print(f"Question: {question}")
print(f"Ground truth: {gt}")
print(f"Image size: {image.size}")

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

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)

input_len = inputs["input_ids"].shape[-1]
result = processor.decode(output[0][input_len:], skip_special_tokens=True)
print(f"Prediction: {result}")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What is shown in this image? Answer briefly."}
    ]}
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda:0")

print("Input keys:", list(inputs.keys()))
print("pixel_values present:", "pixel_values" in inputs)
print("input_ids shape:", inputs["input_ids"].shape)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

input_len = inputs["input_ids"].shape[-1]
result = processor.decode(output[0][input_len:], skip_special_tokens=True)
print("Output:", result)