import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
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

# Load one sample
ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
item = next(iter(ds))
image = item["image"]
question = item["question"]
gt = item["multiple_choice_answer"]

print(f"Q: {question}")
print(f"GT: {gt}")
print(f"Image size: {image.size}, mode: {image.mode}")

# Save image to verify it looks correct
image.save("test_image.jpg")
print("Image saved to test_image.jpg")

# Check what processor does to image
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": f"Question: {question} Answer briefly."}
]}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to("cuda:0")

print(f"pixel_values shape: {inputs['pixel_values'].shape}")
print(f"pixel_values min: {inputs['pixel_values'].min():.3f}")
print(f"pixel_values max: {inputs['pixel_values'].max():.3f}")
print(f"pixel_values mean: {inputs['pixel_values'].mean():.3f}")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

input_len = inputs["input_ids"].shape[-1]
result = processor.decode(output[0][input_len:], skip_special_tokens=True)
print(f"Prediction: {result}")