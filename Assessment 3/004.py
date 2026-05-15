import torch
import time
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from codecarbon import EmissionsTracker
from datasets import load_dataset
import random

model_id = "google/gemma-4-E4B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading model in FP8...")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.float8_e4m3fn,
    device_map="cuda:0"
)
model.eval()
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")