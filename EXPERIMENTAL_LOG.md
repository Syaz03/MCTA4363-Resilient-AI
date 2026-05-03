# 📓 Experimental Log — Resilient AI Challenge (Track 3)
**Team:** Syaz03 | **Model:** Gemma 4 E4B (Image-to-Text)  
**Hardware:** NVIDIA RTX 4060 (local) / T4 16GB (Colab)  
**Log Started:** April 27, 2026

---

## 🗂️ Log Format
Each experiment entry documents: hypothesis → implementation → results → decision.

---

## Experiment 001 — Baseline: Raw Pipeline Load
**Date:** 2026-04-27  
**Status:** ❌ Failed  

### Hypothesis
Load `google/gemma-4-31B-it` directly and run image-to-text inference on T4.

### Implementation
```python
from transformers import pipeline
pipe = pipeline("image-text-to-text", model="google/gemma-4-31B-it")
```

### Result
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 222.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 187.81 MiB is free.
```

### Analysis
- 31B model requires ~62GB VRAM in bf16 — 4× beyond T4 capacity
- Wrong model: competition uses `gemma-4-E4B-it` (4B effective params via MoE), not 31B
- `pipeline()` loads to single device without memory offloading

### Decision
→ Switch to correct model `google/gemma-4-E4B-it`  
→ Use `device_map='auto'` for memory offloading  
→ Apply 4-bit quantization as primary compression strategy

---

## Experiment 002 — Correct Model: NF4 4-bit Quantization
**Date:** 2026-04-28  
**Status:** ✅ Success  

### Hypothesis
Loading `gemma-4-E4B-it` with NF4 4-bit quantization will fit on T4 (16GB) and produce coherent image-to-text outputs.

### Implementation
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForImageTextToText.from_pretrained(
    'google/gemma-4-E4B-it',
    quantization_config=bnb_config,
    device_map='auto',
)
```

### Result
| Metric | Value |
|--------|-------|
| VRAM used | ~4.8 GB |
| VRAM saved vs bf16 | ~5.2 GB (~52%) |
| Load time | ~3.5 min (download + quant) |
| Inference on test image | Coherent, accurate response |
| CO₂ emissions (3 images) | Measured (see baseline_results.json) |

### Sample Output
- Q: "What animal is on the candy?"  
- A: "The candy features a bear design..." ✅

### Analysis
- NF4 fits comfortably on T4
- Double quantization adds further ~0.4 bits/param savings
- Responses are coherent — no hallucination observed on sanity checks
- This is the **Round 1 submission baseline**

### Decision
→ Submit NF4 model as Round 1 trial  
→ Proceed to Phase 2 experiments: AWQ + MoE routing reduction

---

## Experiment 003 — MoE Expert Routing Reduction
**Date:** 2026-04-29  
**Status:** 🔄 In Progress  

### Hypothesis
Gemma 4 E4B is a Mixture-of-Experts model. Reducing `num_experts_per_tok` from default (likely 2) to 1 will halve active expert computation, reducing energy without proportional accuracy loss.

### Implementation
```python
config = AutoConfig.from_pretrained('google/gemma-4-E4B-it')
original_topk = config.num_experts_per_tok  # inspect first
config.num_experts_per_tok = 1              # reduce top-k routing

model = AutoModelForImageTextToText.from_pretrained(
    'google/gemma-4-E4B-it',
    config=config,
    quantization_config=bnb_config,
    device_map='auto',
)
```

### Result
| Metric | NF4 Baseline | NF4 + top-k=1 | Delta |
|--------|-------------|----------------|-------|
| VRAM | ~4.8 GB | ~4.6 GB | -4% |
| CO₂ (3 images) | TBD | TBD | TBD |
| Accuracy (local) | TBD | TBD | TBD |

> ⏳ Results pending — run in progress on RTX 4060

### Analysis
- MoE routing is architecture-level change → must justify in README per competition rules
- Risk: expert specialization may degrade if wrong expert is always dropped
- Mitigation: evaluate on diverse image types (natural scenes, documents, diagrams)

### Decision
→ TBD pending accuracy evaluation

---

## Experiment 004 — Token Selection (Attention-Based)
**Date:** 2026-04-30  
**Status:** 🔄 In Progress  

### Hypothesis
The 256 vision tokens are not equally important. Retaining only the top-50% most attended tokens (128 tokens) before the VL projector will reduce compute without significant accuracy loss.

### Implementation
Calibrated on 10 diverse images from VQAv2. Importance score = average column-wise attention received across all heads and layers.

```python
token_selector = AttentionTokenSelector(teacher, processor, keep_ratio=0.5)
importance = token_selector.compute_importance_scores(calib_images)
token_mask = token_selector.get_token_mask(importance, num_vision_tokens=256)
# 128 tokens retained
```

### Result
| Vision tokens | CO₂ (estimate) | Accuracy (local) |
|--------------|----------------|-----------------|
| 256 (baseline) | TBD | TBD |
| 128 (50% selected) | TBD | TBD |
| 64 (25% selected) | TBD | TBD |

> ⏳ Pending

### Analysis
- Attention rollout may not capture cross-modal importance accurately
- Alternative: use gradient-based saliency instead of attention scores
- ToMe (Token Merging) was rejected: merges tokens which risks VL projector misalignment

### Decision
→ TBD

---

## Experiment 005 — Sequential Duo-Teacher KD (Planned)
**Date:** Planned for 2026-05-05  
**Status:** 📋 Planned  

### Hypothesis
Decoupled vision-then-language distillation prevents gradient interference between modalities, producing better accuracy retention than joint distillation at equivalent compression levels.

### Planned Implementation
- Phase A: Load E4B teacher + student (E2B extracted). Freeze student language head. Distill vision encoder using KL divergence on vision-layer outputs. Run 2 epochs on VQAv2-subset (5k samples).
- Phase B: Freeze student vision encoder (weights from Phase A). Distill language head using E4B teacher soft labels. Run 3 epochs on mixed VQA + captioning data.

### Evaluation Plan
Compare against joint distillation baseline on:
- VQAv2 accuracy
- CO₂ emissions
- TTFT (time to first token)

### Compute Budget
- Phase A: ~6 hours on RTX 4060
- Phase B: ~8 hours on RTX 4060

---

## 📊 Summary Comparison Table

| Experiment | Technique | VRAM (GB) | CO₂ Relative | Accuracy (local) | Status |
|-----------|-----------|-----------|--------------|-----------------|--------|
| 001 | Raw 31B load | OOM | — | — | ❌ Failed |
| 002 | NF4 4-bit (E4B) | 4.8 | Baseline | TBD | ✅ Round 1 sub |
| 003 | NF4 + MoE top-k=1 | 4.6 | TBD | TBD | 🔄 In Progress |
| 004 | NF4 + Token selection 50% | TBD | TBD | TBD | 🔄 In Progress |
| 005 | Duo-teacher KD | TBD | TBD | TBD | 📋 Planned |

---

## 🔧 Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.x |
| Transformers | latest |
| bitsandbytes | latest |
| CodeCarbon | latest |
| CUDA | 12.1 |
| GPU (local) | NVIDIA RTX 4060 8GB |
| GPU (Colab) | T4 16GB |

---

## 📝 Notes & Observations

- `gemma-4-E4B-it` uses a Matryoshka-style nested architecture — E2B is baked in, not trained separately
- The VL projector layer naming in Gemma 4 differs from standard PaliGemma patterns — inspect with `named_modules()` before applying targeted quantization
- vLLM 0.17.1 and llama.cpp have different energy profiles on L4 — evaluate both before final submission engine choice
- Competition energy measurement uses CodeCarbon (CPU + GPU via NVML) — replicate this exactly during local testing for accurate comparison
