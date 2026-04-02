# Track 3 — Innovation Challenge

## Assessment 1: Challenge Analysis & Baseline Solution

### 1. Description of the Innovation Challenge
This project directly embodies the competition’s thesis of sustainable and inclusive AI. The challenge is to prove that publication-grade, cutting-edge VLM (Vision-Language Model) compression research can be achieved without relying on datacenter-scale compute. By aggressively optimizing complex multimodal architectures, we aim to democratize advanced AI access, making it viable for consumer-grade local hardware setups.

### 2. Problem Statement & Evaluation Criteria
**Problem Statement:**
Deploying high-capability multimodal models like Gemma 3n (E4B) on local hardware is restricted by severe memory bottlenecks, particularly the 256+ vision token bottleneck, and the high energy costs of inference. Compressing these models often leads to cross-modal misalignment and hallucination due to architectural mismatch and gradient interference during joint training. The objective is to compress the E4B architecture to a nested E2B sub-model while maintaining representational stability.

**Evaluation Criteria:**
* **Energy Metrics (Primary):** Joules-per-token measured via `nvidia-smi` power logging.
* **Latency:** Time To First Token (TTFT).
* **Efficiency:** FLOPs evaluated as a theoretical anchor.
* **Accuracy:** VQA score compared against the uncompressed E4B baseline.

### 3. Analysis of Dataset / Available Resources
**Dataset:**
* **VQAv2 Dataset:** Utilizing stratified, paired image-text data. This will be specifically used to calibrate the quantization process and preserve cross-modal alignment at the vision-language projector boundary.

**Available Resources:**
* **Hardware:** Executed entirely on a local consumer-grade machine equipped with an NVIDIA RTX 4060 running Ubuntu.

### 4. Proposed Deep Learning Approach
We propose a Resilient AI Methodology using a highly optimized, multi-phase compression pipeline:

* **Phase 1: Architecture Extraction & Token Selection:** We will extract the nested E2B sub-model from the E4B weights using MatFormer’s pre-defined Matryoshka boundaries. To reduce the vision token bottleneck without ToMe’s architectural mismatch risk, we will apply attention-based static token selection derived from the E4B teacher.
* **Phase 2: Hybrid Quantization:** To maximize memory efficiency while protecting performance, we will apply AWQ on the MobileNet-v5 vision encoder (protecting high-activation salient weights), GPTQ on the language head for global compression, and INT8 integer-only quantization on the vision-language projector to eliminate FP16 memory shuffles.
* **Phase 3: Sequential Duo-Teacher Knowledge Distillation:** To prevent gradient interference and hallucinations, we will decouple the distillation process. We first distill the vision student from the uncompressed MobileNet-v5. Once representations stabilize, we freeze it, and subsequently distill the language student from the E4B teacher.

---
*(Note: Sections 5 & 6 covering the codebase and supplementary documentation will be appended here upon completion.)*
