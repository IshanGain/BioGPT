# BioGPT: LLM-Based BioScript Compiler Automation

**Author:** Ishan Gain  
**Institution:** [Your Institution]  
**Date:** August 2026

---

## Overview

BioGPT is the first LLM-based pipeline for automatically generating
**BioScript** (`.bs`) code from natural language descriptions of biological
laboratory protocols.

BioScript is a domain-specific language (DSL) used by the
[lilott8/BioScript](https://github.com/lilott8/BioScript) ANTLR4-based
compiler for automating digital microfluidic (DMF) protocols.

---

## Demo

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/ishangain481/biogpt-bioscript)**

---

## Results

| Model | Struct% | BLEU-1 | Instructions | Dispense |
|-------|---------|--------|--------------|----------|
| Qwen2.5-Coder-3B (3 epochs) | 67.4% | 0.223 | 6/18 | 10/18 |
| Qwen2.5-Coder-3B (5 epochs) | 67.4% | 0.261 | 7/18 | 9/18 |
| Llama-3.2-3B (5 epochs) | 73.6% | 0.344 | 9/18 | 12/18 |
| **Qwen2.5-Coder-7B (5 epochs)** | **88.2%** | **0.354** | **12/18** | **14/18** |

**Best model:** Qwen2.5-Coder-7B-Instruct fine-tuned with QLoRA for 5 epochs.

---

## Dataset: OpenBioSet

- **175** biological protocol folders
- **174** valid (description, BioScript) pairs
- **553** training examples after augmentation (4x)
- **17** validation examples
- **18** test examples

Each folder contains:
```
protocol_name/
├── protocol_name.bs          # BioScript source code (target)
├── protocol_name.json        # Protocol metadata
├── description.txt           # Natural language description (input)
├── output.ir                 # Compiled intermediate representation
└── output.dot                # Control flow graph
```

---

## Repository Structure

```
BioGPT/
├── app.py                    # HuggingFace Spaces Gradio demo
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
├── dataset/
│   ├── raw_extraction.json   # All 174 protocol records
│   ├── train_augmented.jsonl # 553 training examples
│   ├── val.jsonl             # 17 validation examples
│   └── test.jsonl            # 18 test examples
│
├── models/
│   ├── qwen3b_3ep/           # Qwen2.5-Coder-3B (3 epochs)
│   ├── qwen3b_5ep/           # Qwen2.5-Coder-3B (5 epochs)
│   ├── llama3b_5ep/          # Llama-3.2-3B (5 epochs)
│   └── qwen7b_5ep/           # Qwen2.5-Coder-7B (5 epochs) ← Best
│
├── evaluation/
│   ├── comparison_summary.json
│   └── *_results.json        # Per-model evaluation results
│
├── notebooks/
│   └── biogpt.ipynb          # Full training pipeline (Kaggle)
│
└── src/
    ├── data_extraction.py    # Step 1: Extract protocols
    ├── augmentation.py       # Step 2: Groq augmentation
    ├── training.py           # Step 3: QLoRA fine-tuning
    ├── evaluation.py         # Step 4: Evaluation metrics
    └── inference.py          # Generation function
```

---

## Methods

### 1. Data Extraction
Extracted 174 valid (description, BioScript) pairs from OpenBioSet
using custom parsers handling Kaggle's double-extension file naming.

### 2. Data Augmentation
Used Groq API (Llama-3.3-70B) to generate 3 paraphrased variants
per training description, growing the dataset from 139 to 553 examples.

### 3. Fine-tuning
Applied QLoRA (4-bit quantization + LoRA adapters) using Unsloth
on Kaggle T4×2 GPU. Trained 4 models for comparison:

| Model | Parameters | Epochs | Training Time |
|-------|-----------|--------|---------------|
| Qwen2.5-Coder-3B | 3B | 3 | ~84 min |
| Qwen2.5-Coder-3B | 3B | 5 | ~120 min |
| Llama-3.2-3B | 3B | 5 | ~120 min |
| Qwen2.5-Coder-7B | 7B | 5 | ~286 min |

**LoRA Configuration:**
- Rank (r): 16, Alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Dropout: 0.05, Optimizer: adamw_8bit

### 4. Evaluation
Evaluated on 18 held-out test protocols using:
- **Structural Accuracy**: 8-element BioScript structure check
- **BLEU-1**: Word-level overlap with reference code

---

## Key Findings

1. **Model size is the strongest predictor** — Qwen-7B outperforms all 3B models
2. **Code specialization helps** — Qwen models beat Llama despite similar size
3. **Overfitting after step 50** — Dataset size is the main bottleneck
4. **Zero repetitive outputs** — Repetition penalty (1.3) eliminates looping

---

## Installation

```bash
git clone https://github.com/ishangain481/BioGPT
cd BioGPT
pip install -r requirements.txt
```

## Quick Inference

```python
from src.inference import BioGPTInference

biogpt = BioGPTInference("./models/qwen7b_5ep/lora_adapter")
code   = biogpt.generate(
    "PCR amplification protocol: Mix DNA template with primers, "
    "heat to 95C for denaturation, cycle and detect by fluorescence."
)
print(code)
```

---

## Limitations

- Small dataset (174 protocols) limits generalization
- Compiler-based validation not yet integrated
- Module/variable names differ from reference (expected behavior)
- Instructions section missing in some outputs (39-67% coverage)

---

## Future Work

- Expand OpenBioSet with more protocol types
- Integrate BioScript compiler for execution-based evaluation
- Explore larger models (13B+)
- Implement RLHF with compiler feedback as reward signal
- Add few-shot prompting for improved consistency

---

## Citation

```bibtex
@misc{gain2026biogpt,
  title  = {BioGPT: LLM-Based BioScript Compiler Automation},
  author = {Ishan Gain},
  year   = {2026},
  url    = {https://github.com/ishangain481/BioGPT}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.