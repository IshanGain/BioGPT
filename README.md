<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,30:0d2137,70:0f3d2e,100:1a6b4a&height=280&section=header&text=BioGPT&desc=LLM-Based%20BioScript%20Compiler%20Automation&fontSize=54&descSize=22&fontColor=ffffff&descAlignY=62&fontAlignY=40&animation=fadeIn" />

<br/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=19&pause=1000&color=1DB954&center=true&vCenter=true&width=960&lines=Fine-tuned+LLMs+for+Automated+BioScript+DSL+Generation;Qwen2.5-Coder-7B+%E2%80%94+88.2%25+Structural+Accuracy+on+Held-Out+Protocols;QLoRA+%2B+Unsloth+%E2%80%94+4-bit+Quantized+Training+on+Kaggle+T4x2;174+Real-World+DMF+Protocols+%E2%80%94+OpenBioSet+Dataset;4+Models+Benchmarked+%E2%80%94+Full+Robustness+and+BLEU+Evaluation" alt="Typing SVG" />
</a>

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/hellokitty1212/biogpt-qwen7b-lora)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4x2-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-1DB954?style=for-the-badge)](LICENSE)

<br/>

**Authors:** Sayan Mondal &nbsp;&middot;&nbsp; Sanju De &nbsp;&middot;&nbsp; Ishan Gain &nbsp;&nbsp;|&nbsp;&nbsp; **Institution:** Indian Institute of Technology, Roorkee &nbsp;&nbsp;|&nbsp;&nbsp; **Date:** August 2026

[![Pipeline](https://img.shields.io/badge/Pipeline%20Architecture%20%26%20Implementation-Ishan%20Gain-1DB954?style=for-the-badge)](https://github.com/IshanGain)

<br/>

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0a0a0a,100:0d2137&height=80&section=header&text=Table%20of%20Contents&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

<div align="center">

<br/>

<table>
  <tr>
    <td align="center" width="200">
      <a href="#overview">
        <img src="https://img.shields.io/badge/01-Overview-1DB954?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#pipeline">
        <img src="https://img.shields.io/badge/02-Pipeline-0d2137?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#architecture">
        <img src="https://img.shields.io/badge/03-Architecture-1a1a2e?style=for-the-badge&logoColor=white" />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="200">
      <a href="#results">
        <img src="https://img.shields.io/badge/04-Results-0f3d2e?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#dataset-openbioscript">
        <img src="https://img.shields.io/badge/05-Dataset-1a3a1a?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#data-augmentation">
        <img src="https://img.shields.io/badge/06-Data%20Augmentation-2a1a0e?style=for-the-badge&logoColor=white" />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="200">
      <a href="#fine-tuning-configuration">
        <img src="https://img.shields.io/badge/07-Fine--tuning%20Config-1a1a2e?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#repository-structure">
        <img src="https://img.shields.io/badge/08-Repo%20Structure-0d2137?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#key-findings">
        <img src="https://img.shields.io/badge/09-Key%20Findings-0f3d2e?style=for-the-badge&logoColor=white" />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="200">
      <a href="#limitations">
        <img src="https://img.shields.io/badge/10-Limitations-2a0a0a?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#future-work">
        <img src="https://img.shields.io/badge/11-Future%20Work-1a3a1a?style=for-the-badge&logoColor=white" />
      </a>
    </td>
    <td align="center" width="200">
      <a href="#citation">
        <img src="https://img.shields.io/badge/12-Citation-1a1a2e?style=for-the-badge&logoColor=white" />
      </a>
    </td>
  </tr>
</table>

<br/>

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0a0a0a,100:0d2137&height=80&section=header&text=Overview&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Overview

<div align="center">

[![DSL](https://img.shields.io/badge/Domain-Digital%20Microfluidics%20(DMF)-1DB954?style=flat-square)]()
[![Target](https://img.shields.io/badge/Output-BioScript%20DSL%20(.bs)-0d2137?style=flat-square)]()
[![Compiler](https://img.shields.io/badge/Compiler-ANTLR4%20BioScript-1a1a2e?style=flat-square)]()
[![Barrier](https://img.shields.io/badge/Problem-Manual%20DSL%20Expertise%20Barrier-2a0a0a?style=flat-square)]()
[![Solution](https://img.shields.io/badge/Solution-LLM%20Fine--tuning%20on%20174%20Protocols-0f3d2e?style=flat-square)]()

</div>

<br/>

**BioGPT** is the first LLM-based pipeline for automatically generating **BioScript** (`.bs`) code from natural language descriptions of biological laboratory protocols.

BioScript is a domain-specific language (DSL) compiled by the [lilott8/BioScript](https://github.com/lilott8/BioScript) ANTLR4-based compiler for automating digital microfluidic (DMF) protocols. Writing BioScript manually requires deep expertise in both the biological domain and compiler syntax — a significant barrier for wet-lab scientists. BioGPT eliminates this barrier by fine-tuning large language models on a curated dataset of 174 real-world protocols.

> [!IMPORTANT]
> *"Can large language models learn to generate syntactically valid, structurally complete BioScript from plain-English protocol descriptions?"*

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0d2137,100:0a0a0a&height=80&section=header&text=Pipeline&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Pipeline

<div align="center">

[![Step 1](https://img.shields.io/badge/Step%201-Data%20Extraction-1DB954?style=flat-square)]()
[![Step 2](https://img.shields.io/badge/Step%202-Augmentation%20via%20Groq-0d2137?style=flat-square)]()
[![Step 3](https://img.shields.io/badge/Step%203-QLoRA%20Fine--tuning-1a1a2e?style=flat-square)]()
[![Step 4](https://img.shields.io/badge/Step%204-Structural%20%2B%20BLEU%20Eval-0f3d2e?style=flat-square)]()

</div>

```mermaid
flowchart LR
    A([OpenBioSet\n174 Protocols]) --> B[Data Extraction\nStep 1]
    B --> C[Train / Val / Test\n139 / 17 / 18]
    C --> D[Data Augmentation\nGroq API\nStep 2]
    D --> E[553 Training\nExamples]
    E --> F[QLoRA Fine-tuning\nUnsloth + Kaggle T4x2\nStep 3]
    F --> G[4 Fine-tuned\nModels]
    G --> H[Evaluation\nStructural + BLEU\nStep 4]
    H --> I([Best Model\nQwen2.5-Coder-7B\n88.2% Structural])

    style A fill:#0f3d2e,color:#fff,stroke:#1DB954,stroke-width:2px
    style I fill:#0f3d2e,color:#fff,stroke:#1DB954,stroke-width:2px
    style F fill:#0d2137,color:#fff,stroke:#20BEFF,stroke-width:2px
    style H fill:#2a0a0a,color:#fff,stroke:#e74c3c,stroke-width:2px
    style B fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style C fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style D fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style E fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style G fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0a0a0a,100:1a1a2e&height=80&section=header&text=Architecture&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Architecture

<div align="center">

[![Backbone](https://img.shields.io/badge/Base%20Model-Qwen2.5--Coder--7B--Instruct-EE4C2C?style=flat-square)]()
[![Adapter](https://img.shields.io/badge/Adapter-QLoRA%20Rank%2016-7B61FF?style=flat-square)]()
[![Quantization](https://img.shields.io/badge/Quantization-4--bit%20NF4-1DB954?style=flat-square)]()
[![Input](https://img.shields.io/badge/Input-Natural%20Language%20Protocol-0d2137?style=flat-square)]()
[![Output](https://img.shields.io/badge/Output-BioScript%20.bs%20Code-0f3d2e?style=flat-square)]()

</div>

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        A["Natural Language\nProtocol Description"]
    end

    subgraph BioGPT Pipeline
        B["System Prompt\nBioScript Rules"]
        C["Chat Template\nFormatting"]
        D["Qwen2.5-Coder-7B\nBase Model"]
        E["QLoRA Adapter\nLoRA Rank 16"]
        F["Generation\nTemp=0.7\nRepPenalty=1.3"]
    end

    subgraph OUTPUT["Output Layer"]
        G["BioScript Code\n.bs file"]
        H["module declarations"]
        I["manifest declarations"]
        J["instructions block"]
        K["operations\ndispense / mix / heat"]
    end

    A --> B --> C --> D
    D --> E --> F --> G
    G --> H & I & J
    J --> K

    style INPUT fill:#0f3d2e,color:#fff,stroke:#1DB954,stroke-width:2px
    style OUTPUT fill:#0d2137,color:#fff,stroke:#20BEFF,stroke-width:2px
    style D fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:2px
    style E fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:2px
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:1a1a2e,100:0f3d2e&height=80&section=header&text=Results&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Results

<div align="center">

[![Best Structural](https://img.shields.io/badge/Best%20Structural%20Accuracy-88.2%25-1DB954?style=for-the-badge)]()
[![Best BLEU](https://img.shields.io/badge/Best%20BLEU--1-0.354-0d2137?style=for-the-badge)]()
[![Best Val Loss](https://img.shields.io/badge/Best%20Val%20Loss-1.200-1a1a2e?style=for-the-badge)]()
[![Best Train Loss](https://img.shields.io/badge/Best%20Train%20Loss-0.018-0f3d2e?style=for-the-badge)]()

</div>

<br/>

### Model Comparison on 18 Held-Out Test Protocols

<div align="center">

| Model | Parameters | Epochs | Struct% | BLEU-1 | Module | Manifest | Instructions | Dispense | Non-Rep |
|:------|:---------:|:------:|:-------:|:------:|:------:|:--------:|:------------:|:--------:|:-------:|
| Qwen2.5-Coder-3B | 3B | 3 | 67.4% | 0.223 | 14/18 | 10/18 | 6/18 | 10/18 | 18/18 |
| Qwen2.5-Coder-3B | 3B | 5 | 67.4% | 0.261 | 15/18 | 13/18 | 7/18 | 9/18 | 18/18 |
| Llama-3.2-3B | 3B | 5 | 73.6% | 0.344 | 14/18 | 9/18 | 9/18 | 12/18 | 18/18 |
| **Qwen2.5-Coder-7B** | **7B** | **5** | **88.2%** | **0.354** | **17/18** | **14/18** | **12/18** | **14/18** | **18/18** |

> **Best model:** Qwen2.5-Coder-7B-Instruct fine-tuned with QLoRA for 5 epochs.

</div>

<br/>

### Training Loss Summary

<div align="center">

| Model | Best Val Loss | Best Step | Final Train Loss |
|:------|:------------:|:---------:|:---------------:|
| Qwen2.5-Coder-3B (3ep) | 1.326 | 50 | 0.481 |
| Qwen2.5-Coder-3B (5ep) | 1.330 | 50 | 0.054 |
| Llama-3.2-3B (5ep) | 1.418 | 50 | 0.049 |
| **Qwen2.5-Coder-7B (5ep)** | **1.200** | **50** | **0.018** |

</div>

<br/>

### Training Curves

<table>
  <tr>
    <td align="center" width="50%">
      <b>Training and Validation Loss</b><br/><br/>
      <img src="results/training_curves.png" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>Model Comparison — Structural Accuracy & BLEU</b><br/><br/>
      <img src="results/model_comparison.png" width="100%"/>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <b>Validation Loss Comparison Across All Models</b><br/><br/>
      <img src="results/validation_loss_comparison.png" width="60%"/>
    </td>
  </tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0f3d2e,100:1a1a2e&height=80&section=header&text=Dataset%3A%20OpenBioSet&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Dataset: OpenBioSet

<div align="center">

[![Source](https://img.shields.io/badge/Source-OpenBioSet-1DB954?style=flat-square)]()
[![Protocols](https://img.shields.io/badge/Total%20Protocols-174-0d2137?style=flat-square)]()
[![Format](https://img.shields.io/badge/Format-COCO%20%2F%20Folder%20per%20Protocol-1a1a2e?style=flat-square)]()
[![Augmented](https://img.shields.io/badge/After%20Augmentation-588%20Total-0f3d2e?style=flat-square)]()

</div>

<br/>

```mermaid
pie title OpenBioSet Dataset Split (174 protocols)
    "Training (139)" : 139
    "Validation (17)" : 17
    "Test (18)" : 18
```

<div align="center">

<table>
  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/TRAIN-139%20protocols%20%C2%B7%2080%25-0f3d2e?style=for-the-badge" /><br/>
      <sub>553 after augmentation</sub>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/VAL-17%20protocols%20%C2%B7%2010%25-0d2137?style=for-the-badge" /><br/>
      <sub>No augmentation applied</sub>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/TEST-18%20protocols%20%C2%B7%2010%25-1a1a2e?style=for-the-badge" /><br/>
      <sub>No augmentation applied</sub>
    </td>
  </tr>
</table>

| Split | Original | After Augmentation |
|:-----:|:-------:|:-----------------:|
| Train | 139 | 553 (4x) |
| Val | 17 | 17 (no augmentation) |
| Test | 18 | 18 (no augmentation) |
| **Total** | **174** | **588** |

</div>

Each protocol folder contains:

```
protocol_name/
├── protocol_name.bs          # BioScript source code  (target output)
├── protocol_name.json        # Protocol metadata
├── description.txt           # Natural language description  (model input)
├── output.ir                 # Compiled intermediate representation
└── output.dot                # Control flow graph
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:1a1a2e,100:0d2137&height=80&section=header&text=Data%20Augmentation&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Data Augmentation

<div align="center">

[![API](https://img.shields.io/badge/API-Groq%20%2B%20Llama--3.3--70B-1DB954?style=flat-square)]()
[![Variants](https://img.shields.io/badge/Variants-3%20per%20Description-0d2137?style=flat-square)]()
[![Multiplier](https://img.shields.io/badge/Coverage-4x%20Multiplier-0f3d2e?style=flat-square)]()
[![Target](https://img.shields.io/badge/BioScript%20Target-Identical%20Across%20Variants-1a1a2e?style=flat-square)]()

</div>

```mermaid
flowchart LR
    A([Original Description\n1 per protocol]) --> B[Groq API\nLlama-3.3-70B]
    B --> C[Variant 1\nConcise Technical]
    B --> D[Variant 2\nStep-by-Step]
    B --> E[Variant 3\nContext Focused]
    A & C & D & E --> F([4x Training Examples\nSame BioScript Target])

    style A fill:#0f3d2e,color:#fff,stroke:#1DB954,stroke-width:2px
    style F fill:#0d2137,color:#fff,stroke:#20BEFF,stroke-width:2px
    style B fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:2px
    style C fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style D fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style E fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
```

<div align="center">

| Stat | Count |
|:-----|:-----:|
| Original training descriptions | 139 |
| Augmented variants (3 per description) | 417 |
| Total training examples | 556 |
| Coverage multiplier | 4x |

</div>

BioScript code remains **identical** for all variants — only the natural language input is diversified.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0d2137,100:1a1a2e&height=80&section=header&text=Fine-tuning%20Configuration&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Fine-tuning Configuration

<div align="center">

[![LoRA Rank](https://img.shields.io/badge/LoRA%20Rank-r%3D16-1DB954?style=flat-square)]()
[![Quantization](https://img.shields.io/badge/Quantization-4--bit%20NF4-0d2137?style=flat-square)]()
[![LR](https://img.shields.io/badge/Learning%20Rate-2e--4-1a1a2e?style=flat-square)]()
[![Optimizer](https://img.shields.io/badge/Optimizer-adamw__8bit-0f3d2e?style=flat-square)]()
[![GPU](https://img.shields.io/badge/GPU-Kaggle%20T4x2-20BEFF?style=flat-square)]()
[![Epochs](https://img.shields.io/badge/Best%20Config-5%20Epochs-EE4C2C?style=flat-square)]()

</div>

```mermaid
flowchart TD
    A([Base Model\nQwen2.5-Coder-7B-Instruct]) --> B[4-bit Quantization\nBitsAndBytes NF4]
    B --> C[LoRA Adapters\nr=16, alpha=32]
    C --> D[Target Modules\nq k v o gate up down proj]
    D --> E[Training\nAdamW 8-bit\nLR=2e-4\nCosine Schedule]
    E --> F[Best Checkpoint\nload_best_model_at_end]
    F --> G([LoRA Adapter\n173 MB saved])

    style A fill:#0f3d2e,color:#fff,stroke:#1DB954,stroke-width:2px
    style G fill:#0d2137,color:#fff,stroke:#20BEFF,stroke-width:2px
    style B fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style C fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style D fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style E fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
    style F fill:#1a1a2e,color:#fff,stroke:#7B61FF,stroke-width:1px
```

<div align="center">

| Hyperparameter | Value |
|:--------------|:-----:|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Learning Rate | 2e-4 |
| Batch Size (effective) | 8 (1 x 8 accumulation) |
| Epochs | 5 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.05 |
| Optimizer | adamw_8bit |
| Max Seq Length | 2048 |
| Quantization | 4-bit NF4 |
| GPU | Kaggle T4 x 2 |

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:1a1a2e,100:0a0a0a&height=80&section=header&text=Repository%20Structure&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Repository Structure

```
BioGPT/
├── dataset/
│   ├── raw_extraction.json       # All 174 protocol records
│   ├── train_augmented.jsonl     # 553 training examples
│   ├── val.jsonl                 # 17 validation examples
│   └── test.jsonl                # 18 test examples
│
├── evaluation/
│   ├── comparison_summary.json   # Final comparison metrics
│   ├── Qwen2.5-Coder-3B-3ep_results.json
│   ├── Qwen2.5-Coder-3B-5ep_results.json
│   ├── Llama-3.2-3B-5ep_results.json
│   └── Qwen2.5-Coder-7B-5ep_results.json
│
├── models/
│   ├── qwen3b_3ep/lora_adapter/  # Qwen2.5-Coder-3B (3 epochs)
│   ├── qwen3b_5ep/lora_adapter/  # Qwen2.5-Coder-3B (5 epochs)
│   ├── llama3b_5ep/lora_adapter/ # Llama-3.2-3B (5 epochs)
│   └── qwen7b_5ep/lora_adapter/  # Qwen2.5-Coder-7B (5 epochs) [Best]
│                                    -> HuggingFace: hellokitty1212/biogpt-qwen7b-lora
│
├── notebooks/
│   └── biogpt.ipynb              # Complete training pipeline (Kaggle)
│
├── results/
│   ├── training_curves.png       # Training and validation loss curves
│   ├── model_comparison.png      # Structural accuracy, BLEU, coverage
│   └── validation_loss_comparison.png
│
├── src/
│   ├── __init__.py
│   ├── data_extraction.py        # Step 1: Extract protocols from OpenBioSet
│   ├── augmentation.py           # Step 2: Groq API paraphrasing
│   ├── training.py               # Step 3: QLoRA fine-tuning with Unsloth
│   ├── evaluation.py             # Step 4: Structural accuracy and BLEU
│   └── inference.py              # BioGPTInference class
│
├── generate_plots.py             # Generate result plots locally
├── .gitignore
├── README.md
└── requirements.txt
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0a0a0a,100:0d2137&height=80&section=header&text=Installation%20and%20Quick%20Inference&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Installation

```bash
git clone https://github.com/IshanGain/BioGPT
cd BioGPT
pip install -r requirements.txt
```

## Quick Inference

```python
from src.inference import BioGPTInference

biogpt = BioGPTInference(
    adapter_path = "./models/qwen7b_5ep/lora_adapter",
    hf_token     = "your_hf_token"
)

code = biogpt.generate(
    "PCR amplification protocol: Mix DNA template with primers "
    "and master mix, heat to 95C for denaturation, cycle through "
    "annealing at 60C and extension at 72C, detect by fluorescence."
)
print(code)
```

**Expected output:**

```
module pcr

manifest DNA_Template
manifest Primers
manifest PCR_MasterMix

instructions:

// Step 1: Prepare PCR reaction
dna  = dispense DNA_Template into $1 for 5s
pri  = dispense Primers into $2 for 5s
mmix = dispense PCR_MasterMix into $3 for 5s
rxn  = mix dna with pri for 10s
rxn  = mix rxn with mmix for 10s

// Step 2: PCR cycling
heat rxn at 95c for 5m
heat rxn at 60c for 30s
heat rxn at 72c for 1m

// Step 3: Detect amplification
result = detect fluorescence on rxn for 10s
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0f3d2e,100:0a0a0a&height=80&section=header&text=Key%20Findings&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Key Findings

<div align="center">

[![Finding 1](https://img.shields.io/badge/Finding%201-Model%20Size%20is%20the%20Strongest%20Predictor-1DB954?style=flat-square)]()
[![Finding 2](https://img.shields.io/badge/Finding%202-Code%20Specialization%20Beats%20General%20Tuning-0d2137?style=flat-square)]()
[![Finding 3](https://img.shields.io/badge/Finding%203-Overfitting%20After%20Step%2050-EE4C2C?style=flat-square)]()
[![Finding 4](https://img.shields.io/badge/Finding%204-Epoch%20Count%20Minimal%20Impact%20on%203B-1a1a2e?style=flat-square)]()
[![Finding 5](https://img.shields.io/badge/Finding%205-Zero%20Repetitive%20Outputs-0f3d2e?style=flat-square)]()

</div>

<br/>

<div align="center">

| # | Finding |
|:-:|:--------|
| 1 | **Model size is the strongest predictor** — Qwen-7B outperforms all 3B models by a significant margin (88.2% vs 67.4% structural accuracy). |
| 2 | **Code specialization outperforms general instruction tuning** — Qwen models outperform Llama-3B despite comparable parameter counts, validating the use of code-specialized base models for DSL generation. |
| 3 | **Overfitting occurs after step 50** — All models achieve best validation loss at step 50 and overfit beyond that, confirming dataset size (553 examples) as the primary bottleneck. |
| 4 | **Epoch count has minimal impact on 3B models** — 3 vs 5 epochs produces identical structural accuracy (67.4%), further confirming dataset size limitations. |
| 5 | **Zero repetitive outputs across all models** — Repetition penalty (1.3) completely eliminates looping behavior (18/18 non-repetitive). |

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0a0a0a,100:2a0a0a&height=80&section=header&text=Limitations&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Limitations

<div align="center">

| Limitation | Detail |
|:-----------|:-------|
| Small dataset | 174 protocols limits generalization to unseen protocol types |
| No compiler validation | Structural accuracy is a proxy metric — compiler-based validation not yet integrated |
| Name divergence | Module and variable names differ from reference code (expected generative behavior) |
| Instructions coverage | Instructions section coverage ranges from 33% to 67% depending on model size |

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0d2137,100:0f3d2e&height=80&section=header&text=Future%20Work&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Future Work

<div align="center">

[![High Priority](https://img.shields.io/badge/High%20Priority-2%20directions-EE4C2C?style=flat-square)]()
[![Medium Priority](https://img.shields.io/badge/Medium%20Priority-2%20directions-f39c12?style=flat-square)]()
[![Exploratory](https://img.shields.io/badge/Exploratory-2%20directions-7B61FF?style=flat-square)]()

<br/><br/>

| Priority | Direction |
|:--------:|:----------|
| ![](https://img.shields.io/badge/-High-EE4C2C?style=flat-square) | **Expand OpenBioSet** — grow coverage across more protocol types and biological domains |
| ![](https://img.shields.io/badge/-High-EE4C2C?style=flat-square) | **Compiler-based evaluation** — integrate BioScript compiler for execution-based validation (compilation rate metric) |
| ![](https://img.shields.io/badge/-Medium-f39c12?style=flat-square) | **Larger models** — explore 13B+ architectures with extended fine-tuning |
| ![](https://img.shields.io/badge/-Medium-f39c12?style=flat-square) | **RLHF with compiler feedback** — use compilation success/failure as a reward signal |
| ![](https://img.shields.io/badge/-Exploratory-7B61FF?style=flat-square) | **Few-shot prompting** — improve naming consistency through in-context examples |
| ![](https://img.shields.io/badge/-Exploratory-7B61FF?style=flat-square) | **Multi-turn refinement** — iterative BioScript correction through dialogue |

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0f3d2e,100:1a1a2e&height=80&section=header&text=Citation&fontSize=24&fontColor=ffffff&fontAlignY=55&animation=fadeIn" width="100%"/>

## Citation

```bibtex
@misc{gain2026biogpt,
  title       = {BioGPT: LLM-Based BioScript Compiler Automation},
  author      = {Ishan Gain},
  year        = {2026},
  institution = {Indian Institute of Technology, Roorkee},
  url         = {https://github.com/IshanGain/BioGPT}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1DB954,30:0f3d2e,70:0d2137,100:0a0a0a&height=140&section=footer&animation=fadeIn" width="100%" alt="footer wave" />

<div align="center">

[![Built by](https://img.shields.io/badge/Built%20by-Ishan%20Gain-1DB954?style=for-the-badge&logo=github)](https://github.com/IshanGain)
[![HuggingFace](https://img.shields.io/badge/Model-hellokitty1212%2Fbiogpt--qwen7b--lora-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/hellokitty1212/biogpt-qwen7b-lora)
[![Institution](https://img.shields.io/badge/IIT%20Roorkee-August%202026-0d2137?style=for-the-badge)](https://www.iitr.ac.in)

</div>
