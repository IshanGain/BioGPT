"""
training.py
===========
Step 3: QLoRA fine-tuning using Unsloth.

Trains a model on the augmented BioScript dataset using
4-bit quantization and LoRA adapters.

Usage:
    from src.training import train_model
    train_model(
        model_id    = "Qwen/Qwen2.5-Coder-3B-Instruct",
        output_name = "qwen3b_3ep",
        train_data  = train_examples,
        val_data    = val_examples,
        epochs      = 3,
    )
"""

import json
import os
import torch
from datetime import datetime
from pathlib import Path


# ── Default Training Config ───────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "max_seq_length": 2048,
    "lora_rank"     : 16,
    "lora_alpha"    : 32,
    "lora_dropout"  : 0.05,
    "learning_rate" : 2e-4,
    "warmup_ratio"  : 0.05,
    "random_seed"   : 42,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


# ── Training Function ─────────────────────────────────────────────────────────

def train_model(
    model_id    : str,
    output_name : str,
    train_data  : list,
    val_data    : list,
    output_dir  : str  = "./models",
    epochs      : int  = 3,
    batch_size  : int  = 2,
    grad_accum  : int  = 4,
    hf_token    : str  = None,
    config      : dict = None,
) -> dict:
    """
    Fine-tune a model using QLoRA (Unsloth).

    Args:
        model_id    : HuggingFace model ID
        output_name : Name for the output folder
        train_data  : List of training examples (chat format)
        val_data    : List of validation examples (chat format)
        output_dir  : Base directory for saving models
        epochs      : Number of training epochs
        batch_size  : Per-device batch size
        grad_accum  : Gradient accumulation steps
        hf_token    : HuggingFace API token
        config      : Training config (uses DEFAULT_CONFIG if None)

    Returns:
        Training summary dict
    """
    # Merge config
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Import here to avoid issues if unsloth not installed
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        raise ImportError(
            f"Required package not found: {e}\n"
            "Install with: pip install unsloth[kaggle-new] trl datasets"
        )

    out_dir = Path(output_dir) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {model_id}")
    print(f"Epochs  : {epochs} | Batch: {batch_size} × {grad_accum} = {batch_size*grad_accum}")
    print(f"Output  : {out_dir}")
    print(f"{'='*60}")

    # ── Load Model ────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_id,
        max_seq_length = cfg["max_seq_length"],
        dtype          = None,
        load_in_4bit   = True,
        token          = hf_token,
    )

    # Verify tokenization preserves spaces/newlines
    test = "module pcr\n\nmanifest DNA\n\ninstructions:\n\nvar1 = dispense DNA into $1 for 5s"
    decoded = tokenizer.decode(
        tokenizer(test, return_tensors="pt")["input_ids"][0],
        skip_special_tokens=True
    )
    assert " " in decoded, f"Tokenizer strips spaces for {model_id}!"
    assert "\n" in decoded, f"Tokenizer strips newlines for {model_id}!"
    print(f"✅ Tokenization verified — spaces and newlines preserved")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r                        = cfg["lora_rank"],
        target_modules           = cfg["target_modules"],
        lora_alpha               = cfg["lora_alpha"],
        lora_dropout             = cfg["lora_dropout"],
        bias                     = "none",
        use_gradient_checkpointing = "unsloth",
        random_state             = cfg["random_seed"],
    )
    model.print_trainable_parameters()

    # ── Format Dataset ────────────────────────────────────────────────────────
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize              = False,
            add_generation_prompt = False,
        )
        return {"text": text}

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    val_dataset   = Dataset.from_list([format_example(ex) for ex in val_data])

    # Verify formatting
    sample = train_dataset[0]["text"]
    assert " " in sample and "\n" in sample, "Formatting broke spaces/newlines!"
    print(f"✅ Dataset formatted: {len(train_dataset)} train, {len(val_dataset)} val")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model        = model,
        tokenizer    = tokenizer,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        args         = SFTConfig(
            output_dir                  = str(out_dir / "checkpoints"),
            dataset_text_field          = "text",
            max_seq_length              = cfg["max_seq_length"],
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = grad_accum,
            learning_rate               = cfg["learning_rate"],
            warmup_ratio                = cfg["warmup_ratio"],
            lr_scheduler_type           = "cosine",
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 10,
            eval_strategy               = "steps",
            eval_steps                  = 50,
            save_strategy               = "steps",
            save_steps                  = 50,
            save_total_limit            = 2,
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_loss",
            optim                       = "adamw_8bit",
            report_to                   = "none",
            packing                     = False,
            dataset_num_proc            = 2,
            seed                        = cfg["random_seed"],
        ),
    )

    started = datetime.now()
    print(f"\n🚀 Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")

    trainer.train()

    finished = datetime.now()
    duration = (finished - started).seconds // 60
    print(f"\n✅ Finished: {finished.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {duration} minutes")

    # ── Save Adapter ──────────────────────────────────────────────────────────
    adapter_path = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    size = sum(f.stat().st_size for f in adapter_path.rglob("*")) / 1e6
    print(f"✅ Adapter saved: {size:.1f} MB → {adapter_path}")

    # ── Save Summary ──────────────────────────────────────────────────────────
    summary = {
        "model_id"    : model_id,
        "output_name" : output_name,
        "epochs"      : epochs,
        "batch_size"  : batch_size,
        "grad_accum"  : grad_accum,
        "duration_min": duration,
        "lora_rank"   : cfg["lora_rank"],
        "lora_alpha"  : cfg["lora_alpha"],
        "train_size"  : len(train_data),
        "val_size"    : len(val_data),
        "finished_at" : finished.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"✅ GPU memory freed")

    return summary
