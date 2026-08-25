"""
evaluation.py
=============
Step 4: Evaluation metrics and model comparison.

Evaluates fine-tuned models on held-out test protocols using:
  - Structural accuracy (8 BioScript element checks)
  - BLEU-1 score (word-level overlap with reference)

Usage:
    from src.evaluation import evaluate_model, print_comparison_table
    results = evaluate_model("Qwen-7B", adapter_path, test_data)
    print_comparison_table(all_results)
"""

import json
import os
import torch
from pathlib import Path


# ── Structural Accuracy ───────────────────────────────────────────────────────

def _is_repetitive(text: str, threshold: float = 0.1) -> bool:
    """Check if generated text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False
    for i in range(len(words) - 3):
        phrase = " ".join(words[i:i+3])
        if text.count(phrase) > len(words) * threshold:
            return True
    return False


def check_structure(code: str) -> dict:
    """
    Check if generated BioScript has correct structural elements.

    Checks 8 elements:
        has_module, has_manifest, has_instructions, has_dispense,
        has_mix, has_braces, no_repetition, non_empty

    Returns:
        Dict of checks with boolean values + 'score' (0.0 to 1.0)
    """
    full  = code.lower()
    lines = full.split("\n")

    checks = {
        "has_module"      : any("module" in l for l in lines),
        "has_manifest"    : any("manifest" in l for l in lines),
        "has_instructions": "instructions:" in full,
        "has_dispense"    : "dispense" in full,
        "has_mix"         : "mix" in full,
        "has_braces"      : "{" in code and "}" in code,
        "no_repetition"   : not _is_repetitive(code),
        "non_empty"       : len(code.strip()) > 50,
    }

    num_checks      = len(checks)
    checks["score"] = sum(checks.values()) / num_checks
    return checks


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU-1 score (word-level unigram overlap).

    Note: Multiple valid BioScript implementations exist for the same
    protocol, so BLEU alone is insufficient — use with structural accuracy.

    Args:
        reference:  Reference BioScript code
        hypothesis: Generated BioScript code

    Returns:
        BLEU-1 score between 0.0 and 1.0
    """
    ref_words = set(reference.lower().split())
    hyp_words = hypothesis.lower().split()
    if not hyp_words:
        return 0.0
    return sum(1 for w in hyp_words if w in ref_words) / len(hyp_words)


# ── Model Evaluation ──────────────────────────────────────────────────────────

def evaluate_model(
    model_name  : str,
    adapter_path: str,
    test_data   : list,
    system_prompt: str,
    max_seq_len : int   = 2048,
    max_new_tokens: int = 1024,
    temperature : float = 0.7,
    hf_token    : str   = None,
) -> list:
    """
    Evaluate a fine-tuned model on the test set.

    Args:
        model_name   : Display name for the model
        adapter_path : Path to saved LoRA adapter
        test_data    : List of test examples (chat format)
        system_prompt: System prompt used during training
        max_seq_len  : Maximum sequence length
        max_new_tokens: Maximum tokens to generate
        temperature  : Sampling temperature
        hf_token     : HuggingFace API token

    Returns:
        List of result dicts with generated code and metrics
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Install unsloth: pip install unsloth[kaggle-new]")

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = adapter_path,
        max_seq_length = max_seq_len,
        dtype          = None,
        load_in_4bit   = True,
        token          = hf_token,
    )
    FastLanguageModel.for_inference(model)

    results = []
    print(f"\n{'#':<4} {'ID':<42} {'Struct%':<10} {'BLEU':<8} {'Status'}")
    print("-" * 72)

    for i, example in enumerate(test_data):
        folder_name  = example["id"].replace("__original", "")
        description  = example["messages"][1]["content"]
        reference_bs = example["messages"][2]["content"]

        # Generate
        generated = _generate(
            model, tokenizer, description, system_prompt,
            max_seq_len, max_new_tokens, temperature
        )

        # Metrics
        struct_checks = check_structure(generated)
        bleu_score    = compute_bleu(reference_bs, generated)
        is_rep        = not struct_checks["no_repetition"]
        status        = "⚠️" if is_rep else "✅"

        short_id = folder_name[:40] + ".." if len(folder_name) > 40 else folder_name
        print(f"{i+1:<4} {short_id:<42} "
              f"{struct_checks['score']*100:<10.1f} "
              f"{bleu_score:<8.3f} {status}")

        results.append({
            "id"           : folder_name,
            "generated"    : generated,
            "reference"    : reference_bs,
            "struct_checks": struct_checks,
            "bleu_score"   : bleu_score,
            "struct_score" : struct_checks["score"],
            "is_repetitive": is_rep,
        })

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"\n✅ {model_name} complete — GPU freed.")

    return results


def _generate(
    model, tokenizer, description, system_prompt,
    max_seq_len, max_new_tokens, temperature
) -> str:
    """Internal generation function."""
    desc = description[:1500] + "..." if len(description) > 1500 else description
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": desc},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    prompt += "module "

    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True,
        max_length=max_seq_len - max_new_tokens,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens    = max_new_tokens,
            temperature       = temperature,
            do_sample         = True,
            top_p             = 0.9,
            top_k             = 50,
            repetition_penalty= 1.3,
            pad_token_id      = tokenizer.eos_token_id,
            eos_token_id      = tokenizer.eos_token_id,
        )

    return "module " + tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


# ── Comparison Table ──────────────────────────────────────────────────────────

def compute_summary(results: list) -> dict:
    """Compute aggregate metrics from evaluation results."""
    total = len(results)
    return {
        "avg_struct"  : sum(r["struct_score"] for r in results) / total * 100,
        "avg_bleu"    : sum(r["bleu_score"] for r in results) / total,
        "has_module"  : sum(1 for r in results if r["struct_checks"]["has_module"]),
        "has_manifest": sum(1 for r in results if r["struct_checks"]["has_manifest"]),
        "has_instruct": sum(1 for r in results if r["struct_checks"]["has_instructions"]),
        "has_dispense": sum(1 for r in results if r["struct_checks"]["has_dispense"]),
        "non_rep"     : sum(1 for r in results if not r["is_repetitive"]),
        "total"       : total,
    }


def print_comparison_table(all_results: dict) -> dict:
    """
    Print a formatted comparison table for all evaluated models.

    Args:
        all_results: Dict mapping model_name → list of result dicts

    Returns:
        Summary dict mapping model_name → aggregate metrics
    """
    print("\n" + "=" * 95)
    print("FINAL RESULTS: BioGPT Model Comparison")
    print("=" * 95)
    print(f"{'Model':<28} {'Struct%':>8} {'BLEU':>8} {'Module':>8} "
          f"{'Manifest':>10} {'Instruct':>10} {'Dispense':>10} {'Non-Rep':>9}")
    print("-" * 95)

    summary = {}
    for model_name, results in all_results.items():
        s = compute_summary(results)
        summary[model_name] = s
        t = s["total"]
        print(f"{model_name:<28} {s['avg_struct']:>8.1f} {s['avg_bleu']:>8.3f} "
              f"{s['has_module']}/{t:>5} {s['has_manifest']}/{t:>7} "
              f"{s['has_instruct']}/{t:>7} {s['has_dispense']}/{t:>7} "
              f"{s['non_rep']}/{t:>6}")

    print("=" * 95)

    # Best model (weighted score)
    best = max(summary, key=lambda x: (
        summary[x]["avg_struct"] * 0.4 +
        summary[x]["avg_bleu"] * 100 * 0.3 +
        (summary[x]["has_instruct"] / summary[x]["total"]) * 100 * 0.3
    ))

    print(f"\n🏆 Best Model: {best}")
    print(f"   Structural Accuracy : {summary[best]['avg_struct']:.1f}%")
    print(f"   BLEU-1 Score        : {summary[best]['avg_bleu']:.3f}")
    print(f"   Instructions        : {summary[best]['has_instruct']}/{summary[best]['total']}")

    return summary


def save_results(all_results: dict, output_dir: str) -> None:
    """Save all evaluation results to JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for model_name, results in all_results.items():
        # Save per-model results
        with open(out / f"{model_name}_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary[model_name] = compute_summary(results)

    # Save comparison summary
    with open(out / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to: {output_dir}")
