"""
augmentation.py
===============
Step 2: Data augmentation using Groq API (Llama-3.3-70B).

Paraphrases each protocol description into 3 style variants:
  - Variant 1: Concise and technical
  - Variant 2: Step-by-step focused
  - Variant 3: Context and rationale focused

The BioScript code stays identical for all variants.

Usage:
    from src.augmentation import augment_dataset
    augmented = augment_dataset(train_records, groq_api_key, progress_file)
"""

import json
import time
from pathlib import Path

from groq import Groq


# ── Constants ─────────────────────────────────────────────────────────────────

GROQ_MODEL    = "llama-3.3-70b-versatile"
SLEEP_BETWEEN = 3   # seconds between API calls
NUM_VARIANTS  = 3   # paraphrases per description


# ── Paraphrase Function ───────────────────────────────────────────────────────

def paraphrase_all_variants(
    client: Groq,
    description: str,
    folder_name: str,
    max_chars: int = 2500,
) -> list[str]:
    """
    Generate 3 paraphrased variants of a protocol description.

    Args:
        client:      Groq API client
        description: Original protocol description
        folder_name: Protocol folder name (for error messages)
        max_chars:   Max characters to send (avoids token limit errors)

    Returns:
        List of 3 paraphrased descriptions.
        Falls back to [original, original, original] on failure.
    """
    desc   = description[:max_chars] + "..." if len(description) > max_chars else description
    prompt = f"""Rewrite this biology lab protocol description in 3 styles.
Keep all scientific details identical.

Return ONLY this exact JSON format with no other text:
{{"v1": "concise technical version", "v2": "step by step version", "v3": "context focused version"}}

DESCRIPTION:
{desc}"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.5,
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            # Extract JSON object
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            data     = json.loads(text)
            variants = [
                data.get("v1", ""),
                data.get("v2", ""),
                data.get("v3", ""),
            ]

            if all(len(v) > 50 for v in variants):
                return variants

            print(f"    ⚠️  Short response (attempt {attempt+1}), retrying...")

        except json.JSONDecodeError:
            print(f"    ⚠️  JSON parse failed (attempt {attempt+1}), retrying...")
            time.sleep(5)
        except Exception as e:
            wait = (2 ** attempt) * 10
            print(f"    ⚠️  API error: {str(e)[:60]} — waiting {wait}s...")
            time.sleep(wait)

    print(f"    ❌ All retries failed for {folder_name} — using fallback")
    return [description, description, description]


# ── Main Augmentation Function ────────────────────────────────────────────────

def augment_dataset(
    train_records: list,
    groq_api_key: str,
    progress_file: str = "./dataset/progress.json",
) -> list[tuple[str, int, str]]:
    """
    Augment training records with paraphrased descriptions.

    Args:
        train_records:  List of training records (from data_extraction.py)
        groq_api_key:   Groq API key
        progress_file:  Path to save/load progress (for resuming)

    Returns:
        List of (folder_name, variant_idx, paraphrased_description) tuples
    """
    client    = Groq(api_key=groq_api_key)
    prog_path = Path(progress_file)
    prog_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress
    progress = {}
    if prog_path.exists():
        with open(prog_path) as f:
            progress = json.load(f)

    already_done = len([k for k in progress if k.endswith("__v0")])
    print(f"Starting augmentation...")
    print(f"  Training records  : {len(train_records)}")
    print(f"  Already completed : {already_done}")
    print(f"  Remaining         : {len(train_records) - already_done}")

    augmented_variants = []
    completed = skipped = 0

    for record in train_records:
        folder_name = record["folder_name"]
        description = record["description"]
        prog_key    = f"{folder_name}__v0"

        # Resume from cache
        if prog_key in progress:
            for v_idx in range(NUM_VARIANTS):
                k = f"{folder_name}__v{v_idx}"
                if k in progress:
                    augmented_variants.append((folder_name, v_idx, progress[k]))
            skipped += 1
            continue

        # Generate new variants
        variants = paraphrase_all_variants(client, description, folder_name)

        for v_idx, variant_text in enumerate(variants):
            k = f"{folder_name}__v{v_idx}"
            progress[k] = variant_text
            augmented_variants.append((folder_name, v_idx, variant_text))

        # Save progress after each record
        with open(prog_path, "w") as f:
            json.dump(progress, f, indent=2)

        completed += 1

        if completed % 10 == 0:
            print(f"  Progress: {completed + skipped}/{len(train_records)} records")

        time.sleep(SLEEP_BETWEEN)

    print(f"\n✅ Augmentation complete!")
    print(f"  New variants    : {completed * NUM_VARIANTS}")
    print(f"  Cached variants : {skipped * NUM_VARIANTS}")
    print(f"  Total variants  : {len(augmented_variants)}")

    return augmented_variants


# ── Build Augmented Training Set ──────────────────────────────────────────────

def build_augmented_training_set(
    train_records: list,
    augmented_variants: list,
    system_prompt: str,
    seed: int = 42,
) -> list:
    """
    Build the final augmented training set combining originals + variants.

    Args:
        train_records:      Original training records
        augmented_variants: List of (folder_name, v_idx, paraphrased_desc)
        system_prompt:      System prompt for chat format
        seed:               Random seed for shuffling

    Returns:
        List of chat-format training examples
    """
    import random
    random.seed(seed)

    bs_lookup = {r["folder_name"]: r["bioscript"] for r in train_records}
    aug_train = []

    # Add originals
    for record in train_records:
        aug_train.append({
            "id": record["folder_name"] + "__original",
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": record["description"]},
                {"role": "assistant", "content": record["bioscript"]},
            ],
            "source": "original",
        })

    # Add augmented variants
    for folder_name, v_idx, paraphrased in augmented_variants:
        bs = bs_lookup.get(folder_name, "")
        if not bs:
            continue
        aug_train.append({
            "id": f"{folder_name}__v{v_idx+1}",
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": paraphrased},
                {"role": "assistant", "content": bs},
            ],
            "source": f"augmented_v{v_idx+1}",
        })

    random.shuffle(aug_train)

    print(f"✅ Augmented training set: {len(aug_train)} examples")
    print(f"  Originals : {len(train_records)}")
    print(f"  Augmented : {len(augmented_variants)}")

    return aug_train


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from src.data_extraction import extract_dataset, split_dataset

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    assert GROQ_API_KEY, "Set GROQ_API_KEY environment variable"

    good_records, _ = extract_dataset("./Dataset")
    train, val, test = split_dataset(good_records)

    variants = augment_dataset(train, GROQ_API_KEY)
    print(f"Done! {len(variants)} variants generated.")
