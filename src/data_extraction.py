"""
data_extraction.py
==================
Step 1: Extract (description, BioScript) pairs from OpenBioSet dataset folders.

Usage:
    from src.data_extraction import extract_dataset, split_dataset
    good_records, error_records = extract_dataset("/path/to/Dataset")
    train, val, test = split_dataset(good_records)
"""

import json
import random
from pathlib import Path


# ── File Finders ──────────────────────────────────────────────────────────────

def find_file_by_suffix(folder: Path, extension: str) -> Path | None:
    """
    Find a file ending with a given extension inside a folder.
    Handles Kaggle's double-extension issue (e.g. description.txt.txt).
    """
    matches = [
        f for f in folder.iterdir()
        if f.is_file() and f.name.endswith(extension)
    ]
    if not matches:
        return None
    # Prefer file whose name contains the folder name
    for f in matches:
        if folder.name in f.name:
            return f
    return matches[0]


def find_description_file(folder: Path) -> Path | None:
    """
    Find the description text file inside a folder.
    Handles: description.txt OR description.txt.txt
    """
    candidates = [
        f for f in folder.iterdir()
        if f.is_file()
        and "description" in f.name.lower()
        and f.name.lower().endswith(".txt")
    ]
    return candidates[0] if candidates else None


# ── Record Extraction ─────────────────────────────────────────────────────────

def extract_record(folder: Path) -> dict:
    """
    Extract one complete record from a protocol folder.

    Args:
        folder: Path to the protocol folder

    Returns:
        dict with keys: folder_name, status, errors, description,
                        bioscript, json_metadata, ir_output, dot_output
    """
    record = {
        "folder_name"  : folder.name,
        "status"       : "ok",
        "errors"       : [],
        "warnings"     : [],
        "description"  : None,
        "bioscript"    : None,
        "json_metadata": None,
        "ir_output"    : None,
        "dot_output"   : None,
    }

    # 1. description.txt
    desc_path = find_description_file(folder)
    if desc_path:
        text = desc_path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            record["description"] = text
        else:
            record["errors"].append("description.txt is empty")
    else:
        record["errors"].append("No description file found")

    # 2. .bs file
    bs_path = find_file_by_suffix(folder, ".bs")
    if bs_path:
        code = bs_path.read_text(encoding="utf-8", errors="replace").strip()
        if code:
            record["bioscript"] = code
        else:
            record["errors"].append(f"{bs_path.name} is empty")
    else:
        record["errors"].append("No .bs file found")

    # 3. .json metadata (optional)
    json_path = find_file_by_suffix(folder, ".json")
    if json_path:
        try:
            record["json_metadata"] = json.loads(
                json_path.read_text(encoding="utf-8", errors="replace")
            )
        except json.JSONDecodeError as e:
            record["warnings"].append(f"JSON parse error: {e}")
    else:
        record["warnings"].append("No .json metadata file found")

    # 4. output.ir (optional)
    ir_path = find_file_by_suffix(folder, ".ir")
    if ir_path:
        record["ir_output"] = ir_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()

    # 5. output.dot (optional)
    dot_path = find_file_by_suffix(folder, ".dot")
    if dot_path:
        record["dot_output"] = dot_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()

    if record["errors"]:
        record["status"] = "error"

    return record


# ── Dataset Extraction ────────────────────────────────────────────────────────

def extract_dataset(dataset_path: str) -> tuple[list, list]:
    """
    Extract all records from the OpenBioSet dataset directory.

    Args:
        dataset_path: Path to the Dataset folder containing protocol subfolders

    Returns:
        Tuple of (good_records, error_records)
    """
    root = Path(dataset_path)
    assert root.exists(), f"Dataset path not found: {dataset_path}"

    all_folders = sorted([
        f for f in root.iterdir()
        if f.is_dir() and not f.name.startswith(".")
    ])

    print(f"Scanning {len(all_folders)} folders...")

    all_records   = [extract_record(f) for f in all_folders]
    good_records  = [r for r in all_records if r["status"] == "ok"]
    error_records = [r for r in all_records if r["status"] == "error"]

    print(f"✅ Valid pairs   : {len(good_records)}")
    print(f"❌ Error records : {len(error_records)}")

    if error_records:
        for r in error_records:
            print(f"   Excluded: {r['folder_name']} — {r['errors'][0]}")

    return good_records, error_records


# ── Train/Val/Test Split ──────────────────────────────────────────────────────

def split_dataset(
    records: list,
    train_ratio: float = 0.80,
    val_ratio: float   = 0.10,
    seed: int          = 42,
) -> tuple[list, list, list]:
    """
    Split records into train/val/test sets.

    Args:
        records:     List of extracted records
        train_ratio: Fraction for training (default 0.80)
        val_ratio:   Fraction for validation (default 0.10)
        seed:        Random seed for reproducibility

    Returns:
        Tuple of (train_records, val_records, test_records)
    """
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val:]

    print(f"Split (seed={seed}): Train={len(train)} | Val={len(val)} | Test={len(test)}")
    return train, val, test


# ── JSONL Utilities ───────────────────────────────────────────────────────────

def save_jsonl(data: list, path: str) -> None:
    """Save a list of dicts as a JSONL file (one JSON object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {p.name}: {len(data)} examples")


def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./Dataset"
    output_dir   = sys.argv[2] if len(sys.argv) > 2 else "./dataset"

    good_records, error_records = extract_dataset(dataset_path)
    train, val, test = split_dataset(good_records)

    # Save raw extraction
    with open(f"{output_dir}/raw_extraction.json", "w") as f:
        json.dump(good_records + error_records, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Results saved to {output_dir}/")
