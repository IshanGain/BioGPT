"""
inference.py
============
Generation function for BioScript code from natural language descriptions.

Usage:
    from src.inference import BioGPTInference

    biogpt = BioGPTInference("./models/qwen7b_5ep/lora_adapter")
    code   = biogpt.generate("Perform PCR amplification of DNA...")
    print(code)
"""

import os
import torch
from pathlib import Path


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are BioGPT, an expert biological protocol compiler. "
    "Given a natural language description of a biological laboratory protocol, "
    "generate syntactically correct BioScript (.bs) code.\n\n"
    "You MUST always generate ALL sections in this exact order:\n"
    "1. module declarations\n"
    "2. manifest declarations\n"
    "3. instructions: keyword\n"
    "4. operations (dispense, mix, heat, detect, dispose)\n\n"
    "CRITICAL: Output ONLY valid BioScript code.\n"
    "DO NOT output JSON, XML, or any other format.\n\n"
    "STRICT FORMAT EXAMPLE:\n"
    "module myModule\n\n"
    "manifest Reagent1\n"
    "manifest Reagent2\n\n"
    "instructions:\n\n"
    "// Step 1: description\n"
    "var1 = dispense Reagent1 into $1 for 5s\n"
    "var2 = dispense Reagent2 into $2 for 5s\n"
    "mix1 = mix var1 with var2 for 30s\n"
    "result = detect fluorescence on mix1 for 10s\n\n"
    "RULES:\n"
    "- Always include instructions: section with actual operations\n"
    "- Duration format: 5s or 5m (never 5h)\n"
    "- Temperature format: 37c, 95c, 4c\n"
    "- Output ONLY valid BioScript code. No explanations."
)


# ── BioGPT Inference Class ────────────────────────────────────────────────────

class BioGPTInference:
    """
    BioGPT inference wrapper for generating BioScript from natural language.

    Example:
        biogpt = BioGPTInference("./models/qwen7b_5ep/lora_adapter")
        code   = biogpt.generate("PCR amplification protocol...")
        print(code)
    """

    def __init__(
        self,
        adapter_path  : str,
        max_seq_length: int = 2048,
        hf_token      : str = None,
        device        : str = "auto",
    ):
        """
        Load the fine-tuned BioGPT model.

        Args:
            adapter_path  : Path to saved LoRA adapter directory
            max_seq_length: Maximum sequence length
            hf_token      : HuggingFace API token (optional)
            device        : Device to load model on ('auto', 'cuda', 'cpu')
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth not installed. Install with:\n"
                "pip install unsloth[kaggle-new]"
            )

        print(f"Loading BioGPT from: {adapter_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name     = adapter_path,
            max_seq_length = max_seq_length,
            dtype          = None,
            load_in_4bit   = True,
            token          = hf_token or os.environ.get("HF_TOKEN"),
        )
        FastLanguageModel.for_inference(self.model)
        self.max_seq_length = max_seq_length

        print(f"✅ BioGPT loaded successfully!")

    def generate(
        self,
        description    : str,
        max_new_tokens : int   = 1024,
        temperature    : float = 0.7,
        top_p          : float = 0.9,
        top_k          : int   = 50,
        repetition_penalty: float = 1.3,
    ) -> str:
        """
        Generate BioScript code from a natural language protocol description.

        Args:
            description       : Natural language protocol description
            max_new_tokens    : Maximum tokens to generate
            temperature       : Sampling temperature (higher = more creative)
            top_p             : Nucleus sampling probability
            top_k             : Top-k sampling
            repetition_penalty: Penalty for repeated tokens (>1.0 reduces repetition)

        Returns:
            Generated BioScript code as a string
        """
        # Truncate long descriptions
        desc = description[:1500] + "..." if len(description) > 1500 else description

        # Format prompt using chat template
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": desc},
        ]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize              = False,
            add_generation_prompt = True,
        )
        # Seed with "module " to force BioScript format
        prompt += "module "

        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = self.max_seq_length - max_new_tokens,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens     = max_new_tokens,
                temperature        = temperature,
                do_sample          = True,
                top_p              = top_p,
                top_k              = top_k,
                repetition_penalty = repetition_penalty,
                pad_token_id       = self.tokenizer.eos_token_id,
                eos_token_id       = self.tokenizer.eos_token_id,
            )

        generated = "module " + self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens          = True,
            clean_up_tokenization_spaces = False,
        ).strip()

        return generated

    def unload(self):
        """Free GPU memory after inference."""
        import gc
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Model unloaded — GPU memory freed")


# ── Standalone Generation Function ───────────────────────────────────────────

def generate_bioscript(
    description  : str,
    adapter_path : str,
    hf_token     : str = None,
    **kwargs,
) -> str:
    """
    One-shot generation function (loads model, generates, unloads).

    Use BioGPTInference class for multiple generations (more efficient).

    Args:
        description : Natural language protocol description
        adapter_path: Path to saved LoRA adapter
        hf_token    : HuggingFace API token
        **kwargs    : Generation parameters passed to BioGPTInference.generate()

    Returns:
        Generated BioScript code
    """
    biogpt  = BioGPTInference(adapter_path, hf_token=hf_token)
    code    = biogpt.generate(description, **kwargs)
    biogpt.unload()
    return code


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "./models/qwen7b_5ep/lora_adapter"
    description  = sys.argv[2] if len(sys.argv) > 2 else (
        "PCR amplification protocol: Mix DNA template with primers "
        "and PCR master mix, heat to 95C for denaturation, cycle "
        "through annealing at 60C and extension at 72C, detect "
        "amplification by fluorescence."
    )

    print(f"Adapter : {adapter_path}")
    print(f"Input   : {description[:100]}...")
    print()

    code = generate_bioscript(description, adapter_path)
    print("Generated BioScript:")
    print("=" * 50)
    print(code)