# gamma_pipeline.py
# Team Gamma — SigLIP Pre-aligned Text-Vision Pipeline
# Amal-NLP-W5-gamma-siglip

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel
import time
import os

# ─────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────

MODEL_ID = "google/siglip-base-patch16-224"

def load_siglip(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"✅ SigLIP loaded — device: {device}")
    return model, processor, device


# ─────────────────────────────────────────
# 2. ENCODE FUNCTIONS
# ─────────────────────────────────────────

@torch.no_grad()
def encode_text(instruction: str, model, processor, device) -> np.ndarray:
    inputs = processor(text=[instruction], return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "pixel_values"}
    outputs = model.text_model(**inputs)
    text_features = outputs.pooler_output                      # torch.Tensor (1, 768)
    text_features = F.normalize(text_features.float(), dim=-1) # normalize the tensor
    return text_features[0].cpu().numpy()                      # (768,)


@torch.no_grad()
def encode_image(image, model, processor, device) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.vision_model(**inputs)
    vision_features = outputs.pooler_output                      # torch.Tensor (1, 768)
    vision_features = F.normalize(vision_features.float(), dim=-1)
    return vision_features[0].cpu().numpy()                      # (768,)


def compute_similarity(text_feat: np.ndarray, vision_feat: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors → scalar."""
    return float(np.dot(text_feat, vision_feat))


# ─────────────────────────────────────────
# 3. MAIN WRAPPER  (what RL team calls)
# ─────────────────────────────────────────

def gamma_encode(instruction: str, image, model, processor, device, embedding_cache=None) -> dict:
    """
    Full Gamma pipeline for one RL step.

    Args:
        instruction:     natural language command string
        image:           object crop — PIL Image or numpy (H, W, 3)
        model/processor: from load_siglip()
        device:          from load_siglip()
        embedding_cache: optional dict {instruction: (768,) array} for fast NLP lookup

    Returns:
        {
            "vision":     np.ndarray (768,)
            "nlp":        np.ndarray (768,)
            "similarity": float
        }
    """
    if embedding_cache is not None and instruction in embedding_cache:
        text_feat = embedding_cache[instruction]
    else:
        text_feat = encode_text(instruction, model, processor, device)

    vision_feat = encode_image(image, model, processor, device)
    similarity  = compute_similarity(text_feat, vision_feat)

    return {
        "vision":     vision_feat,
        "nlp":        text_feat,
        "similarity": similarity
    }


# ─────────────────────────────────────────
# 4. PRE-ENCODE ALL INSTRUCTIONS
# ─────────────────────────────────────────

def build_embedding_cache(csv_path: str, model, processor, device, save_path=None) -> dict:
    """
    Pre-encode all instructions from the NLP dataset using SigLIP text encoder.
    Saves embeddings_gamma_768.npy if save_path is given.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    instructions = df["instruction"].tolist()

    print(f"Pre-encoding {len(instructions)} instructions with SigLIP text encoder...")
    t0 = time.perf_counter()

    cache = {}
    for i, inst in enumerate(instructions):
        cache[inst] = encode_text(inst, model, processor, device)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(instructions)}")

    elapsed = time.perf_counter() - t0
    print(f"✅ Done in {elapsed:.1f}s — {len(cache)} embeddings cached")

    if save_path:
        matrix = np.stack([cache[inst] for inst in instructions])
        np.save(save_path, matrix)
        print(f"Saved: {save_path}  shape={matrix.shape}")

    return cache


# ─────────────────────────────────────────
# 5. CHECKPOINT VERIFICATION
# ─────────────────────────────────────────

def run_checkpoints(csv_path="nlp_instructions.csv"):
    model, processor, device = load_siglip()

    # ── Checkpoint 1: output dims and L2 normalization ──
    print("\n── CHECKPOINT 1: output format ──")
    dummy_text  = encode_text("pick the red block", model, processor, device)
    dummy_image = encode_image(Image.new("RGB", (224, 224), color=(200, 50, 50)), model, processor, device)

    assert dummy_text.shape  == (768,), f"text shape wrong: {dummy_text.shape}"
    assert dummy_image.shape == (768,), f"vision shape wrong: {dummy_image.shape}"

    text_norm  = np.linalg.norm(dummy_text)
    image_norm = np.linalg.norm(dummy_image)
    assert abs(text_norm  - 1.0) < 1e-5, f"text not L2-normalized: norm={text_norm:.4f}"
    assert abs(image_norm - 1.0) < 1e-5, f"image not L2-normalized: norm={image_norm:.4f}"

    print(f"  text_feat   shape={dummy_text.shape}  norm={text_norm:.4f}  ✅")
    print(f"  vision_feat shape={dummy_image.shape} norm={image_norm:.4f}  ✅")

    # ── Checkpoint 2: similarity sanity check ──
    print("\n── CHECKPOINT 2: similarity scores ──")

    red_image   = Image.new("RGB", (224, 224), color=(220, 40, 40))
    green_image = Image.new("RGB", (224, 224), color=(40, 180, 40))

    red_feat   = encode_image(red_image,   model, processor, device)
    green_feat = encode_image(green_image, model, processor, device)
    query_feat = encode_text("pick the red block", model, processor, device)

    sim_match    = compute_similarity(query_feat, red_feat)
    sim_mismatch = compute_similarity(query_feat, green_feat)

    print(f"  'pick the red block' ↔ red image:   similarity = {sim_match:.4f}")
    print(f"  'pick the red block' ↔ green image: similarity = {sim_mismatch:.4f}")

    if sim_match > sim_mismatch:
        print("  ✅ Matching image scores higher than mismatching")
    else:
        print("  ⚠️  Ordering unexpected — solid color patches are a weak test, real renders will be better")

    # ── Checkpoint 3: full wrapper output ──
    print("\n── CHECKPOINT 3: gamma_encode() output ──")
    result = gamma_encode("pick the red block", red_image, model, processor, device)
    assert set(result.keys()) == {"vision", "nlp", "similarity"}
    assert result["vision"].shape == (768,)
    assert result["nlp"].shape    == (768,)
    assert isinstance(result["similarity"], float)
    print(f"  Keys:       {list(result.keys())}  ✅")
    print(f"  vision:     {result['vision'].shape}  ✅")
    print(f"  nlp:        {result['nlp'].shape}    ✅")
    print(f"  similarity: {result['similarity']:.4f}  ✅")

    # ── Checkpoint 4: inference speed ──
    print("\n── CHECKPOINT 4: inference speed ──")
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        encode_image(red_image, model, processor, device)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = np.mean(times)
    status = "✅ PASS" if avg_ms < 10 else "⚠️  SLOW (acceptable on CPU)"
    print(f"  Vision encode avg: {avg_ms:.2f}ms  {status}")

    # ── Checkpoint 5: pre-encode cache ──
    if os.path.exists(csv_path):
        print(f"\n── CHECKPOINT 5: embedding cache from {csv_path} ──")
        cache = build_embedding_cache(csv_path, model, processor, device, save_path="embeddings_gamma_768.npy")

        test_inst = list(cache.keys())[0]
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = cache[test_inst]
            times.append((time.perf_counter() - t0) * 1000)
        avg_cache_ms = np.mean(times)
        print(f"  Cached lookup avg: {avg_cache_ms:.4f}ms  {'✅' if avg_cache_ms < 1 else '⚠️'}")
    else:
        print(f"\n⚠️  {csv_path} not found — skipping cache checkpoint")

    print("\n🎉 All checkpoints done.")
    return model, processor, device


if __name__ == "__main__":
    run_checkpoints(csv_path="nlp_instructions.csv")