#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV -> PKL with MatSciBERT material embeddings + similarity to concept phrases.

Material embedding logic:
- Encode each element symbol once with MatSciBERT to get an element embedding.
- For each row, compute the normalized concentration-weighted sum of present element embeddings.
- L2-normalize the resulting material vector.

Example:
python matscibert_preprocess_with_concepts.py \
  --input_directory data/in \
  --output_directory data/out \
  --num_workers 4 \
  --prompt_style composition \
  --extra_tags '{"process":"co-sputtering"}' \
  --concepts '{"conductivity":"high electrical conductivity", "dielectric":"dielectric material"}'
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS OpenMP quick unblock

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

_MODEL_NAME = "m3rg-iitd/matscibert"

# ---------- Periodic table ----------
PERIODIC_TABLE = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db",
    "Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]

# ---------- Prompt builder (kept for API stability; not used in new logic) ----------
def _row_to_prompt(
    row: pd.Series,
    element_cols: List[str],
    *,
    style: str = "composition",  # "formula" | "composition" | "explicit" | "sentence"
    round_dp: int = 3,
    extra_tags: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> Optional[str]:
    parts = []
    for el in element_cols:
        try:
            v = float(row.get(el, 0.0))
        except Exception:
            v = 0.0
        if v > 0:
            parts.append((el, v))
    if not parts:
        return None

    total = sum(v for _, v in parts)
    if total <= 0:
        return None
    parts = [(el, v / total) for el, v in parts]
    parts.sort(key=lambda x: x[0].lower())

    def fmt(x: float) -> str:
        return f"{x:.{round_dp}f}"

    if style == "formula":
        core = " ".join([f"{el} {fmt(frac)}" for el, frac in parts])
    elif style == "composition":
        core = "composition: " + ", ".join([f"{el}={fmt(frac)}" for el, frac in parts])
    elif style == "explicit":
        core = " ; ".join([f"Element {el} concentration {fmt(frac)}" for el, frac in parts])
    elif style == "sentence":
        core = "This sample is a composition: " + ", ".join([f"{el} {fmt(frac)}" for el, frac in parts]) + "."
    else:
        raise ValueError(f"Unknown style '{style}'")

    if extra_tags:
        tag_str = "; ".join(f"{k}={v}" for k, v in extra_tags.items())
        core = f"{core} [tags: {tag_str}]"

    return core

# ---------- Model helpers ----------
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

def _encode_texts(tokenizer, model, texts: List[str], device: str, max_length: int = 128) -> torch.Tensor:
    """Return L2-normalized embeddings for a list of texts."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled

def _encode_elements(tokenizer, model, elements: List[str], device: str, max_length: int = 16) -> Dict[str, np.ndarray]:
    """
    Encode each element symbol with a short contextual sentence to align with MatSciBERT pretraining.
    Example prompt: "O is a chemical element."
    """
    element_prompts = [f"{el} is a chemical element." for el in elements]
    embs = _encode_texts(tokenizer, model, element_prompts, device=device, max_length=max_length)
    embs = embs.detach().cpu().numpy()
    return {el: embs[i] for i, el in enumerate(elements)}

def _l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

def matscibert_embed_and_similarity(
    df: pd.DataFrame,
    concepts: Optional[Dict[str, str]] = None,   # label -> phrase
    *,
    element_cols: Optional[List[str]] = None,
    prompt_style: str = "composition",   # kept for API compatibility; not used for embedding now
    round_dp: int = 3,                   # kept for API compatibility
    extra_tags: Optional[Dict[str, Union[str, int, float, bool]]] = None,  # kept for API compatibility
    batch_size: int = 64,                # kept for API compatibility
    device: Optional[str] = None,
    max_length: int = 128,               # used for concept phrases; element max_length is smaller
    output_col: str = "Material_Vec",
) -> pd.DataFrame:
    """
    Append Material_Vec and optional Similarity_to_<label> columns.

    NEW MATERIAL EMBEDDING LOGIC:
    - Identify element columns (intersection with periodic table).
    - Encode each element symbol once to get its embedding.
    - For each row, compute the normalized concentration-weighted sum of its present element vectors.
    - L2-normalize each resulting material vector.
    """
    work_df = df.copy()

    # element columns
    if element_cols is None:
        element_cols = [c for c in work_df.columns if c in PERIODIC_TABLE]
    if not element_cols:
        raise ValueError("No element columns found in DataFrame (H..Og).")

    # device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # load model/tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME, add_pooling_layer=False).to(device).eval()

    # ---- 1) Encode elements once ----
    # Keep a fixed order for matrix math
    el_order = list(element_cols)
    el_vecs_dict = _encode_elements(tokenizer, model, el_order, device=device, max_length=16)
    # Stack into [E, d]
    elem_mat = np.stack([el_vecs_dict[el] for el in el_order], axis=0)  # L2-normalized rows

    # ---- 2) Build weighted sums per row ----
    # Extract concentrations as a matrix [N, E]; coerce to float, negatives -> 0, NaNs -> 0
    conc = work_df[el_order].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    conc = np.clip(conc, 0.0, None)  # no negative fractions

    # Normalize concentrations per row to sum to 1 (for rows with any positive concentration)
    row_sums = conc.sum(axis=1, keepdims=True)
    has_any = (row_sums.squeeze(-1) > 0.0)
    conc_norm = np.zeros_like(conc, dtype=float)
    conc_norm[has_any] = conc[has_any] / row_sums[has_any]

    # Weighted sum -> [N, d]
    mat_vecs = conc_norm @ elem_mat  # rows with no elements will be zeros
    # L2-normalize non-zero rows
    mat_vecs[has_any] = _l2_normalize_rows(mat_vecs[has_any])

    # Convert to list[Optional[list[float]]]
    out_vecs: List[Optional[List[float]]] = []
    for i in range(mat_vecs.shape[0]):
        if has_any[i]:
            out_vecs.append(mat_vecs[i].tolist())
        else:
            out_vecs.append(None)

    work_df[output_col] = out_vecs

    # ---- 3) Concept similarities (optional) ----
    if concepts:
        labels = list(concepts.keys())
        phrases = [concepts[k] for k in labels]
        # encode concept phrases (single batch; small)
        concept_mat = _encode_texts(tokenizer, model, phrases, device=device, max_length=max_length)  # [C, d], L2-norm
        concept_mat = concept_mat.cpu().numpy()

        # rows with valid vectors
        mat_mask = work_df[output_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
        if mat_mask.any():
            mat_mat = np.stack(work_df.loc[mat_mask, output_col].values, axis=0)  # [M, d], L2-normalized
            sims = mat_mat @ concept_mat.T  # cosine similarity
            for c_idx, label in enumerate(labels):
                col = f"Similarity_to_{label}"
                work_df[col] = np.nan
                work_df.loc[mat_mask, col] = sims[:, c_idx]
        else:
            for label in labels:
                work_df[f"Similarity_to_{label}"] = np.nan

    return work_df

# ---------- Logging ----------
def load_processed_files(log_path: str) -> set:
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def log_processed_file(log_path: str, filename: str):
    with open(log_path, "a") as f:
        f.write(filename + "\n")

# ---------- File loop ----------
def process_one_file(
    input_path: str,
    output_path: str,
    *,
    prompt_style: str,
    extra_tags: Optional[Dict[str, Union[str, int, float, bool]]],
    concepts: Optional[Dict[str, str]],
    batch_size: int,
    device: Optional[str],
    max_length: int,
):
    df = pd.read_csv(input_path)
    df_emb = matscibert_embed_and_similarity(
        df,
        concepts=concepts,
        element_cols=None,
        prompt_style=prompt_style,
        round_dp=3,
        extra_tags=extra_tags,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        output_col="Material_Vec",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_emb.to_pickle(output_path)

def process_all_files_in_directory(
    input_directory: str,
    output_directory: str,
    *,
    filename_suffix: str,
    num_workers: int,
    prompt_style: str,
    extra_tags: Optional[Dict[str, Union[str, int, float, bool]]],
    concepts: Optional[Dict[str, str]],
    batch_size: int,
    device: Optional[str],
    max_length: int,
    output_suffix: str,
):
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.join(output_directory, "processed_files.log")
    already = load_processed_files(log_path)

    filenames = [
        f for f in os.listdir(input_directory)
        if f.endswith(filename_suffix) and os.path.isfile(os.path.join(input_directory, f))
    ]

    futures = {}
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        for fn in filenames:
            if fn in already:
                continue
            in_path = os.path.join(input_directory, fn)
            base = os.path.splitext(fn)[0] + (output_suffix or "_with_matscibert")
            out_path = os.path.join(output_directory, base + ".pkl")
            futures[exe.submit(
                process_one_file,
                in_path,
                out_path,
                prompt_style=prompt_style,
                extra_tags=extra_tags,
                concepts=concepts,
                batch_size=batch_size,
                device=device,
                max_length=max_length,
            )] = fn

        for fut in as_completed(futures):
            fn = futures[fut]
            try:
                fut.result()
                log_processed_file(log_path, fn)
                print(f"Processed: {fn}")
            except Exception as e:
                print(f"[ERROR] {fn}: {e}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="MatSciBERT embeddings + concept similarity (CSV -> PKL).")
    p.add_argument("--input_directory", type=str, required=True, help="Input directory")
    p.add_argument("--output_directory", type=str, required=True, help="Output directory")
    p.add_argument("--filename_suffix", type=str, default="_material_system.csv", help="Files ending with this suffix will be processed")
    p.add_argument("--num_workers", type=int, default=2, help="Parallel workers (files)")
    p.add_argument("--prompt_style", type=str, default="composition",
                   choices=["formula","composition","explicit","sentence"], help="Prompt style")
    p.add_argument("--extra_tags", type=str, default="", help='JSON dict of extra tags, e.g. \'{"process":"co-sputtering"}\'')
    p.add_argument("--concepts", type=str, default="", help='JSON dict label->phrase, e.g. \'{"conductivity":"high electrical conductivity","dielectric":"dielectric material"}\'')
    p.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    p.add_argument("--device", type=str, default="", help="Force device: cpu|mps|cuda (auto if empty)")
    p.add_argument("--max_length", type=int, default=128, help="Max tokens per prompt")
    p.add_argument("--output_suffix", type=str, default="_with_matscibert", help="Suffix for output filenames (before .pkl)")
    return p.parse_args()

def main():
    args = parse_args()

    extra_tags = None
    if args.extra_tags:
        try:
            extra_tags = json.loads(args.extra_tags)
            if not isinstance(extra_tags, dict):
                raise ValueError("extra_tags must be a JSON object")
        except Exception as e:
            raise SystemExit(f"Failed to parse --extra_tags JSON: {e}")

    concepts = None
    if args.concepts:
        try:
            concepts = json.loads(args.concepts)
            if not isinstance(concepts, dict):
                raise ValueError("concepts must be a JSON object mapping label->phrase")
        except Exception as e:
            raise SystemExit(f"Failed to parse --concepts JSON: {e}")

    device = args.device if args.device else None

    process_all_files_in_directory(
        input_directory=args.input_directory,
        output_directory=args.output_directory,
        filename_suffix=args.filename_suffix,
        num_workers=args.num_workers,
        prompt_style=args.prompt_style,
        extra_tags=extra_tags,
        concepts=concepts,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
        output_suffix=args.output_suffix,
    )

if __name__ == "__main__":
    main()