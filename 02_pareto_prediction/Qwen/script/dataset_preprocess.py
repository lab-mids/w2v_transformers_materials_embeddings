#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV -> PKL with material embeddings + similarity to concept phrases,
using Blablador embeddings

Material embedding logic (unchanged conceptually):
- Encode each element symbol once via Blablador embeddings to get an element embedding.
- For each row, compute the normalized concentration-weighted sum of present element embeddings.
- L2-normalize the resulting material vector.

Blablador backend:
- Calls an OpenAI-compatible /v1/embeddings endpoint.
- Uses a fixed base URL, model, and API key (hardcoded for local use).

Example:
python matscibert_preprocess_with_concepts_blablador.py \
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
import requests

#----------- Blablador config (from env; safe for version control) ----------
BLABLADOR_BASE_URL = os.getenv("BLABLADOR_BASE_URL", "https://api.helmholtz-blablador.fz-juelich.de/v1")
BLABLADOR_MODEL = os.getenv("BLABLADOR_MODEL", "alias-embeddings")
BLABLADOR_API_KEY = os.getenv("BLABLADOR_API_KEY", "")

if not BLABLADOR_API_KEY:
    raise RuntimeError("BLABLADOR_API_KEY is not set. Please export it or provide it via Snakemake/env.")
# -------------------------------------------------------------------------------

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

# ---------- Prompt builder (kept for API stability; not used in embedding logic) ----------
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

# ---------- Blablador embedding helpers ----------
def _encode_texts_blablador(
    texts: List[str],
    *,
    base_url: str = BLABLADOR_BASE_URL,
    model: str = BLABLADOR_MODEL,
    api_key: str = BLABLADOR_API_KEY,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    Call Blablador /v1/embeddings and return an [N, d] numpy array of L2-normalized embeddings.
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_vecs: List[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        batch = texts[start:start + chunk_size]
        payload = {"model": model, "input": batch}

        r = requests.post(url, headers=headers, json=payload, timeout=120)

        if r.status_code != 200:
            # helpful error
            try:
                err_json = r.json()
                raise RuntimeError(f"Embeddings HTTP {r.status_code}. Response JSON: {err_json}")
            except Exception:
                raise RuntimeError(f"Embeddings HTTP {r.status_code}. Text: {r.text[:500]}")

        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Cannot parse JSON from embeddings response: {e}. Raw: {r.text[:500]}")

        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            batch_vecs = [np.asarray(item["embedding"], dtype=np.float32) for item in data["data"]]
        elif isinstance(data, dict) and "embeddings" in data and isinstance(data["embeddings"], list):
            batch_vecs = [np.asarray(e, dtype=np.float32) for e in data["embeddings"]]
        elif isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"Embeddings error: {data['error']}")
        else:
            raise RuntimeError(f"Unexpected embeddings schema: {str(data)[:500]}")

        arr = np.stack(batch_vecs, axis=0)
        # L2-normalize rows
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        all_vecs.append(arr)

    return np.concatenate(all_vecs, axis=0)

def _encode_elements_blablador(elements: List[str]) -> Dict[str, np.ndarray]:
    """
    Encode each element symbol via Blablador using a short contextual sentence.
    Example prompt: "O is a chemical element."
    """
    prompts = [f"{el} is a chemical element." for el in elements]
    embs = _encode_texts_blablador(prompts)  # [E, d], L2-normalized rows
    return {el: embs[i] for i, el in enumerate(elements)}

def _l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

def matscibert_embed_and_similarity(  # name kept for API compatibility
    df: pd.DataFrame,
    concepts: Optional[Dict[str, str]] = None,   # label -> phrase
    *,
    element_cols: Optional[List[str]] = None,
    prompt_style: str = "composition",   # kept for API compatibility; not used for embedding now
    round_dp: int = 3,                   # kept for API compatibility
    extra_tags: Optional[Dict[str, Union[str, int, float, bool]]] = None,  # kept for API compatibility
    batch_size: int = 64,                # kept for API compatibility (unused)
    device: Optional[str] = None,        # kept for API compatibility (unused)
    max_length: int = 128,               # kept for API compatibility (unused here)
    output_col: str = "Material_Vec",
) -> pd.DataFrame:
    """
    Append Material_Vec and optional Similarity_to_<label> columns.

    MATERIAL EMBEDDING LOGIC (Blablador backend):
    - Identify element columns (intersection with periodic table).
    - Encode each element symbol once via Blablador to get its embedding.
    - For each row, compute the normalized concentration-weighted sum of its present element vectors.
    - L2-normalize each resulting material vector.

    Concept similarities:
    - Encode each concept phrase via Blablador.
    - Compute cosine similarity between material vectors and concept vectors.
    """
    work_df = df.copy()

    # element columns
    if element_cols is None:
        element_cols = [c for c in work_df.columns if c in PERIODIC_TABLE]
    if not element_cols:
        raise ValueError("No element columns found in DataFrame (H..Og).")

    # ---- 1) Encode elements once via Blablador ----
    el_order = list(element_cols)
    el_vecs_dict = _encode_elements_blablador(el_order)
    elem_mat = np.stack([el_vecs_dict[el] for el in el_order], axis=0)  # [E, d], L2-normalized

    # ---- 2) Build weighted sums per row ----
    conc = work_df[el_order].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    conc = np.clip(conc, 0.0, None)

    row_sums = conc.sum(axis=1, keepdims=True)
    has_any = (row_sums.squeeze(-1) > 0.0)
    conc_norm = np.zeros_like(conc, dtype=float)
    conc_norm[has_any] = conc[has_any] / row_sums[has_any]

    mat_vecs = conc_norm @ elem_mat  # [N, d]
    mat_vecs[has_any] = _l2_normalize_rows(mat_vecs[has_any])

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

        concept_mat = _encode_texts_blablador(phrases)  # [C, d], L2-normalized rows

        mat_mask = work_df[output_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
        if mat_mask.any():
            mat_mat = np.stack(work_df.loc[mat_mask, output_col].values, axis=0)  # [M, d]
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
            base = os.path.splitext(fn)[0] + (output_suffix or "_with_qwen")
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
    p = argparse.ArgumentParser(description="Blablador-based material embeddings + concept similarity (CSV -> PKL).")
    p.add_argument("--input_directory", type=str, required=True, help="Input directory")
    p.add_argument("--output_directory", type=str, required=True, help="Output directory")
    p.add_argument("--filename_suffix", type=str, default="_material_system.csv", help="Files ending with this suffix will be processed")
    p.add_argument("--num_workers", type=int, default=2, help="Parallel workers (files)")
    p.add_argument("--prompt_style", type=str, default="composition",
                   choices=["formula","composition","explicit","sentence"], help="Prompt style (kept for compatibility)")
    p.add_argument("--extra_tags", type=str, default="", help='JSON dict of extra tags, e.g. \'{"process":"co-sputtering"}\'')
    p.add_argument("--concepts", type=str, default="", help='JSON dict label->phrase, e.g. \'{"conductivity":"high electrical conductivity","dielectric":"dielectric material"}\'')
    p.add_argument("--batch_size", type=int, default=64, help="Inference batch size (kept for compatibility)")
    p.add_argument("--device", type=str, default="", help="Force device: cpu|mps|cuda (unused; kept for compatibility)")
    p.add_argument("--max_length", type=int, default=128, help="Max tokens per prompt (unused; kept for compatibility)")
    p.add_argument("--output_suffix", type=str, default="_with_qwen", help="Suffix for "
                                                                        "output filenames (before .pkl)")
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