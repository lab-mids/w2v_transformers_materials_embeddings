import inspect
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from gensim.models import KeyedVectors, Word2Vec


# --- Periodic table symbols (1-118) ---
_PERIODIC_TABLE_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
_PERIODIC_TABLE_SET = set(_PERIODIC_TABLE_SYMBOLS)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def _has_token(model, token: str) -> bool:
    # Supports Word2Vec (model.wv) or KeyedVectors
    if hasattr(model, "wv"):
        model = model.wv
    if hasattr(model, "key_to_index"):
        return token in model.key_to_index
    try:
        return token in model
    except Exception:
        try:
            _ = model[token]
            return True
        except Exception:
            return False


def _get_vec(model, token: str) -> np.ndarray:
    if hasattr(model, "wv"):
        model = model.wv
    return np.asarray(model[token], dtype=float)


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".txt":
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _system_label_from_filename(path: str) -> str:
    """
    'Ag_Pd_material_system.csv' -> 'Ag-Pd'
    'Ag_Pd_Ru_material_system.csv' -> 'Ag-Pd-Ru'
    Fallback: stem before '_material_system'
    """
    base = os.path.basename(path)
    stem = base[:-4] if base.endswith(".csv") else base
    stem = stem.replace("_material_system", "")
    return "-".join([p for p in stem.split("_") if p])


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def _load_model(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        return KeyedVectors.load(path, mmap="r")
    except Exception:
        return Word2Vec.load(path)


@dataclass
class MaterialEmbeddingVisualizer:
    """
    Visualize embeddings for ONE OR MULTIPLE material system files together.

    New features:
      1) Takes output_dir and can save a PDF with bbox_inches='tight' and dpi=600.
      2) Takes multiple material files and plots them together.
         Each system gets its own color (legend shows system names like 'Ag-Pd').

    Notes:
      - Points are colored by material system (not performance metric in this version).
      - t-SNE is fit on all materials across all systems + the two property words.
    """
    w2v_model: object
    input_files: Union[str, List[str]]  # single file or list of files
    output_dir: Optional[str] = None

    # Optional output naming
    output_basename: str = "materials_embedding"

    # Words to overlay
    word_dielectric: str = "dielectric"
    word_conductivity: str = "conductivity"

    # t-SNE config
    tsne_perplexity: float = 30.0
    tsne_learning_rate: Union[str, float] = "auto"
    tsne_n_iter: int = 1500
    tsne_random_state: int = 42

    # Preprocessing
    standardize_before_tsne: bool = True

    # Column handling
    drop_position_cols: Tuple[str, ...] = ("x", "y")

    # Plot styling
    material_alpha: float = 0.65
    material_size: float = 14.0
    word_size: float = 70.0
    dielectric_color: str = "tab:blue"
    conductivity_color: str = "tab:orange"

    def _drop_xy(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_set = {c.lower() for c in self.drop_position_cols}
        keep_cols = [c for c in df.columns if c.lower() not in drop_set]
        return df[keep_cols].copy()

    def _element_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c in _PERIODIC_TABLE_SET]

    def _compute_material_vectors(
        self, df: pd.DataFrame, element_cols: List[str]
    ) -> Tuple[np.ndarray, List[int]]:
        missing_elements = set()
        skipped_rows = 0

        dim = None
        for el in element_cols:
            if _has_token(self.w2v_model, el):
                dim = _get_vec(self.w2v_model, el).shape[0]
                break
        if dim is None:
            raise ValueError("Could not find any element tokens from your file inside the provided w2v model.")

        mat_vecs = []
        kept = []

        for idx, row in df[element_cols].iterrows():
            v = np.zeros(dim, dtype=float)
            used_any = False

            for el in element_cols:
                conc = row[el]
                if pd.isna(conc):
                    continue
                try:
                    w = float(conc)
                except Exception:
                    continue
                if w == 0.0:
                    continue

                if _has_token(self.w2v_model, el):
                    v += _get_vec(self.w2v_model, el) * w
                    used_any = True
                else:
                    missing_elements.add(el)

            if used_any:
                mat_vecs.append(v)
                kept.append(idx)
            else:
                skipped_rows += 1

        if missing_elements:
            warnings.warn(
                f"{len(missing_elements)} element(s) were not found in the w2v model and were skipped: "
                + ", ".join(sorted(list(missing_elements))[:20])
                + (" ..." if len(missing_elements) > 20 else "")
            )
        if skipped_rows > 0:
            warnings.warn(f"Skipped {skipped_rows} row(s) that had no usable element concentrations.")

        return np.vstack(mat_vecs), kept

    def _tsne_2d(self, X: np.ndarray) -> np.ndarray:
        X_in = X
        if self.standardize_before_tsne:
            X_in = StandardScaler().fit_transform(X_in)

        n = X_in.shape[0]
        perplexity = float(self.tsne_perplexity)
        if n <= 3:
            raise ValueError("Need at least 4 samples to run t-SNE meaningfully.")
        if perplexity >= n:
            perplexity = max(2.0, min(30.0, (n - 1) / 3.0))
            warnings.warn(f"Adjusted t-SNE perplexity to {perplexity:.2f} because n_samples={n}.")

        tsne_sig = inspect.signature(TSNE.__init__)
        kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            learning_rate=self.tsne_learning_rate,
            init="pca",
            random_state=self.tsne_random_state,
        )
        if "n_iter" in tsne_sig.parameters:
            kwargs["n_iter"] = self.tsne_n_iter
        elif "max_iter" in tsne_sig.parameters:
            kwargs["max_iter"] = self.tsne_n_iter
        else:
            warnings.warn("Could not find n_iter/max_iter in TSNE signature; using sklearn defaults.")

        return TSNE(**kwargs).fit_transform(X_in)

    def _save_pdf(self, fig: plt.Figure, basename: Optional[str] = None) -> str:
        if not self.output_dir:
            raise ValueError("output_dir is None. Provide output_dir to save figures.")
        os.makedirs(self.output_dir, exist_ok=True)

        if basename is None:
            basename = self.output_basename

        outpath = os.path.join(self.output_dir, f"{basename}.pdf")
        fig.savefig(outpath, format="pdf", bbox_inches="tight", dpi=600)
        return outpath

    def plot(
        self,
        figsize: Tuple[float, float] = (12, 5),
        save: bool = False,
        basename: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        # Normalize input files to list
        files = [self.input_files] if isinstance(self.input_files, str) else list(self.input_files)
        if not files:
            raise ValueError("No input_files provided.")

        # Check property words
        if not _has_token(self.w2v_model, self.word_dielectric):
            raise ValueError(f"Word '{self.word_dielectric}' not found in the w2v model.")
        if not _has_token(self.w2v_model, self.word_conductivity):
            raise ValueError(f"Word '{self.word_conductivity}' not found in the w2v model.")

        v_dielectric = _get_vec(self.w2v_model, self.word_dielectric)
        v_conductivity = _get_vec(self.w2v_model, self.word_conductivity)
        l2_dist = float(np.linalg.norm(v_dielectric - v_conductivity))

        # Load + embed each system; concatenate for joint t-SNE
        all_vecs = []
        system_labels = []
        per_system_counts = []

        for sys_id, fp in enumerate(files):
            df = _read_table(fp)
            df_embed = self._drop_xy(df)

            element_cols = self._element_columns(df_embed)
            if not element_cols:
                raise ValueError(
                    f"No element columns detected in {os.path.basename(fp)} "
                    f"(expected columns like Fe, O, Ag...)."
                )

            vecs, _ = self._compute_material_vectors(df_embed, element_cols)

            all_vecs.append(vecs)
            label = _system_label_from_filename(fp)
            system_labels.append(label)
            per_system_counts.append(vecs.shape[0])

        mat_vecs = np.vstack(all_vecs)

        # t-SNE on all materials + two words
        X_for_tsne = np.vstack([mat_vecs, v_dielectric[None, :], v_conductivity[None, :]])
        emb2d = self._tsne_2d(X_for_tsne)

        mat_2d = emb2d[:-2]
        die_2d = emb2d[-2]
        con_2d = emb2d[-1]

        # Similarities
        sim_die = np.array([_cosine_sim(v, v_dielectric) for v in mat_vecs], dtype=float)
        sim_con = np.array([_cosine_sim(v, v_conductivity) for v in mat_vecs], dtype=float)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        ax1, ax2 = axes

        # Use matplotlib default color cycle for systems
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not color_cycle:
            color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

        # Plot each system separately so legend reflects system color
        start = 0
        for sys_id, label in enumerate(system_labels):
            n = per_system_counts[sys_id]
            end = start + n
            c = color_cycle[sys_id % len(color_cycle)]

            # (a) t-SNE scatter
            ax1.scatter(
                mat_2d[start:end, 0],
                mat_2d[start:end, 1],
                s=self.material_size,
                c=c,
                alpha=self.material_alpha,
                edgecolors="none",
                label=label,
            )

            # (b) similarity scatter
            ax2.scatter(
                sim_die[start:end],
                sim_con[start:end],
                s=self.material_size,
                c=c,
                alpha=self.material_alpha,
                edgecolors="none",
                label=label,
            )

            start = end

        # dielectric (bigger, more transparent, plotted first)
        ax1.scatter(
            die_2d[0], die_2d[1],
            s=self.word_size * 1.6,
            c=self.dielectric_color,
            alpha=0.45,
            marker="o",
            edgecolors="k",
            linewidths=0.8,
            zorder=10,
            label=self.word_dielectric,
        )

        # conductivity (smaller, less transparent, plotted after)
        ax1.scatter(
            con_2d[0], con_2d[1],
            s=self.word_size * 0.9,
            c=self.conductivity_color,
            alpha=0.85,
            marker="X",
            edgecolors="k",
            linewidths=0.8,
            zorder=11,
            label=self.word_conductivity,
        )

        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.set_title("t-SNE embedding")
        ax1.legend(frameon=False, loc="best")
        ax1.text(
            0.98,
            0.02,
            f"L2(dielectric, conductivity) = {l2_dist:.3f}",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="black",
        )

        ax2.set_xlabel(f"Similarity to '{self.word_dielectric}'")
        ax2.set_ylabel(f"Similarity to '{self.word_conductivity}'")
        ax2.set_title("Cosine similarity distribution")
        ax2.legend(frameon=False, loc="best")

        # Panel labels outside top-left
        ax1.text(-0.12, 1.05, "(a)", transform=ax1.transAxes,
                 fontsize=13, fontweight="bold", va="bottom", ha="left", clip_on=False)
        ax2.text(-0.12, 1.05, "(b)", transform=ax2.transAxes,
                 fontsize=13, fontweight="bold", va="bottom", ha="left", clip_on=False)

        if save:
            outpath = self._save_pdf(fig, basename=basename)
            print(f"Saved: {outpath}")

        return fig, axes


def _coerce_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    s = str(value).strip()
    if not s:
        return []
    return [v.strip() for v in s.split(",") if v.strip()]


def main(
    model_path: str,
    input_files: Union[str, List[str]],
    output_dir: str,
    output_basename: str,
    done_flag: str,
    word_dielectric: str = "dielectric",
    word_conductivity: str = "conductivity",
    figsize: Tuple[float, float] = (12, 5),
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: Union[str, float] = "auto",
    tsne_n_iter: int = 1500,
    tsne_random_state: int = 42,
    standardize_before_tsne: bool = True,
    material_alpha: float = 0.65,
    material_size: float = 14.0,
    word_size: float = 70.0,
    dielectric_color: str = "tab:blue",
    conductivity_color: str = "tab:orange",
) -> None:
    model = _load_model(model_path)

    viz = MaterialEmbeddingVisualizer(
        w2v_model=model,
        input_files=input_files,
        output_dir=output_dir,
        output_basename=output_basename,
        word_dielectric=word_dielectric,
        word_conductivity=word_conductivity,
        tsne_perplexity=float(tsne_perplexity),
        tsne_learning_rate=tsne_learning_rate,
        tsne_n_iter=int(tsne_n_iter),
        tsne_random_state=int(tsne_random_state),
        standardize_before_tsne=bool(standardize_before_tsne),
        material_alpha=float(material_alpha),
        material_size=float(material_size),
        word_size=float(word_size),
        dielectric_color=dielectric_color,
        conductivity_color=conductivity_color,
    )

    viz.plot(figsize=figsize, save=True)

    done_path = os.path.abspath(done_flag)
    os.makedirs(os.path.dirname(done_path), exist_ok=True)
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("OK\n")


if __name__ == "__main__":
    if "snakemake" not in globals():
        raise RuntimeError("This script is intended to be run via Snakemake.")

    plot_cfg = snakemake.params.get("plot", {})
    figsize = plot_cfg.get("figsize", [12, 5])
    if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        figsize = (float(figsize[0]), float(figsize[1]))
    else:
        figsize = (12.0, 5.0)

    main(
        model_path=snakemake.params["model_path"],
        input_files=_coerce_list(snakemake.params.get("input_files")),
        output_dir=snakemake.params["output_dir"],
        output_basename=snakemake.params.get("output_basename", "materials_embedding"),
        done_flag=snakemake.output["done"],
        word_dielectric=snakemake.params.get("word_dielectric", "dielectric"),
        word_conductivity=snakemake.params.get("word_conductivity", "conductivity"),
        figsize=figsize,
        tsne_perplexity=plot_cfg.get("tsne_perplexity", 30.0),
        tsne_learning_rate=plot_cfg.get("tsne_learning_rate", "auto"),
        tsne_n_iter=plot_cfg.get("tsne_n_iter", 1500),
        tsne_random_state=plot_cfg.get("tsne_random_state", 42),
        standardize_before_tsne=plot_cfg.get("standardize_before_tsne", True),
        material_alpha=plot_cfg.get("material_alpha", 0.65),
        material_size=plot_cfg.get("material_size", 14.0),
        word_size=plot_cfg.get("word_size", 70.0),
        dielectric_color=plot_cfg.get("dielectric_color", "tab:blue"),
        conductivity_color=plot_cfg.get("conductivity_color", "tab:orange"),
    )
