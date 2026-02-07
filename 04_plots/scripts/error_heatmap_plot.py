import matplotlib
matplotlib.use("Agg")  # headless-safe

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ErrorHeatmapPlotter:
    """
    Plot a heatmap of relative error (%) for different material systems and methods.

    Expects a CSV with at least:
      - 'material_system'
      - 'overpotential (mV)'
      - 'method'
      - 'error (%)'
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        required = {"material_system", "overpotential (mV)", "method", "error (%)"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        self.df["material_system"] = self.df["material_system"].astype(str)

        # overpotential as integer for labelling
        self.df["overpotential (mV)"] = pd.to_numeric(
            self.df["overpotential (mV)"], errors="coerce"
        )
        self.df = self.df.dropna(subset=["overpotential (mV)"])
        self.df["overpotential (mV)"] = self.df["overpotential (mV)"].round().astype(int)

        # ensure numeric error
        self.df["error (%)"] = pd.to_numeric(self.df["error (%)"], errors="coerce")

        self.df["system_label"] = (
            self.df["material_system"]
            + "\n@ "
            + self.df["overpotential (mV)"].astype(str)
            + " mV"
        )

    def _preferred_method_order(self, methods_in_data: list[str]) -> list[str]:
        preferred = ["W2V", "MatBERT", "MatBERT_Full", "Qwen", "Qwen_Full"]
        ordered = [m for m in preferred if m in methods_in_data]
        ordered.extend([m for m in methods_in_data if m not in ordered])
        return ordered

    def _build_pivot(
        self,
        methods: list[str] | None = None,
        sort_by_error: bool = False,
    ) -> pd.DataFrame:
        df = self.df.copy()

        if methods is not None:
            df = df[df["method"].isin(methods)]

        if df.empty:
            raise ValueError("No data left after filtering by methods.")

        if sort_by_error:
            sys_order = (
                df.groupby("system_label")["error (%)"]
                .mean()
                .sort_values(ascending=False)
                .index
            )
        else:
            sys_order = (
                df.sort_values(["material_system", "overpotential (mV)"])["system_label"]
                .drop_duplicates()
            )

        pivot = df.pivot_table(
            index="system_label",
            columns="method",
            values="error (%)",
            aggfunc="mean",
        )

        pivot = pivot.loc[sys_order]
        pivot = pivot[self._preferred_method_order(list(pivot.columns))]
        return pivot

    def plot_error_heatmap(
        self,
        methods: list[str] | None = None,
        *,
        sort_by_error: bool = False,
        cmap: str = "viridis",
        annotate: bool | None = None,
        max_annot_cells: int = 80,
        outfile: str | Path | None = None,
        dpi: int = 300,
    ) -> tuple[plt.Figure, plt.Axes]:
        pivot = self._build_pivot(methods=methods, sort_by_error=sort_by_error)
        data = pivot.to_numpy(dtype=float)
        n_systems, n_methods = data.shape

        if annotate is None:
            annotate = (n_systems * n_methods) <= max_annot_cells

        mask = np.isnan(data)
        data_masked = np.ma.masked_where(mask, data)

        if np.isfinite(data).any():
            vmax = np.nanmax(np.abs(data))
        else:
            vmax = 1.0
        vmin = 0.0

        base_cmap = plt.get_cmap(cmap).copy()
        base_cmap.set_bad(color="lightgray")

        width = max(6.0, 1.2 * n_methods + 2.0)
        height = max(4.0, 0.45 * n_systems + 1.5)
        fig, ax = plt.subplots(figsize=(width, height))

        im = ax.imshow(
            data_masked,
            aspect="auto",
            interpolation="nearest",
            cmap=base_cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(np.arange(n_methods))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(np.arange(n_systems))
        ax.set_yticklabels(pivot.index, fontsize=9)

        ax.set_xlabel("Embedding method", fontsize=11)
        ax.set_ylabel("Material system, $\\eta$ (mV)", fontsize=11)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Relative deviation in best current, error (%)", fontsize=10)

        if annotate:
            for i in range(n_systems):
                for j in range(n_methods):
                    if not mask[i, j]:
                        ax.text(
                            j,
                            i,
                            f"{data[i, j]:.1f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="black",
                        )

        ax.set_xticks(np.arange(-0.5, n_methods, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_systems, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        fig.tight_layout()

        if outfile is not None:
            outfile = Path(outfile)
            outfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outfile, dpi=dpi, bbox_inches="tight")

        return fig, ax


def _coerce_methods(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    s = str(value).strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _coerce_optional_bool(value):
    # From YAML: null -> None; true/false -> bool; strings also possible.
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"none", "null", ""}:
        return None
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError("annotate must be true/false/null")


if __name__ == "__main__":
    if "snakemake" not in globals():
        raise RuntimeError("This script is intended to be run via Snakemake.")

    plotter = ErrorHeatmapPlotter(snakemake.input["csv"])
    plotter.plot_error_heatmap(
        methods=_coerce_methods(snakemake.params["methods"]),
        sort_by_error=snakemake.params["sort_by_error"],
        cmap=snakemake.params["cmap"],
        annotate=_coerce_optional_bool(snakemake.params["annotate"]),
        max_annot_cells=snakemake.params["max_annot_cells"],
        outfile=snakemake.output["fig"],
        dpi=snakemake.params["dpi"],
    )