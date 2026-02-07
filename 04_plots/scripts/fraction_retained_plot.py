import matplotlib
matplotlib.use("Agg")  # safe for headless runs

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FractionRetainedPlotter:
    """
    Plot bar + jittered points for fraction (%) of candidates retained
    in the Pareto subset, grouped by embedding method.

    Expects a CSV with at least:
      - 'method'
      - 'fraction (%)'
    Optionally:
      - 'material_system'
      - 'overpotential (mV)'
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        required = {"method", "fraction (%)"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        self.df["method"] = self.df["method"].astype(str)
        self.df["fraction (%)"] = pd.to_numeric(self.df["fraction (%)"], errors="coerce")
        self.df = self.df.dropna(subset=["fraction (%)"])

    def _preferred_method_order(self, methods_in_data: list[str]) -> list[str]:
        preferred = ["W2V", "MatBERT", "MatBERT_Full", "Qwen", "Qwen_Full"]
        ordered = [m for m in preferred if m in methods_in_data]
        ordered.extend([m for m in methods_in_data if m not in ordered])
        return ordered

    def plot_fraction_bar_points(
        self,
        methods: list[str] | None = None,
        *,
        aggregate: str = "mean",   # or "median"
        jitter_width: float = 0.20,
        outfile: str | Path | None = None,
        dpi: int = 300,
    ) -> tuple[plt.Figure, plt.Axes]:
        df = self.df.copy()
        if methods is not None:
            df = df[df["method"].isin(methods)]

        if df.empty:
            raise ValueError("No data left after filtering by methods.")

        methods_in_data = sorted(df["method"].unique())
        methods_ordered = self._preferred_method_order(methods_in_data)

        # aggregate stats for bars
        agg = aggregate.lower().strip()
        if agg == "median":
            agg_vals = df.groupby("method")["fraction (%)"].median()
        elif agg == "mean":
            agg_vals = df.groupby("method")["fraction (%)"].mean()
        else:
            raise ValueError("aggregate must be 'mean' or 'median'")

        agg_vals = agg_vals.reindex(methods_ordered)

        x = np.arange(len(methods_ordered))
        bar_vals = agg_vals.values

        width = max(6.0, 1.4 * len(methods_ordered))
        height = 4.5
        fig, ax = plt.subplots(figsize=(width, height))

        ax.bar(x, bar_vals, width=0.6, alpha=0.8)

        rng = np.random.RandomState(42)  # reproducible jitter
        for i, m in enumerate(methods_ordered):
            vals = df.loc[df["method"] == m, "fraction (%)"].values
            if vals.size == 0:
                continue
            jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
            ax.plot(
                np.full_like(vals, x[i], dtype=float) + jitter,
                vals,
                linestyle="none",
                marker="o",
                markersize=4,
                alpha=0.7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods_ordered, rotation=0)
        ax.set_ylabel("Fraction of candidates retained, fraction (%)")
        ax.set_xlabel("Embedding method")

        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

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


if __name__ == "__main__":
    if "snakemake" not in globals():
        raise RuntimeError("This script is intended to be run via Snakemake.")

    plotter = FractionRetainedPlotter(snakemake.input["csv"])
    plotter.plot_fraction_bar_points(
        methods=_coerce_methods(snakemake.params["methods"]),
        aggregate=snakemake.params["aggregate"],
        jitter_width=snakemake.params["jitter_width"],
        outfile=snakemake.output["fig"],
        dpi=snakemake.params["dpi"],
    )