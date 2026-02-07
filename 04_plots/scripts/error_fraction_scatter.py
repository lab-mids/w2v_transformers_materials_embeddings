import matplotlib
matplotlib.use("Agg")  # safe for headless runs

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class ErrorFractionScatterPlotter:
    """
    Scatter plot of error (%) vs fraction (%) for all
    (material system, overpotential, method) combinations.

    Expects a CSV with at least:
      - 'method'
      - 'error (%)'
      - 'fraction (%)'
    Optionally:
      - 'material_system'
      - 'overpotential (mV)'
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required = {"method", "error (%)", "fraction (%)"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        df["method"] = df["method"].astype(str)

        for col in ["error (%)", "fraction (%)"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["error (%)", "fraction (%)"])

        self.df = df

    def _preferred_method_order(self, methods_in_data: list[str]) -> list[str]:
        preferred = ["W2V", "MatBERT", "MatBERT_Full", "Qwen", "Qwen_Full"]
        ordered = [m for m in preferred if m in methods_in_data]
        ordered.extend([m for m in methods_in_data if m not in ordered])
        return ordered

    def plot_error_vs_fraction(
        self,
        methods: list[str] | None = None,
        *,
        use_abs_error: bool = True,
        annotate_zero_error: bool = True,
        outfile: str | Path | None = None,
        dpi: int = 300,
    ) -> tuple[plt.Figure, plt.Axes]:
        df = self.df.copy()
        if methods is not None:
            df = df[df["method"].isin(methods)]

        if df.empty:
            raise ValueError("No data left after filtering by methods.")

        if use_abs_error:
            df["err_plot"] = df["error (%)"].abs()
            y_label = "Relative deviation in best current, |error| (%)"
        else:
            df["err_plot"] = df["error (%)"]
            y_label = "Relative deviation in best current, error (%)"

        methods_in_data = sorted(df["method"].unique())
        methods_ordered = self._preferred_method_order(methods_in_data)

        markers = ["o", "s", "D", "^", "v", "P", "X"]
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
        method_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(methods_ordered)}
        method_to_color = {m: colors[i % len(colors)] for i, m in enumerate(methods_ordered)}

        fig, ax = plt.subplots(figsize=(6.0, 4.8))

        for m in methods_ordered:
            sub = df[df["method"] == m]
            if sub.empty:
                continue
            ax.scatter(
                sub["fraction (%)"],
                sub["err_plot"],
                label=m,
                marker=method_to_marker[m],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.8,
                s=40,
                c=method_to_color[m],
            )

        ax.set_xlabel("Fraction of candidates retained, fraction (%)")
        ax.set_ylabel(y_label)

        x_min = max(0, df["fraction (%)"].min() - 2)
        x_max = min(100, df["fraction (%)"].max() + 2)
        ax.set_xlim(x_min, x_max)

        y_min = 0 if use_abs_error else df["err_plot"].min() - 2
        y_max = df["err_plot"].max() + 2
        ax.set_ylim(y_min, y_max)

        if annotate_zero_error and not use_abs_error:
            ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)

        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(frameon=False, title="Method")
        fig.tight_layout()

        if outfile is not None:
            outfile = Path(outfile)
            outfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outfile, dpi=dpi, bbox_inches="tight")

        return fig, ax


def _coerce_methods(value):
    # Snakemake will pass None or a Python object from config.
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    # allow comma-separated string as a fallback
    s = str(value).strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main(csv_path: str | Path, out_fig: str | Path, methods, use_abs_error: bool, annotate_zero_error: bool, dpi: int):
    plotter = ErrorFractionScatterPlotter(csv_path)
    plotter.plot_error_vs_fraction(
        methods=_coerce_methods(methods),
        use_abs_error=use_abs_error,
        annotate_zero_error=annotate_zero_error,
        outfile=out_fig,
        dpi=dpi,
    )


if __name__ == "__main__":
    # If run via Snakemake, a global "snakemake" object exists.
    if "snakemake" in globals():
        main(
            csv_path=snakemake.input["csv"],
            out_fig=snakemake.output["fig"],
            methods=snakemake.params["methods"],
            use_abs_error=snakemake.params["use_abs_error"],
            annotate_zero_error=snakemake.params["annotate_zero_error"],
            dpi=snakemake.params["dpi"],
        )
    else:
        # Minimal standalone mode (optional)
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--csv", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--methods", default=None, help="Comma-separated list, or omit for all")
        p.add_argument("--use-abs-error", action="store_true")
        p.add_argument("--signed-error", dest="use_abs_error", action="store_false")
        p.set_defaults(use_abs_error=True)
        p.add_argument("--annotate-zero-error", action="store_true")
        p.add_argument("--no-annotate-zero-error", dest="annotate_zero_error", action="store_false")
        p.set_defaults(annotate_zero_error=True)
        p.add_argument("--dpi", type=int, default=300)
        args = p.parse_args()

        main(
            csv_path=args.csv,
            out_fig=args.out,
            methods=args.methods,
            use_abs_error=args.use_abs_error,
            annotate_zero_error=args.annotate_zero_error,
            dpi=args.dpi,
        )