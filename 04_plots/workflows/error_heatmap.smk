# workflows/error_heatmap.smk
configfile: "../04_plots/configs/config_error_heatmap.yaml"

rule all:
    input:
        config["output_fig"]

rule plot_error_heatmap:
    input:
        csv=config["input_csv"]
    output:
        fig=config["output_fig"]
    params:
        methods=config.get("plot", {}).get("methods", None),
        sort_by_error=bool(config.get("plot", {}).get("sort_by_error", False)),
        cmap=str(config.get("plot", {}).get("cmap", "viridis")),
        annotate=config.get("plot", {}).get("annotate", None),  # None/true/false
        max_annot_cells=int(config.get("plot", {}).get("max_annot_cells", 80)),
        dpi=int(config.get("plot", {}).get("dpi", 300)),
    script:
        "../scripts/error_heatmap_plot.py"