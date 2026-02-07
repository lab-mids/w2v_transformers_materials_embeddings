# Snakefile
configfile: "../04_plots/configs/config_error_plot.yaml"

rule all:
    input:
        config["output_fig"]

rule plot_error_vs_fraction:
    input:
        csv=config["input_csv"]
    output:
        fig=config["output_fig"]
    params:
        # keep params explicit so the plotting script is fully driven by config
        methods=config.get("plot", {}).get("methods", None),
        use_abs_error=bool(config.get("plot", {}).get("use_abs_error", True)),
        annotate_zero_error=bool(config.get("plot", {}).get("annotate_zero_error", True)),
        dpi=int(config.get("plot", {}).get("dpi", 300)),
    script:
        "../scripts/error_fraction_scatter.py"