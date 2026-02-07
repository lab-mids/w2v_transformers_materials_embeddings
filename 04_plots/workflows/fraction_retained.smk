# workflows/fraction_retained.smk
configfile: "../04_plots/configs/config_fraction_retained.yaml"

rule all:
    input:
        config["output_fig"]

rule plot_fraction_retained:
    input:
        csv=config["input_csv"]
    output:
        fig=config["output_fig"]
    params:
        methods=config.get("plot", {}).get("methods", None),
        aggregate=str(config.get("plot", {}).get("aggregate", "mean")),
        jitter_width=float(config.get("plot", {}).get("jitter_width", 0.20)),
        dpi=int(config.get("plot", {}).get("dpi", 300)),
    script:
        "../scripts/fraction_retained_plot.py"