# workflows/material_system_panels.smk
configfile: "../04_plots/configs/config_material_system_panels.yaml"

rule all:
    input:
        config["done_flag"]

rule plot_material_system_panels:
    input:

    output:
        done=config["done_flag"]
    params:
        method_dirs=config["method_dirs"],
        output_dir=config["output_dir"],
        grid_rows=int(config.get("plot", {}).get("grid_shape", [2, 3])[0]),
        grid_cols=int(config.get("plot", {}).get("grid_shape", [2, 3])[1]),
        dpi=int(config.get("plot", {}).get("dpi", 300)),
    script:
        "../scripts/material_system_panels.py"