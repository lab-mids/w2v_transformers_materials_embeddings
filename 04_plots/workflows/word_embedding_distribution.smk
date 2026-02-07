# workflows/word_embedding_distribution.smk
configfile: "../04_plots/configs/config_word_embedding_distribution.yaml"

rule all:
    input:
        config["done_flag"]

rule plot_word_embedding_distribution:
    output:
        done=config["done_flag"]
    params:
        model_path=config["model_path"],
        input_files=config.get("input_files", []),
        output_dir=config["output_dir"],
        output_basename=config.get("output_basename", "materials_embedding"),
        word_dielectric=config.get("word_dielectric", "dielectric"),
        word_conductivity=config.get("word_conductivity", "conductivity"),
        plot=config.get("plot", {}),
    script:
        "../scripts/word_embedding_distribution_plot.py"
