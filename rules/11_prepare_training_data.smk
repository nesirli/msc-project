"""
Step 11: Prepare Training Data for All Models
Calculate class weights and prepare data for consistent class weighting strategy
Run independently: snakemake --use-conda --cores 4 -s rules/11_prepare_training_data.smk
"""

configfile: "config/config.yaml"

rule prepare_training_data:
    input:
        train="results/features/{antibiotic}_train_batch_corrected.csv",
        test="results/features/{antibiotic}_test_batch_corrected.csv",
        train_metadata="results/features/metadata_train_processed.csv",
        test_metadata="results/features/metadata_test_processed.csv"
    output:
        train="results/features/tree_models/{antibiotic}_train_final.csv",
        test="results/features/tree_models/{antibiotic}_test_final.csv",
        summary="results/features/tree_models/{antibiotic}_balance_summary.json"
    params:
        temp_plot=temp("results/features/tree_models/{antibiotic}_balance_plot.png")
    conda:
        "../envs/balanced_datasets.yaml"
    threads: 2
    log:
        "logs/11_prepare_training_data/{antibiotic}.log"
    script:
        "../scripts/11_prepare_training_data.py"

rule prepare_training_data_all:
    input:
        expand("results/features/tree_models/{antibiotic}_train_final.csv", antibiotic=config["antibiotics"]),
        expand("results/features/tree_models/{antibiotic}_test_final.csv", antibiotic=config["antibiotics"]),
        expand("results/features/tree_models/{antibiotic}_balance_summary.json", antibiotic=config["antibiotics"])