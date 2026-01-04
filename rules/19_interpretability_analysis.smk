"""
Step 19: Cross-Model Interpretability Analysis
Compares feature importance across all models and identifies consensus features
Run independently: snakemake --use-conda --cores 4 -s rules/19_interpretability_analysis.smk
"""

configfile: "config/config.yaml"

rule interpretability_analysis:
    input:
        # Require all model results to be completed first
        xgboost_results="results/models/xgboost/{antibiotic}_results.json",
        lightgbm_results="results/models/lightgbm/{antibiotic}_results.json",
        cnn_results="results/models/cnn/{antibiotic}_results.json",
        sequence_cnn_results="results/models/sequence_cnn/{antibiotic}_results.json",
        dnabert_results="results/models/dnabert/{antibiotic}_results.json"
    output:
        results="results/interpretability/{antibiotic}_interpretability.json",
        feature_table="results/interpretability/{antibiotic}_consensus_features.csv",
        plots="results/interpretability/plots/{antibiotic}/plots_complete.txt"
    params:
        results_dir="results/models"
    conda:
        "../envs/interpretability.yaml"
    log:
        "logs/19_interpretability_analysis/{antibiotic}.log"
    script:
        "../scripts/19_interpretability_analysis.py"

rule interpretability_all:
    input:
        expand("results/interpretability/{antibiotic}_interpretability.json", antibiotic=config["antibiotics"]),
        expand("results/interpretability/{antibiotic}_consensus_features.csv", antibiotic=config["antibiotics"]),
        expand("results/interpretability/plots/{antibiotic}/plots_complete.txt", antibiotic=config["antibiotics"])