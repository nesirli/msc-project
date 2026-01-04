"""
Step 20: Ensemble Analysis of AMR prediction models
Combines predictions from all 5 models and evaluates ensemble performance.
Run independently: snakemake --use-conda --cores 4 -s rules/20_ensemble_analysis.smk
"""

configfile: "config/config.yaml"

rule ensemble_analysis:
    input:
        # Require all model results
        xgboost = "results/models/xgboost/{antibiotic}_results.json",
        lightgbm = "results/models/lightgbm/{antibiotic}_results.json", 
        cnn = "results/models/cnn/{antibiotic}_results.json",
        sequence_cnn = "results/models/sequence_cnn/{antibiotic}_results.json",
        dnabert = "results/models/dnabert/{antibiotic}_results.json"
    output:
        "results/ensemble/{antibiotic}_ensemble_analysis.json"
    conda:
        "../envs/interpretability.yaml"
    script:
        "../scripts/20_ensemble_analysis.py"

rule ensemble_analysis_all:
    input:
        expand("results/ensemble/{antibiotic}_ensemble_analysis.json", 
               antibiotic=config["antibiotics"])

rule ensemble_summary:
    input:
        expand("results/ensemble/{antibiotic}_ensemble_analysis.json", 
               antibiotic=config["antibiotics"])
    output:
        summary = "results/ensemble/ensemble_summary_report.json",
        plots = directory("results/ensemble/plots")
    conda:
        "../envs/interpretability.yaml" 
    script:
        "../scripts/21_ensemble_summary.py"