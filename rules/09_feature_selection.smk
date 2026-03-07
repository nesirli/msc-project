"""
Step 09: Advanced Feature Selection
Multi-stage feature selection: variance filtering, SNP genomic grouping, 
AMR gene prioritization, and DDG-like statistical selection with FDR correction
Run independently: snakemake --use-conda --cores 4 -s rules/09_feature_selection.smk
"""

configfile: "config/config.yaml"

rule advanced_feature_selection:
    input:
        train="results/features/{antibiotic}_train.csv",
        test="results/features/{antibiotic}_test.csv"
    output:
        train="results/features/{antibiotic}_train_selected.csv",
        test="results/features/{antibiotic}_test_selected.csv",
        importance="results/features/{antibiotic}_feature_importance.csv",
        report="results/features/{antibiotic}_selection_report.json"
    params:
        n_features=config["feature_selection"]["n_features"],
        alpha=config["feature_selection"]["alpha"],
        method=config["feature_selection"]["method"],
        variance_threshold=0.01,
        sparsity_threshold=0.95,
        frequency_threshold=2,
        snp_window_size=50000,
        correlation_threshold=0.8
    conda:
        "../envs/feature_selection.yaml"
    log:
        "logs/09_feature_selection/{antibiotic}.log"
    script:
        "../scripts/09_feature_selection.py"

rule feature_selection_all:
    input:
        expand("results/features/{antibiotic}_train_selected.csv", antibiotic=config["antibiotics"]),
        expand("results/features/{antibiotic}_test_selected.csv", antibiotic=config["antibiotics"]),
        expand("results/features/{antibiotic}_selection_report.json", antibiotic=config["antibiotics"])