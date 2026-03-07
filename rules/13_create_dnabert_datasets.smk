"""
Step 13: DNABERT Dataset Creation AFTER Balancing
Efficiently creates DNABERT datasets from only the selected balanced samples
Run independently: snakemake --use-conda --cores 8 -s rules/13_create_dnabert_datasets.smk
"""

configfile: "config/config.yaml"

rule create_dnabert_datasets:
    input:
        balanced_train="results/features/tree_models/{antibiotic}_train_final.csv",
        balanced_test="results/features/tree_models/{antibiotic}_test_final.csv"
    output:
        train="results/features/deep_models/{antibiotic}_dnabert_train_final.npz",
        test="results/features/deep_models/{antibiotic}_dnabert_test_final.npz",
        tokenizer="results/features/deep_models/{antibiotic}_dnabert_tokenizer.pkl",
        summary="results/features/deep_models/{antibiotic}_dnabert_summary.json"
    params:
        processed_dir="data/processed",
        max_seq_len=512,
        n_reads_per_sample=1000
    conda:
        "../envs/transformer.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/13_create_dnabert_datasets/{antibiotic}.log"
    script:
        "../scripts/13_create_dnabert_datasets.py"

rule create_dnabert_datasets_all:
    input:
        expand("results/features/deep_models/{antibiotic}_dnabert_train_final.npz", antibiotic=config["antibiotics"]),
        expand("results/features/deep_models/{antibiotic}_dnabert_test_final.npz", antibiotic=config["antibiotics"]),
        expand("results/features/deep_models/{antibiotic}_dnabert_tokenizer.pkl", antibiotic=config["antibiotics"]),
        expand("results/features/deep_models/{antibiotic}_dnabert_summary.json", antibiotic=config["antibiotics"])