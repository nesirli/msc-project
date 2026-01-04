"""
Step 12: K-mer Dataset Creation AFTER Balancing
Efficiently creates k-mer datasets from only the selected balanced samples
Run independently: snakemake --use-conda --cores 8 -s rules/12_create_kmer_datasets.smk
"""

configfile: "config/config.yaml"

rule create_kmer_datasets:
    input:
        balanced_train="results/features/tree_models/{antibiotic}_train_final.csv",
        balanced_test="results/features/tree_models/{antibiotic}_test_final.csv"
    output:
        train="results/features/deep_models/{antibiotic}_kmer_train_final.npz",
        test="results/features/deep_models/{antibiotic}_kmer_test_final.npz",
        summary="results/features/deep_models/{antibiotic}_kmer_summary.json"
    params:
        processed_dir="data/processed",
        kmer_size=11,
        n_reads_per_sample=10000
    conda:
        "../envs/cnn.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/12_create_kmer_datasets/{antibiotic}.log"
    script:
        "../scripts/12_create_kmer_datasets.py"

rule create_kmer_datasets_all:
    input:
        expand("results/features/deep_models/{antibiotic}_kmer_train_final.npz", antibiotic=config["antibiotics"]),
        expand("results/features/deep_models/{antibiotic}_kmer_test_final.npz", antibiotic=config["antibiotics"]),
        expand("results/features/deep_models/{antibiotic}_kmer_summary.json", antibiotic=config["antibiotics"])