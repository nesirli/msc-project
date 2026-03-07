"""
Step 08: Create Feature Matrix
Combine AMR genes + SNPs into binary presence/absence matrix
Create 4 datasets (one per antibiotic) with train/test splits
Run independently: snakemake --use-conda --cores 4 -s rules/08_feature_matrix.smk
"""

configfile: "config/config.yaml"

rule create_feature_matrix:
    input:
        amr="results/amr/combined_amrfinder.csv",
        snp="results/snp/combined_snps.csv",
        amr_ready="results/amr/.amr_ready",
        snp_ready="results/snp/.snp_ready",
        meta_train="results/features/metadata_train_processed.csv",
        meta_test="results/features/metadata_test_processed.csv"
    output:
        expand("results/features/{antibiotic}_train.csv", antibiotic=config["antibiotics"]),
        expand("results/features/{antibiotic}_test.csv", antibiotic=config["antibiotics"]),
        ready=touch("results/features/.feature_matrix_ready")
    params:
        antibiotics=config["antibiotics"],
        outdir="results/features"
    conda:
        "../envs/feature_matrix.yaml"
    log:
        "logs/08_feature_matrix/create_matrix.log"
    script:
        "../scripts/08_feature_matrix.py"

rule feature_matrix_all:
    input:
        expand("results/features/{antibiotic}_train.csv", antibiotic=config["antibiotics"]),
        expand("results/features/{antibiotic}_test.csv", antibiotic=config["antibiotics"]),
        "results/features/.feature_matrix_ready"