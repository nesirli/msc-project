"""
Step 10: Batch Effect Assessment and Correction
PCA visualization to assess batch effects from Year, Location, Isolation_source, SNP_cluster
Apply ComBat correction if needed
Run independently: snakemake --use-conda --cores 4 -s rules/10_batch_correction.smk
"""

configfile: "config/config.yaml"

rule assess_batch_effects:
    input:
        train="results/features/{antibiotic}_train_selected.csv",
        test="results/features/{antibiotic}_test_selected.csv"
    output:
        train=temp("results/features/{antibiotic}_train_batch_corrected.csv"),
        test=temp("results/features/{antibiotic}_test_batch_corrected.csv"),
        plots="results/batch_correction/{antibiotic}_batch_effects.png",
        report="results/batch_correction/{antibiotic}_batch_report.json"
    conda:
        "../envs/batch_correction.yaml"
    log:
        "logs/10_batch_correction/{antibiotic}.log"
    script:
        "../scripts/10_batch_correction.py"

rule batch_correction_all:
    input:
        expand("results/features/{antibiotic}_train_batch_corrected.csv", antibiotic=config["antibiotics"]),
        expand("results/features/{antibiotic}_test_batch_corrected.csv", antibiotic=config["antibiotics"]),
        expand("results/batch_correction/{antibiotic}_batch_report.json", antibiotic=config["antibiotics"])