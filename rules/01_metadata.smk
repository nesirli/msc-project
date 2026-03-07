"""
Step 01: Process metadata file
Creates processed metadata with representative drugs (amikacin, ciprofloxacin, ceftazidime, meropenem)
Run independently: snakemake --use-conda --cores 1 -s rules/01_metadata.smk
"""

configfile: "config/config.yaml"

rule process_metadata:
    input:
        metadata=config["metadata"]["raw"]
    output:
        train="results/features/metadata_train_processed.csv",
        test="results/features/metadata_test_processed.csv"
    params:
        train_cutoff=config["splits"]["train_cutoff"],
        test_years=config["splits"]["test_years"],
        antibiotics=config["antibiotics"]
    conda:
        "../envs/metadata.yaml"
    log:
        "logs/01_metadata/process_metadata.log"
    script:
        "../scripts/01_metadata.py"

# For independent execution
rule metadata_all:
    input:
        "results/features/metadata_train_processed.csv",
        "results/features/metadata_test_processed.csv"