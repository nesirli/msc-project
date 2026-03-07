"""
Step 06: AMR Gene Detection with AMRFinderPlus
Runs on assemblies to detect AMR genes (presence/absence)
Run independently: snakemake --use-conda --cores 8 -s rules/06_amr_analysis.smk
"""

import pandas as pd

configfile: "config/config.yaml"

rule update_amrfinder_db:
    output:
        touch("data/reference/.amrfinder_db_updated")
    conda:
        "../envs/amr_analysis.yaml"
    log:
        "logs/06_amr_analysis/update_db.log"
    shell:
        """
        amrfinder -u 2> {log}
        """

rule run_amrfinder:
    input:
        assembly="data/assemblies/{sample}_assembled.fasta",
        db_ready="data/reference/.amrfinder_db_updated"
    output:
        "results/amr/{sample}_amrfinder.tsv"
    conda:
        "../envs/amr_analysis.yaml"
    threads: 4
    log:
        "logs/06_amr_analysis/amrfinder_{sample}.log"
    shell:
        """
        amrfinder -n {input.assembly} \
            --organism Klebsiella_pneumoniae \
            --plus \
            --threads {threads} \
            -o {output} 2> {log}
        """

def get_all_amrfinder_outputs(wildcards):
    import os

    # If ready file exists, skip individual file checks
    if os.path.exists("results/amr/.amr_ready"):
        return []

    try:
        train_df = pd.read_csv("results/features/metadata_train_processed.csv")
        test_df = pd.read_csv("results/features/metadata_test_processed.csv")

        # Use robust extraction like in 02_download.smk
        def extract_valid_runs(df):
            runs = []
            if 'Run' in df.columns:
                for run in df["Run"].astype(str):
                    run = str(run).strip()
                    if run.startswith('SRR') and 'SRR' in run:
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
                    elif ',' in run and 'SRR' in run:
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
            return runs

        train_samples = extract_valid_runs(train_df)
        test_samples = extract_valid_runs(test_df)
        all_samples = list(set(train_samples + test_samples))

        return expand("results/amr/{sample}_amrfinder.tsv", sample=all_samples)
    except Exception as e:
        print(f"Error in get_all_amrfinder_outputs: {e}")
        return []

rule combine_amrfinder:
    input:
        get_all_amrfinder_outputs
    output:
        csv="results/amr/combined_amrfinder.csv",
        ready=touch("results/amr/.amr_ready")
    conda:
        "../envs/amr_analysis.yaml"
    log:
        "logs/06_amr_analysis/combine.log"
    script:
        "../scripts/06_amr_analysis.py"

rule amr_analysis_all:
    input:
        "results/amr/combined_amrfinder.csv",
        "results/amr/.amr_ready"